"""Same / related / different judgments for candidate node pairs, with vLLM on Modal.

Reads candidate pairs (JSONL from the `candidates` stage), asks each model for a one-word
verdict per pair (prompt and rules in src/ragsplit/adjudicate.py), writes the verdicts, and,
when reference labels exist, scores agreement with them (prompt-example pairs left out).

This costs GPU time: ask before running it. Download models first (CPU only):
  modal run modal_app/llm_judge.py::download --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct
Compare models on the 250-pair reference set:
  modal run modal_app/llm_judge.py --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct --gpus L4,L40S
Judge every candidate with the chosen model:
  modal run modal_app/llm_judge.py --pairs data/cache/merge/candidates_<key>.jsonl --models <model> --gpus <gpu>
"""

import json
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parent.parent
hf_cache = modal.Volume.from_name("ragsplit-hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm")
         # FlashInfer's sampler JIT-compiles kernels and needs nvcc, which the slim image lacks;
         # vLLM's PyTorch/Triton sampler needs no compiler (and greedy one-word answers gain nothing)
         .env({"HF_HOME": "/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
         .add_local_python_source("ragsplit"))
app = modal.App("ragsplit-llm-judge")


@app.function(image=image, volumes={"/hf": hf_cache}, timeout=3600)
def download(models: str) -> None:
    """Fetch model weights into the shared volume on a CPU container (cheap)."""
    from huggingface_hub import snapshot_download

    for repo in models.split(","):
        print(repo, snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
    hf_cache.commit()


@app.function(image=image, gpu="L4", volumes={"/hf": hf_cache}, timeout=3600)
def judge(pairs: list[dict], model: str) -> dict:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    from ragsplit.adjudicate import SYSTEM, parse_label, prompt

    t0 = time.perf_counter()
    llm = LLM(model=model, dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=0.9,
              enable_prefix_caching=True, tensor_parallel_size=torch.cuda.device_count())
    load_s = time.perf_counter() - t0
    convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(p["a"], p["b"])}]
             for p in pairs]
    t0 = time.perf_counter()
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False)
    judge_s = time.perf_counter() - t0
    return {"model": model, "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0),
            "load_s": load_s, "judge_s": judge_s,
            "verdicts": [{"id": p["id"], "raw": o.outputs[0].text, "label": parse_label(o.outputs[0].text)}
                         for p, o in zip(pairs, outs)]}


@app.local_entrypoint()
def main(pairs: str = "results/m9_adjudicate_sample.jsonl", models: str = "Qwen/Qwen2.5-7B-Instruct",
         gpus: str = "L4", labels: str = "results/m9_adjudicate_labels.jsonl", out: str = "results/m9_judge",
         min_cos: float = 0.0, chunks: int = 1):
    """`min_cos` drops embedding / fragment candidates below that cosine (exact and alias pairs
    are always kept); `chunks` splits the pairs over that many containers run in parallel
    (each pays its own model load) to stay under the per-container timeout."""
    from ragsplit.adjudicate import is_prompt_example, score

    rows = [json.loads(line) for line in (ROOT / pairs).open()]
    rows = [r for r in rows if r.get("cos") is None or r["source"] == "exact" or r["cos"] >= min_cos]
    print(f"{len(rows)} pairs to judge (min cosine {min_cos}), in {chunks} chunk(s)")
    models_l, gpus_l = models.split(","), gpus.split(",")
    if len(gpus_l) == 1:
        gpus_l *= len(models_l)
    parts = [rows[i::chunks] for i in range(chunks)]
    calls = [[judge.with_options(gpu=g).spawn(part, m) for part in parts] for m, g in zip(models_l, gpus_l)]
    label_path = ROOT / labels
    gold = {}
    if label_path.exists():
        gold = {d["id"]: d["label"] for d in map(json.loads, label_path.open())}
        gold = {r["id"]: gold[r["id"]] for r in rows if r["id"] in gold and not is_prompt_example(r)}
    for group, gpu in zip(calls, gpus_l):
        results = [c.get() for c in group]
        res = {**results[0], "load_s": max(r["load_s"] for r in results),
               "judge_s": max(r["judge_s"] for r in results),
               "verdicts": [v for r in results for v in r["verdicts"]]}
        tag = res["model"].split("/")[-1]
        path = ROOT / f"{out}_{tag}.jsonl"
        with path.open("w") as f:
            for v in res["verdicts"]:
                f.write(json.dumps(v) + "\n")
        print(f"\n{res['model']} on {res['gpu']} (vLLM {res['vllm']}): load {res['load_s']:.0f} s, "
              f"{len(rows)} pairs in {res['judge_s']:.1f} s -> {path}")
        if gold:
            s = score({v["id"]: v["label"] for v in res["verdicts"]}, gold)
            print(f"  vs reference ({s['n']} pairs): accuracy {s['accuracy']:.2f}, false merges {s['false_merges']}, "
                  f"unparsed {s['unparsed']}")
            for lab in ("SAME", "RELATED", "DIFFERENT"):
                d = s[lab]
                fmt = lambda x: "-" if x is None else f"{x:.2f}"
                print(f"  {lab:9s} precision {fmt(d['precision'])} recall {fmt(d['recall'])} "
                      f"(predicted {d['predicted']}, reference {d['gold']})")
