"""GPU cost benchmark on Modal: does the light reranker's advantage over Provence survive on a GPU?

Runs on the inputs exported by configs/m6_gpu_inputs.yaml (results/m6_gpu_inputs.json), so
no corpus or index is needed in the cloud. For each stage it measures
  - online latency: one question at a time (batch 1), as a live service sees it
  - batched throughput: the whole question set at once, as GPU-seconds per question
  - GPU energy from NVML, total and above idle
and replays every logged reader context through vLLM for F1 per method and budget.
Provence and the reranker run in an image pinned to the CPU runs' library versions; the
reader runs in its own vLLM image. Retrieval (BM25 + small embedder, ~30 ms on CPU, the
same order for every method) is not part of this benchmark.

This costs GPU time: ask before running it. First download the models (CPU only):
  modal run modal_app/gpu_bench.py::download
then
  modal run modal_app/gpu_bench.py --gpu L4
"""

from contextlib import nullcontext
import json
from pathlib import Path
import time

import modal

PROVENCE = "naver/provence-reranker-debertav3-v1"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
READERS = "Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct"
INPUTS = Path(__file__).resolve().parent.parent / "results" / "m6_gpu_inputs.json"
# which front stage each reader method uses
FRONT_OF = {"B2": "provence-n20", "B2-n5": "provence-n5", "B2-n10": "provence-n10",
            "B3+R": "rerank-B3", "L-k1+R": "rerank-L-k1"}

hf_cache = modal.Volume.from_name("ragsplit-hf-cache", create_if_missing=True)
NLTK = ("python -c \"import nltk; [nltk.download(p, download_dir='/root/nltk_data') "
        "for p in ('punkt', 'punkt_tab')]\"")
front_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "transformers==5.18.0", "sentence-transformers==6.1.0",
                 "nltk==3.10.3", "numpy==2.5.3", "nvidia-ml-py")
    .run_commands(NLTK)
    .env({"HF_HOME": "/hf", "NLTK_DATA": "/root/nltk_data"})
)
reader_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "nvidia-ml-py")
    # FlashInfer's sampler JIT-compiles kernels and needs nvcc, which the slim image lacks
    .env({"HF_HOME": "/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("ragsplit")
)
app = modal.App("ragsplit-gpu-bench")


class Energy:
    """GPU 0's cumulative energy counter (NVML), in joules."""

    def __init__(self):
        import pynvml

        pynvml.nvmlInit()
        self.nvml, self.handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)

    def joules(self) -> float:
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0

    def idle_watts(self, seconds: float = 10.0) -> float:
        e0, t0 = self.joules(), time.perf_counter()
        time.sleep(seconds)
        return (self.joules() - e0) / (time.perf_counter() - t0)


def measure(fn, energy: Energy, sync) -> tuple[float, float]:
    """(seconds, joules) for one call, GPU work included."""
    sync()
    e0, t0 = energy.joules(), time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t0, energy.joules() - e0


def block_stats(times: list[float], joules: float, seconds: float, idle_w: float) -> dict:
    """Per-question stats for a block of batch-1 calls (energy is read per block:
    NVML updates too coarsely for single millisecond-scale calls)."""
    import numpy as np

    t = np.array(times)
    n = len(t)
    return {"n": n, "median_ms": float(np.median(t) * 1e3), "mean_ms": float(t.mean() * 1e3),
            "p95_ms": float(np.percentile(t, 95) * 1e3), "j_per_q": joules / n,
            "j_per_q_above_idle": (joules - idle_w * seconds) / n}


@app.function(image=front_image, volumes={"/hf": hf_cache}, timeout=600)
def download(readers: str = READERS) -> None:
    """Fetch every model into the shared volume on a CPU container (cheap)."""
    from huggingface_hub import snapshot_download

    for repo in [PROVENCE, RERANKER, *readers.split(",")]:
        print(repo, snapshot_download(repo))
    hf_cache.commit()


@app.function(image=front_image, gpu="L4", volumes={"/hf": hf_cache}, timeout=3600)
def bench_front(inputs: dict, warmup: int) -> dict:
    """Provence (top 5/10/20 paragraphs) and the MiniLM reranker, fp32 and fp16 autocast
    (not bf16: both models hand scores to NumPy, which has no bfloat16)."""
    import torch
    from sentence_transformers import CrossEncoder
    from transformers import AutoModel

    energy = Energy()
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "idle_w": energy.idle_watts(), "online": {}, "batched": {}}
    provence = AutoModel.from_pretrained(PROVENCE, trust_remote_code=True).to("cuda").eval()
    reranker = CrossEncoder(RERANKER, device="cuda")
    qs = list(inputs["questions"].values())
    sync = torch.cuda.synchronize

    def prov(questions, contexts):
        provence.process(questions, contexts, title=None, threshold=0.1, batch_size=64,
                         enable_warnings=False)

    def rerank(pairs):
        reranker.predict(pairs, batch_size=256, show_progress_bar=False)

    stages = {f"provence-n{n}": lambda q, n=n: prov([q["question"]], [q["provence_paras"][:n]])
              for n in (5, 10, 20)}
    stages |= {f"rerank-{m}": lambda q, m=m: rerank([(q["question"], c) for c in q["rerank_candidates"][m]])
               for m in qs[0]["rerank_candidates"]}
    batched = {f"provence-n{n}": lambda n=n: prov([q["question"] for q in qs],
                                                  [q["provence_paras"][:n] for q in qs])
               for n in (5, 10, 20)}
    batched |= {f"rerank-{m}": lambda m=m: rerank([(q["question"], c) for q in qs
                                                   for c in q["rerank_candidates"][m]])
                for m in qs[0]["rerank_candidates"]}

    with torch.inference_mode():
        for dtype in ("fp32", "fp16"):
            ctx = torch.autocast("cuda", dtype=torch.float16) if dtype == "fp16" else nullcontext()
            with ctx:
                for name, fn in stages.items():
                    for q in qs[:warmup]:
                        fn(q)
                    times, e_tot, t_tot = [], 0.0, 0.0
                    sync()
                    e0, t0 = energy.joules(), time.perf_counter()
                    for q in qs:
                        dt, _ = measure(lambda: fn(q), energy, sync)
                        times.append(dt)
                    t_tot, e_tot = time.perf_counter() - t0, energy.joules() - e0
                    out["online"][f"{name}|{dtype}"] = block_stats(times, e_tot, t_tot, out["idle_w"])
                for name, fn in batched.items():
                    fn()  # warm-up at full batch shape
                    dt, de = measure(fn, energy, sync)
                    out["batched"][f"{name}|{dtype}"] = {
                        "ms_per_q": dt / len(qs) * 1e3, "j_per_q": de / len(qs),
                        "j_per_q_above_idle": (de - out["idle_w"] * dt) / len(qs)}
                print(f"front {dtype} done", flush=True)
    return out


@app.function(image=reader_image, gpu="L4", volumes={"/hf": hf_cache}, timeout=3600)
def bench_reader(inputs: dict, model: str, warmup: int, online_budgets: list[int]) -> dict:
    """Replay every logged context through `model` with vLLM (no prefix caching):
    batched for F1 and GPU-seconds, and one at a time at `online_budgets` for latency."""
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    from ragsplit.metrics import exact_match, f1
    from ragsplit.reader import SYSTEM_PROMPT, clean_answer, user_prompt

    energy = Energy()
    idle_w = energy.idle_watts()
    llm = LLM(model=model, dtype="bfloat16", enable_prefix_caching=False, seed=0,
              gpu_memory_utilization=0.85, max_model_len=4096)
    params = SamplingParams(temperature=0.0, max_tokens=32)
    qmap = inputs["questions"]

    def conv(key):
        _, _, qid = key.split("|")
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(qmap[qid]["question"], inputs["contexts"][key])}]

    groups: dict[str, list[str]] = {}
    for key in inputs["contexts"]:
        m, b, _ = key.split("|")
        groups.setdefault(f"{m}|{b}", []).append(key)
    out = {"model": model, "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0),
           "idle_w": idle_w, "batched": {}, "online": {}}
    llm.chat([conv(k) for k in groups[next(iter(groups))][:warmup]], params, use_tqdm=False)

    for group, keys in sorted(groups.items()):
        res = {}
        dt, de = measure(lambda: res.setdefault("o", llm.chat([conv(k) for k in keys], params, use_tqdm=False)),
                         energy, lambda: None)
        answers = [clean_answer(o.outputs[0].text) for o in res["o"]]
        golds = [qmap[k.split("|")[2]]["gold"] for k in keys]
        out["batched"][group] = {
            "n": len(keys), "ms_per_q": dt / len(keys) * 1e3, "j_per_q": de / len(keys),
            "j_per_q_above_idle": (de - idle_w * dt) / len(keys),
            "f1": sum(f1(a, g) for a, g in zip(answers, golds)) / len(keys),
            "em": sum(exact_match(a, g) for a, g in zip(answers, golds)) / len(keys),
            "input_tokens": sum(len(o.prompt_token_ids) for o in res["o"]) / len(keys),
            "output_tokens": sum(len(o.outputs[0].token_ids) for o in res["o"]) / len(keys)}

    for group, keys in sorted(groups.items()):
        if int(group.split("|")[1]) not in online_budgets:
            continue
        times = []
        e0, t0 = energy.joules(), time.perf_counter()
        for k in keys:
            dt, _ = measure(lambda: llm.chat([conv(k)], params, use_tqdm=False), energy, lambda: None)
            times.append(dt)
        out["online"][group] = block_stats(times, energy.joules() - e0, time.perf_counter() - t0, idle_w)
    print(f"reader {model} done", flush=True)
    return out


def summarize(front: dict, readers: list[dict], dtype: str = "fp16") -> str:
    """Markdown tables: front stages, then per reader end-to-end per method and budget."""
    lines = [f"GPU: {front['gpu']}, idle {front['idle_w']:.0f} W, front dtype {dtype}", "",
             "| front stage | online median ms | online J/q above idle | batched ms/q | batched J/q above idle |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in sorted({k.split("|")[0] for k in front["online"]}):
        on, ba = front["online"][f"{name}|{dtype}"], front["batched"][f"{name}|{dtype}"]
        lines.append(f"| {name} | {on['median_ms']:.1f} | {on['j_per_q_above_idle']:.2f} | "
                     f"{ba['ms_per_q']:.2f} | {ba['j_per_q_above_idle']:.3f} |")
    for r in readers:
        lines += ["", f"reader {r['model']} (vLLM {r['vllm']})", "",
                  "| method | budget | F1 | input tok | online ms (front + reader) | batched ms/q (front + reader) | batched J/q above idle |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for group in sorted(r["batched"], key=lambda g: (int(g.split("|")[1]), g)):
            m, b = group.split("|")
            fr = FRONT_OF[m]
            ba, fb = r["batched"][group], front["batched"][f"{fr}|{dtype}"]
            online = ""
            if group in r["online"]:
                online = f"{front['online'][f'{fr}|{dtype}']['median_ms'] + r['online'][group]['median_ms']:.0f}"
            lines.append(f"| {m} | {b} | {ba['f1']:.3f} | {ba['input_tokens']:.0f} | {online} | "
                         f"{fb['ms_per_q'] + ba['ms_per_q']:.1f} | "
                         f"{fb['j_per_q_above_idle'] + ba['j_per_q_above_idle']:.2f} |")
    return "\n".join(lines)


@app.local_entrypoint()
def main(gpu: str = "L4", readers: str = READERS, warmup: int = 5, online_budgets: str = "128,256"):
    inputs = json.loads(INPUTS.read_text())
    budgets = [int(b) for b in online_budgets.split(",")]
    front_call = bench_front.with_options(gpu=gpu).spawn(inputs, warmup)
    reader_calls = [bench_reader.with_options(gpu=gpu).spawn(inputs, m, warmup, budgets)
                    for m in readers.split(",")]
    stem = INPUTS.parent / f"m6_gpu_bench_{gpu}"
    # collect each container on its own and save as soon as it arrives, so one failure
    # cannot throw away the others' results
    results = {}
    for name, call in [("front", front_call)] + [(f"reader{i}", c) for i, c in enumerate(reader_calls)]:
        try:
            results[name] = call.get()
        except Exception as e:  # noqa: BLE001 - report and keep the rest
            print(f"{name} failed: {e!r}")
            results[name] = None
        stem.with_suffix(".partial.json").write_text(json.dumps(results, indent=2) + "\n")
    front = results["front"]
    reader_out = [r for k, r in results.items() if k.startswith("reader") and r]
    if front is None or not reader_out:
        raise SystemExit(f"incomplete benchmark; partial results in {stem}.partial.json")
    stem.with_suffix(".json").write_text(json.dumps({"front": front, "readers": reader_out}, indent=2) + "\n")
    table = summarize(front, reader_out)
    stem.with_suffix(".md").write_text(table + "\n")
    print(table)
    print(f"\nwrote {stem}.json and {stem}.md")
