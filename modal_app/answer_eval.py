"""Answer-level evaluation on SyllabusQA: does better evidence delivery give better answers?

Reads the contexts each method sent the reader (results/<name>_rows.jsonl from a probe run
with eval.dump_contexts), has a reader model answer every question from its context, then
has a judge model grade each answer against the reference answer written by course staff.
Reports accuracy per method and budget, split by covered / uncovered questions, plus a
budget-adaptive policy (one method below a token threshold, another at or above it).

This costs GPU time: ask before running it.
  modal run modal_app/answer_eval.py --rows results/s7_answers_rows.jsonl
"""

from collections import defaultdict
import glob
import json
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parent.parent
hf_cache = modal.Volume.from_name("ragsplit-hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm")
         .env({"HF_HOME": "/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"}))
app = modal.App("ragsplit-answer-eval")

READER_SYSTEM = ("You are a course assistant. Answer the student's question using only the syllabus "
                 "excerpt. Be brief: one or two sentences. If the excerpt does not contain the answer, "
                 "reply exactly: Not stated in the excerpt.")
# v1 (4 Oct) was too strict: it marked 24 of 100 correct answers wrong, mostly ones that added
# detail ("Late assignments will not be accepted" for reference "No"). v2 states the rules from
# docs/same_related_different.md, and is checked against Claude's 100 hand labels.
JUDGE_SYSTEM_V1 = ("You grade answers from a course assistant against a reference answer written by the "
                   "course staff. The response is CORRECT if it states the key facts of the reference (the "
                   "wording may differ) and does not contradict it; extra harmless detail is fine. It is "
                   "INCORRECT if it misses or contradicts a key fact, or says the information is not "
                   "available when the reference gives it. Reply with one word: CORRECT or INCORRECT.")
JUDGE_SYSTEM = (
    "You grade a course assistant's answer against the reference answer from the course staff.\n"
    "CORRECT: the response gives the same answer as the reference on what the question asks. For a "
    "yes/no question, the same yes or no, stated or clearly implied (\"Late work will not be accepted\" "
    "matches \"No\"). Otherwise, the same key facts: numbers, dates, names, policies. Wording may "
    "differ. Extra detail is fine as long as it does not contradict the reference. If the question asks "
    "for \"some\" of something, a correct subset is enough.\n"
    "INCORRECT: the response contradicts the reference, gives a different fact, says the information is "
    "not stated or not available, answers a different question, or leaves out something the question "
    "explicitly asks for (one of two things asked, or items of a list asked for in full).\n"
    "Reply with one word: CORRECT or INCORRECT.")


def reader_prompt(item: dict) -> list[dict]:
    return [{"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": f"Syllabus excerpt:\n{item['context']}\n\nQuestion: {item['question']}\nAnswer:"}]


def judge_prompt(item: dict) -> list[dict]:
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Question: {item['question']}\nReference answer: {item['reference']}\n"
                                        f"Response: {item['response']}\nVerdict:"}]


def run_chat(items: list[dict], model: str, build, max_tokens: int) -> tuple[list[str], dict]:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    t0 = time.perf_counter()
    llm = LLM(model=model, dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=0.9,
              enable_prefix_caching=True)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    outs = llm.chat([build(it) for it in items], SamplingParams(temperature=0.0, max_tokens=max_tokens),
                    use_tqdm=False)
    return [o.outputs[0].text.strip() for o in outs], {
        "model": model, "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0),
        "load_s": load_s, "run_s": time.perf_counter() - t0}


@app.function(image=image, gpu="L4", volumes={"/hf": hf_cache}, timeout=3600)
def answer(items: list[dict], model: str) -> tuple[list[str], dict]:
    return run_chat(items, model, reader_prompt, 96)


@app.function(image=image, gpu="L40S", volumes={"/hf": hf_cache}, timeout=3600)
def judge(items: list[dict], model: str) -> tuple[list[str], dict]:
    return run_chat(items, model, judge_prompt, 4)


@app.local_entrypoint()
def main(rows: str = "results/s7_answers_rows.jsonl", reader: str = "Qwen/Qwen2.5-7B-Instruct",
         grader: str = "Qwen/Qwen2.5-14B-Instruct", adaptive: str = "E-k2+R<256,B1+R",
         regrade: str = "", labels: str = "results/s7_judge_check_labels.json", tag: str = ""):
    """With --regrade <judged.jsonl>, skip answering and only re-grade the saved responses."""
    if regrade:
        items = [json.loads(line) for line in (ROOT / regrade).open()]
        print(f"re-grading {len(items)} saved answers")
        rinfo = {"reused": regrade}
    else:
        raw = {r["id"]: r for path in glob.glob(str(ROOT / "data/raw/syllabusqa/*.json")) for r in json.load(open(path))}
        data = [json.loads(line) for line in (ROOT / rows).open()]
        items = [{**r, "question": raw[r["qid"]]["question"], "reference": raw[r["qid"]]["answer"].strip()} for r in data]
        print(f"{len(items)} answers to generate and grade ({len({r['qid'] for r in data})} questions)")
        responses, rinfo = answer.remote(items, reader)
        for it, resp in zip(items, responses):
            it["response"] = resp
    verdicts, jinfo = judge.remote(items, grader)
    for it, v in zip(items, verdicts):
        it["correct"] = v.upper().startswith("CORRECT")
        it["verdict_raw"] = v
        it.pop("context", None)
    out = ROOT / (regrade.replace(".jsonl", f"{tag or '_v2'}.jsonl") if regrade
                  else rows.replace("_rows.jsonl", "_judged.jsonl"))
    if (ROOT / labels).exists():
        lab = {(d["method"], d["budget"], d["qid"]): d["label"] == "CORRECT" for d in json.load(open(ROOT / labels))}
        got = {(it["method"], it["budget"], it["qid"]): it["correct"] for it in items}
        keys = [k for k in lab if k in got]
        agree = sum(got[k] == lab[k] for k in keys)
        print(f"judge vs Claude's {len(keys)} hand labels: agreement {agree}/{len(keys)}; "
              f"correct marked wrong {sum(lab[k] and not got[k] for k in keys)}, "
              f"wrong marked correct {sum(got[k] and not lab[k] for k in keys)}")
    with out.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"reader {rinfo}\njudge {jinfo}")

    acc = defaultdict(list)
    for it in items:
        for bucket in ("all", it["bucket"]):
            acc[(it["method"], it["budget"], bucket)].append(it["correct"])
    # budget-adaptive policy: first method below the threshold, second at or above it
    low, high = adaptive.split(",")
    low_m, threshold = low.split("<")
    for it in items:
        pick = low_m if it["budget"] < int(threshold) else high
        if it["method"] == pick:
            for bucket in ("all", it["bucket"]):
                acc[(f"adaptive ({adaptive})", it["budget"], bucket)].append(it["correct"])
    methods = sorted({k[0] for k in acc})
    budgets = sorted({k[1] for k in acc})
    for bucket in ("all", "covered", "uncovered"):
        print(f"\nanswer accuracy (judge: {grader}), {bucket} questions")
        print("| method | " + " | ".join(str(b) for b in budgets) + " |")
        print("| --- |" + " ---: |" * len(budgets))
        for m in methods:
            print(f"| {m} | " + " | ".join(f"{sum(acc[(m, b, bucket)]) / max(1, len(acc[(m, b, bucket)])):.3f}"
                                       for b in budgets) + " |")
    print(f"\nwrote {out}")
