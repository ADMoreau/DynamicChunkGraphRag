"""EM and F1 with the official SQuAD v1.1 normalization, plus evidence hit."""

from collections import Counter
import re
import string

import numpy as np

_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace (official script)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_single(prediction: str, truth: str) -> float:
    p, t = normalize_answer(prediction).split(), normalize_answer(truth).split()
    common = Counter(p) & Counter(t)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(t)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, truths: list[str]) -> float:
    return float(max(normalize_answer(prediction) == normalize_answer(t) for t in truths))


def f1(prediction: str, truths: list[str]) -> float:
    return max(f1_single(prediction, t) for t in truths)


def evidence_hit(context: str, gold_sentence_text: str) -> bool:
    """The gold sentence appears whole inside the packed context, compared after
    SQuAD normalization so re-decoded text (e.g. Provence output) still matches."""
    return normalize_answer(gold_sentence_text) in normalize_answer(context)


def answer_in_context(context: str, truths: list[str]) -> bool:
    return any(t in context for t in truths)


def answer_located(parts: list[tuple[str, int, str]], gold_pid: str, prefix: str,
                   truths: list[str]) -> bool:
    """The answer reaches the reader from its own paragraph.

    `parts` are the packed units as (paragraph id, char position, text). The gold
    paragraph's parts, put back in paragraph order with their title prefix removed,
    must contain a gold answer after SQuAD normalization. An answer split across two
    adjacent packed pieces still counts; the same words from another paragraph do not.
    This replaces "the gold sentence is in the context" once units are smaller than
    sentences."""
    gold = sorted((pos, text) for pid, pos, text in parts if pid == gold_pid)
    joined = normalize_answer(" ".join(t[len(prefix):] if t.startswith(prefix) else t for _, t in gold))
    return any(n and n in joined for n in map(normalize_answer, truths))


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10_000, seed: int = 0) -> dict:
    """Mean of a - b over paired questions, with a 95% bootstrap interval."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def summarize_rows(rows: list[dict]) -> dict:
    lat = np.array([r["latency_s"] for r in rows])
    return {
        "n": len(rows),
        "em": float(np.mean([r["em"] for r in rows])),
        "f1": float(np.mean([r["f1"] for r in rows])),
        "evidence_hit": float(np.mean([r["evidence_hit"] for r in rows])),
        "answer_in_context": float(np.mean([r["answer_in_context"] for r in rows])),
        "input_tokens_mean": float(np.mean([r["input_tokens"] for r in rows])),
        "context_tokens_mean": float(np.mean([r["context_tokens"] for r in rows])),
        "latency_mean_s": float(lat.mean()),
        "latency_p50_s": float(np.percentile(lat, 50)),
        "latency_p95_s": float(np.percentile(lat, 95)),
    }
