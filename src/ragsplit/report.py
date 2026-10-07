"""Summaries and plots from per-question JSONL logs."""

from collections import defaultdict
import json
from pathlib import Path

from ragsplit.metrics import summarize_rows


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows: list[dict], by_bucket: bool = True) -> list[dict]:
    """One summary per (method, budget[, bucket]); bucket "all" is always included."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["budget"], "all")].append(r)
        if by_bucket:
            groups[(r["method"], r["budget"], r["bucket"])].append(r)
    out = []
    for (method, budget, bucket), rs in sorted(groups.items()):
        out.append({"method": method, "budget": budget, "bucket": bucket, **summarize_rows(rs)})
    return out


def format_table(summary: list[dict], bucket: str = "all") -> str:
    lines = ["| method | budget | n | EM | F1 | evidence hit | answer in ctx | input tok | p50 s | p95 s |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in summary:
        if s["bucket"] != bucket:
            continue
        lines.append(
            f"| {s['method']} | {s['budget']} | {s['n']} | {s['em']:.3f} | {s['f1']:.3f} | "
            f"{s['evidence_hit']:.3f} | {s['answer_in_context']:.3f} | {s['input_tokens_mean']:.0f} | "
            f"{s['latency_p50_s']:.1f} | {s['latency_p95_s']:.1f} |")
    return "\n".join(lines)


def plot_costs(summary: list[dict], path: Path, title: str) -> None:
    """Evidence hit vs input tokens, and F1 vs mean latency per question."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    by_method = defaultdict(list)
    for s in summary:
        if s["bucket"] == "all":
            by_method[s["method"]].append(s)
    for method, ss in sorted(by_method.items()):
        ss.sort(key=lambda s: s["budget"])
        axes[0].plot([s["input_tokens_mean"] for s in ss], [s["evidence_hit"] for s in ss],
                     marker="o", label=method)
        axes[1].plot([s["latency_mean_s"] for s in ss], [s["f1"] for s in ss], marker="o", label=method)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("mean input tokens (incl. prompt)")
    axes[0].set_ylabel("evidence hit")
    axes[1].set_xlabel("mean latency per question, s (CPU, incl. pruning)")
    axes[1].set_ylabel("F1")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_f1_vs_budget(summary: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, x_key, x_label in ((axes[0], "budget", "context budget (tokens)"),
                               (axes[1], "input_tokens_mean", "mean input tokens (incl. prompt)")):
        by_method = defaultdict(list)
        for s in summary:
            if s["bucket"] == "all":
                by_method[s["method"]].append(s)
        for method, ss in sorted(by_method.items()):
            ss.sort(key=lambda s: s["budget"])
            ax.plot([s[x_key] for s in ss], [s["f1"] for s in ss], marker="o", label=method)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("F1")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
