"""Past / future / tuning question split and overlap buckets.

All splits are by question, deterministic for a seed, and independent of the
order question ids are passed in.
"""

import random

GoldKey = tuple[str, int]  # (paragraph_id, sentence index)


def split_past_future(qids: list[str], past_frac: float, seed: int) -> tuple[list[str], list[str]]:
    ids = sorted(set(qids))
    random.Random(seed).shuffle(ids)
    n_past = round(len(ids) * past_frac)
    return sorted(ids[:n_past]), sorted(ids[n_past:])


def split_tuning(past: list[str], tuning_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Split past questions into (tuning, build). Units and priors used while
    tuning are built from `build` only."""
    ids = sorted(set(past))
    random.Random(f"{seed}:tuning").shuffle(ids)
    n_tune = round(len(ids) * tuning_frac)
    return sorted(ids[:n_tune]), sorted(ids[n_tune:])


def pilot_slice(future: list[str], size: int, seed: int) -> list[str]:
    ids = sorted(set(future))
    if size >= len(ids):
        return ids
    return sorted(random.Random(f"{seed}:pilot").sample(ids, size))


def overlap_buckets(
    eval_gold: dict[str, GoldKey | None],
    reference_gold: list[GoldKey | None],
) -> dict[str, str]:
    """covered: the eval question's gold sentence was gold for at least one
    reference (past) question; uncovered otherwise; unmapped if it has none."""
    seen = {g for g in reference_gold if g is not None}
    return {
        qid: "unmapped" if g is None else ("covered" if g in seen else "uncovered")
        for qid, g in eval_gold.items()
    }
