"""Light cross-encoder reranking (the "+R" methods).

Re-scores the top-n fused candidates with a small cross-encoder that reads the
question and unit together, and puts them in score order ahead of the rest. It
separates the effect of reranking from that of pruning: B2's Provence does both,
with a much larger model.
"""

import time

from ragsplit.units import Unit


class CrossEncoderReranker:
    def __init__(self, cfg: dict):
        import torch
        from sentence_transformers import CrossEncoder

        torch.set_num_threads(cfg.get("n_threads", 8))
        self.cfg = cfg
        self.top_n = cfg["top_n"]
        self.model = CrossEncoder(cfg["model"], device="cpu", cache_folder=cfg.get("model_cache"))

    def rerank(self, question: str, ranked: list[Unit]) -> tuple[list[Unit], float]:
        """Return (reranked units, seconds spent)."""
        t0 = time.perf_counter()
        head, tail = ranked[: self.top_n], ranked[self.top_n:]
        if not head:
            return ranked, 0.0
        scores = self.model.predict([(question, u.search_text) for u in head],
                                    batch_size=self.cfg.get("batch_size", 32), show_progress_bar=False)
        order = sorted(range(len(head)), key=lambda i: -float(scores[i]))
        return [head[i] for i in order] + tail, time.perf_counter() - t0
