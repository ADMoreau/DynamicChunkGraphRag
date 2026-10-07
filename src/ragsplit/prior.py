"""B4 past-question prior (query association).

Every unit is attached to the reference (past) questions whose gold sentence it
contains. At query time units are ranked by the cosine similarity between the
new question and their most similar attached past question; units with no
attached questions, or whose best attached question is below a similarity
threshold tau, are absent from this list. The list joins the fusion with weight w.
"""

import hashlib
from pathlib import Path

import numpy as np

from ragsplit.data import Question
from ragsplit.units import Unit


class QueryPrior:
    def __init__(self, units: list[Unit], ref_questions: list[Question], encoder, cache_dir: str | Path,
                 model_name: str):
        by_key: dict[tuple[str, int], list[int]] = {}
        for i, q in enumerate(ref_questions):
            if q.gold_sent is not None:
                by_key.setdefault((q.paragraph_id, q.gold_sent), []).append(i)
        self.unit_idx: list[int] = []
        self.attached: list[np.ndarray] = []
        for ui, u in enumerate(units):
            att = sorted({qi for key in u.covers() for qi in by_key.get(key, [])})
            if att:
                self.unit_idx.append(ui)
                self.attached.append(np.array(att))
        self.n_attached_units = len(self.unit_idx)

        texts = [q.question for q in ref_questions]
        h = hashlib.sha256(("\0".join(texts) + model_name).encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"past_question_emb_{h}.npy"
        if path.exists():
            print(f"[cache] past-question embeddings {path}")
            self.emb = np.load(path)
        else:
            print(f"[build] embedding {len(texts)} past questions -> {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            # question-to-question similarity: no retrieval instruction on either side
            self.emb = encoder.encode(texts, batch_size=64, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            np.save(path, self.emb)

    def rank(self, q_emb: np.ndarray, min_sim: float = 0.0) -> list[int]:
        """Unit indices with attachments, best first, for one plain question embedding.

        Units whose most similar attached past question is below `min_sim` are left
        out, so the prior only votes when a close past question exists (tuned on
        the tuning set: with bge-small, the nearest past question is on the same
        paragraph 15% of the time below 0.7 similarity and over 80% above 0.9)."""
        idx, scores = self.scored(q_emb)
        return [i for i, s in zip(idx, scores) if s >= min_sim]

    def scored(self, q_emb: np.ndarray) -> tuple[list[int], list[float]]:
        """All attached units, best first, with their best past-question similarity."""
        sims = self.emb @ q_emb
        scores = np.array([sims[att].max() for att in self.attached])
        order = np.argsort(-scores, kind="stable")
        return [self.unit_idx[i] for i in order], scores[order].tolist()
