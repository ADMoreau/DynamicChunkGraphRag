"""BM25 + dense retrieval fused with reciprocal rank fusion.

Indexes are cached under data/cache/, keyed by a hash of the unit texts and
the model, and every cache hit or build is printed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from ragsplit.units import Unit

# bge v1.5 recommends this instruction on queries (not on passages).
QUERY_INSTRUCTION = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}


def units_key(units: list[Unit]) -> str:
    h = hashlib.sha256()
    for u in units:
        h.update(u.id.encode() + b"\0" + u.search_text.encode() + b"\0")
    return h.hexdigest()[:16]


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[int]:
    """Reciprocal rank fusion of ranked lists of unit indices (rank 1 = best)."""
    weights = weights or [1.0] * len(rankings)
    score: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for r, idx in enumerate(ranking, start=1):
            score[idx] = score.get(idx, 0.0) + w / (k + r)
    return sorted(score, key=lambda i: (-score[i], i))


class Retriever:
    def __init__(self, units: list[Unit], cfg: dict, cache_dir: str | Path):
        self.units = units
        self.cfg = cfg
        self.depth = cfg["depth"]
        self.rrf_k = cfg["rrf_k"]
        key = units_key(units)
        model_tag = cfg["dense_model"].replace("/", "__")
        self.cache = Path(cache_dir) / f"index_{key}"
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.cache / "meta.json").write_text(json.dumps(
            {"n_units": len(units), "first_ids": [u.id for u in units[:3]]}) + "\n")

        self._bm25 = self._load_bm25()
        self._encoder = None
        self._emb = self._load_dense(self.cache / f"dense_{model_tag}.npy")

    # --- BM25 ---
    def _load_bm25(self):
        import bm25s

        path = self.cache / "bm25"
        if path.exists():
            print(f"[cache] BM25 index {path}")
            return bm25s.BM25.load(str(path))
        print(f"[build] BM25 index over {len(self.units)} units -> {path}")
        tokens = bm25s.tokenize([u.search_text for u in self.units], stopwords="en", show_progress=False)
        bm25 = bm25s.BM25()
        bm25.index(tokens, show_progress=False)
        bm25.save(str(path))
        return bm25

    def bm25_rank(self, queries: list[str]) -> list[list[int]]:
        import bm25s

        k = min(self.depth, len(self.units))
        q_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
        docs, scores = self._bm25.retrieve(q_tokens, k=k, show_progress=False)
        # bm25s returns zero-score filler for queries with few matching docs; drop it.
        return [[int(d) for d, s in zip(row, srow) if s > 0] for row, srow in zip(docs, scores)]

    # --- dense ---
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self.cfg["dense_model"], device="cpu", cache_folder=self.cfg.get("model_cache"))
        return self._encoder

    def _load_dense(self, path: Path) -> np.ndarray:
        if path.exists():
            print(f"[cache] dense embeddings {path}")
            return np.load(path)
        print(f"[build] embedding {len(self.units)} units with {self.cfg['dense_model']} -> {path}")
        emb = self.encoder().encode(
            [u.search_text for u in self.units], batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False)
        np.save(path, emb.astype(np.float32))
        return emb

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        prefix = QUERY_INSTRUCTION.get(self.cfg["dense_model"], "")
        return self.encoder().encode(
            [prefix + q for q in queries], batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False)

    def dense_rank(self, q_emb: np.ndarray) -> list[list[int]]:
        sims = q_emb @ self._emb.T
        k = min(self.depth, len(self.units))
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        return [[int(i) for i in row[np.argsort(-sims[r, row])]] for r, row in enumerate(top)]

    # --- fused ---
    def search(self, queries: list[str], prior=None, prior_weight: float = 0.0,
               prior_min_sim: float = 0.0) -> list[list[int]]:
        """Fused ranking (unit indices, best first) for each query.

        With a prior (B4), its ranking (units whose best attached past question
        has similarity >= prior_min_sim) joins the fusion as a third list with
        weight `prior_weight`."""
        bm = self.bm25_rank(queries)
        dn = self.dense_rank(self.embed_queries(queries))
        if prior is None or prior_weight == 0:
            return [rrf([b, d], k=self.rrf_k)[: self.depth] for b, d in zip(bm, dn)]
        plain = self.encoder().encode(queries, batch_size=32, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        out = []
        for b, d, qe in zip(bm, dn, plain):
            pr = prior.rank(qe, prior_min_sim)[: self.depth]
            out.append(rrf([b, d, pr], k=self.rrf_k, weights=[1.0, 1.0, prior_weight])[: self.depth])
        return out
