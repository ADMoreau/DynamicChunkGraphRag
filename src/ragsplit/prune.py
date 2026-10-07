"""Provence wrapper (B2): rerank retrieved paragraphs and prune sentences per question.

Provence runs on the raw paragraph text with no title (title=None); the title
prefix is added back afterwards, as for every other unit. Provence decodes its
output from DeBERTa tokens, which silently drops characters outside its vocabulary
("2½" becomes "2"), so we work out which of its (nltk) sentences it kept and send
the original text of those sentences instead.

Results are cached per question under data/cache/provence/, including the
measured pruning latency, so a resumed run logs the real cost.
"""

import json
import os
from pathlib import Path
import time

from ragsplit.units import Unit, title_prefix


def kept_original_text(context: str, pruned: str, decode_sentence) -> str:
    """Original text of the sentences Provence kept.

    Provence splits `context` with nltk, encodes each sentence (with a leading
    space after the first) and decodes the kept ones back to `pruned`. We redo
    the split, decode each sentence the same way, and keep it if its decoded
    form appears in `pruned` (in order).
    """
    import nltk

    kept, pos, search_from = [], 0, 0
    for j, sent in enumerate(nltk.sent_tokenize(context)):
        start = context.find(sent, pos)
        if start == -1:  # nltk normalized the text; fall back to the decoded output
            return pruned
        pos = start + len(sent)
        dec = decode_sentence(sent, first=(j == 0)).strip()
        if not dec:
            continue
        at = pruned.find(dec, search_from)
        if at != -1:
            kept.append(sent)
            search_from = at + len(dec)
    return " ".join(kept)


class ProvencePruner:
    def __init__(self, cfg: dict, cache_dir: str | Path, cache_tag: str):
        import nltk

        nltk.data.path.insert(0, os.path.abspath(cfg["nltk_data"]))
        self.cfg = cfg
        self.top_n = cfg["top_n"]
        self.threshold = cfg["threshold"]
        model_tag = Path(cfg["model_path"]).name
        # use_titles: pass each paragraph's title and always keep it, as in the Provence paper
        self.use_titles = bool(cfg.get("use_titles", False))
        self.cache_path = (Path(cache_dir) / "provence" /
                           f"{model_tag}_th{self.threshold}_n{self.top_n}{'_titles' if self.use_titles else ''}_{cache_tag}.jsonl")
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            with self.cache_path.open() as f:
                for line in f:
                    d = json.loads(line)
                    self.cache[d["qid"]] = d
            print(f"[cache] Provence outputs for {len(self.cache)} questions in {self.cache_path}")
        self._model = None
        self._tokenizer = None

    def decode_sentence(self, sent: str, first: bool) -> str:
        """Round-trip one sentence through Provence's tokenizer, as Provence does."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg["model_path"], trust_remote_code=True)
        ids = self._tokenizer.encode(("" if first else " ") + sent, add_special_tokens=False)
        return self._tokenizer.decode(ids, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)

    def model(self):
        if self._model is None:
            import torch
            from transformers import AutoModel

            torch.set_num_threads(self.cfg.get("n_threads", 8))
            self._model = AutoModel.from_pretrained(
                self.cfg["model_path"], trust_remote_code=True).eval()
        return self._model

    def prune(self, qid: str, question: str, ranked: list[Unit], para_by_id: dict) -> tuple[list[Unit], float]:
        """Return (pruned units reranked by Provence score, best first; pruning seconds).

        Units Provence prunes to nothing are dropped.
        """
        candidates = ranked[: self.top_n]
        hit = self.cache.get(qid)
        if hit is None or hit["unit_ids"] != [u.id for u in candidates]:
            t0 = time.perf_counter()
            titles = [[para_by_id[u.paragraph_id].title.replace("_", " ") for u in candidates]] if self.use_titles else None
            out = self.model().process(
                [question], [[para_by_id[u.paragraph_id].context for u in candidates]],
                title=titles, always_select_title=self.use_titles, threshold=self.threshold,
                batch_size=self.cfg.get("batch_size", 4), enable_warnings=False)
            hit = {
                "qid": qid,
                "unit_ids": [u.id for u in candidates],
                "scores": [float(s) for s in out["reranking_score"][0]],
                "pruned": out["pruned_context"][0],
                "latency_s": time.perf_counter() - t0,
            }
            self.cache[qid] = hit
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                f.write(json.dumps(hit) + "\n")

        order = sorted(range(len(candidates)), key=lambda i: -hit["scores"][i])
        pruned = []
        for i in order:
            if not hit["pruned"][i].strip():
                continue
            u = candidates[i]
            text = kept_original_text(
                para_by_id[u.paragraph_id].context, hit["pruned"][i], self.decode_sentence)
            if not text:
                continue
            title = para_by_id[u.paragraph_id].title
            pruned.append(Unit(u.id, "pruned", u.paragraph_id, u.sent_lo, u.sent_hi,
                               title_prefix(title) + text))
        return pruned, hit["latency_s"]
