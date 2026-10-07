"""Key layer: small machine-facing keys that point to reader-facing sentences.

Keys are what retrieval matches on; they need not read well. Each key points to
one sentence, and retrieval returns sentences in the order of their best-ranked
key, so packing, reranking and metrics work on sentences exactly as for B3.
NOTE: this prototype adds keys *alongside* sentences, which is not the target design: in the
target design a fragment split replaces its sentence (before / fragment / after), as for
paragraphs (see CLAUDE.md). Kept for the record of the 3 Oct 2026 probe.
Key types (from the spaCy dependency parse, no LLM calls):
  sentence  the sentence itself (B3 as a key set)
  clause    each clause head's tokens, minus nested clauses
  triple    compact subject-verb-object and verb-preposition-object strings
Every key keeps the article-title prefix, which disambiguates short keys.
"""

import json
from pathlib import Path

from ragsplit.data import Paragraph
from ragsplit.units import Unit, title_prefix

KEY_TYPES = ("sentence", "clause", "triple")
CLAUSE_DEPS = {"ROOT", "ccomp", "advcl", "relcl", "conj", "xcomp", "acl", "csubj", "parataxis"}
SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "expl"}
OBJ_DEPS = {"dobj", "attr", "oprd", "acomp", "dative"}
MOD_DEPS = {"compound", "amod", "nummod", "flat", "poss", "npadvmod", "quantmod"}
MIN_CLAUSE_TOKENS = 3


def compact(tok) -> str:
    """A noun or verb with its tight modifiers (compounds, numbers, adjectives)."""
    keep, todo = {tok.i}, [tok]
    while todo:
        for c in todo.pop().children:
            if c.dep_ in MOD_DEPS:
                keep.add(c.i)
                todo.append(c)
    return " ".join(t.text for t in tok.doc if t.i in keep)


def clause_keys(sent) -> list[str]:
    """Split a parsed sentence into clauses: every token belongs to its nearest
    clause head (a verb, or the root); clauses shorter than MIN_CLAUSE_TOKENS
    content tokens are folded into the root clause."""
    heads = {t.i for t in sent if t.dep_ == "ROOT" or (t.dep_ in CLAUSE_DEPS and t.pos_ in ("VERB", "AUX"))}
    owner = {}
    for t in sent:
        h = t
        while h.i not in heads and h.head.i != h.i:
            h = h.head
        owner[t.i] = h.i if h.i in heads else sent.root.i
    groups: dict[int, list] = {}
    for t in sent:
        if not t.is_punct:
            groups.setdefault(owner[t.i], []).append(t)
    root = sent.root.i
    merged: dict[int, list] = {root: []}
    for h, toks in groups.items():
        content = [t for t in toks if not t.is_stop]
        merged.setdefault(h if (h == root or len(content) >= MIN_CLAUSE_TOKENS) else root, []).extend(toks)
    return [" ".join(t.text for t in sorted(toks, key=lambda t: t.i))
            for h, toks in sorted(merged.items()) if toks]


def triple_keys(sent) -> list[str]:
    """Compact relation strings: subject-verb-object, subject-verb-preposition-object,
    and subject-verb when the verb has no object. A relative or participle clause
    with no subject of its own takes the noun it modifies as subject."""
    out = []
    for v in sent:
        if v.pos_ not in ("VERB", "AUX"):
            continue
        subj = next((c for c in v.children if c.dep_ in SUBJ_DEPS and c.pos_ != "PRON"), None)
        if subj is None and v.dep_ in ("relcl", "acl"):
            subj = v.head
        objs = [compact(c) for c in v.children if c.dep_ in OBJ_DEPS]
        for p in (c for c in v.children if c.dep_ in ("prep", "agent")):
            objs += [f"{p.text} {compact(o)}" for o in p.children if o.dep_ == "pobj"]
        s = compact(subj) if subj is not None else ""
        if objs:
            out += [" ".join(x for x in (s, v.text, o) if x) for o in objs]
        elif s:
            out.append(f"{s} {v.text}")
    return list(dict.fromkeys(out))  # dedupe, keep order


def parse_keys(sentences: list[Unit], paragraphs: list[Paragraph], spacy_model: str,
               cache_path: Path) -> dict[str, dict[str, list[str]]]:
    """Clause and triple keys (without title) for every sentence unit, cached."""
    if cache_path.exists():
        print(f"[cache] parsed keys {cache_path}")
        return {d["id"]: d["keys"] for d in map(json.loads, cache_path.open())}
    from ragsplit.data import load_spacy

    nlp = load_spacy(spacy_model)
    para = {p.id: p for p in paragraphs}
    texts = [para[u.paragraph_id].context[slice(*para[u.paragraph_id].sents[u.sent_lo])]
             for u in sentences]
    out = {}
    for u, doc in zip(sentences, nlp.pipe(texts, batch_size=256)):
        keys = {"clause": [], "triple": []}
        for sent in doc.sents:
            keys["clause"] += clause_keys(sent)
            keys["triple"] += triple_keys(sent)
        out[u.id] = keys
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        for uid, keys in out.items():
            f.write(json.dumps({"id": uid, "keys": keys}) + "\n")
    print(f"[build] parsed keys for {len(out)} sentences -> {cache_path}")
    return out


def build_keys(sentences: list[Unit], paragraphs: list[Paragraph], types: list[str],
               parsed: dict[str, dict[str, list[str]]],
               fine_only: set[tuple[str, int]] | None = None) -> tuple[list[Unit], list[int]]:
    """Key units and, for each, the index of the sentence it points to.

    `fine_only` limits clause and triple keys to those (paragraph, sentence) pairs
    (the learned variant); sentence keys are always made for every sentence.
    """
    bad = set(types) - set(KEY_TYPES)
    if bad:
        raise ValueError(f"unknown key types {sorted(bad)}")
    title = {p.id: title_prefix(p.title) for p in paragraphs}
    keys, targets = [], []
    for i, u in enumerate(sentences):
        fine = fine_only is None or (u.paragraph_id, u.sent_lo) in fine_only
        for t in types:
            if t == "sentence":
                texts = [u.text]
            elif fine:
                texts = [title[u.paragraph_id] + k for k in parsed[u.id][t]]
            else:
                continue
            for j, text in enumerate(texts):
                keys.append(Unit(f"{u.id}{t[0]}{j:02d}", f"key-{t}", u.paragraph_id,
                                 u.sent_lo, u.sent_hi, text))
                targets.append(i)
    return keys, targets


def first_hits(ranked_keys: list[int], targets: list[int], depth: int) -> list[int]:
    """Sentence indices in the order of their best-ranked key, up to `depth`."""
    seen, out = set(), []
    for j in ranked_keys:
        t = targets[j]
        if t not in seen:
            seen.add(t)
            out.append(t)
            if len(out) == depth:
                break
    return out


class KeyedRetriever:
    """Retriever over keys that returns the sentences they point to.

    The key index is searched `key_depth_factor` times deeper than the
    configured depth so that, after mapping keys to sentences, about `depth`
    distinct sentences remain (same depth as B3)."""

    def __init__(self, keys: list[Unit], targets: list[int], cfg: dict, cache_dir,
                 key_depth_factor: int = 4):
        from ragsplit.retrieve import Retriever

        self.depth = cfg["depth"]
        self.inner = Retriever(keys, {**cfg, "depth": cfg["depth"] * key_depth_factor}, cache_dir)
        self.targets = targets
        self.n_keys = len(keys)

    def encoder(self):
        return self.inner.encoder()

    def search(self, queries: list[str], prior=None, **kw) -> list[list[int]]:
        if prior is not None:
            raise ValueError("keyed retrieval does not support a prior")
        return [first_hits(r, self.targets, self.depth) for r in self.inner.search(queries)]
