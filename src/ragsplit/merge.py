"""Merging nodes that are the same: entity resolution and same-meaning merging.

Concept nodes stand for a thing (an entity, a value, a fact) and link every text unit
that mentions it; the reader still only ever gets text. Merges are made in three tiers,
each kept separate so it can be inspected before the graph relies on it:
  exact    same normalized surface form and entity-type group
  alias    within one article, when unambiguous: surname <-> full name, short form <->
           long form ("Broncos" <-> "Denver Broncos"), acronym <-> expansion
           ("NFL" <-> "National Football League")
  meaning  embedding similarity >= a threshold, same type group, identical numbers and
           negation; a candidate must be similar to the whole cluster (its centroid), so
           merges cannot chain A~B~C into one node
Fragments (cracked clause/phrase pieces) only use the meaning tier.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np

from ragsplit.data import Paragraph

TYPE_GROUP = {
    "PERSON": "person", "ORG": "org", "GPE": "place", "LOC": "place", "FAC": "place",
    "DATE": "time", "TIME": "time", "CARDINAL": "number", "QUANTITY": "number",
    "MONEY": "number", "PERCENT": "number", "ORDINAL": "number", "NORP": "group",
    "EVENT": "thing", "WORK_OF_ART": "thing", "LAW": "thing", "PRODUCT": "thing",
    "LANGUAGE": "thing",
}
NEGATION = re.compile(r"\b(not|no|never|none|nor|without|n't)\b|n't\b", re.I)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)


def normalize(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"'s$", "", t.strip())
    t = ARTICLE.sub("", t)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)          # 1,000 -> 1000
    t = re.sub(r"[^\w\s.%$]", " ", t)
    return re.sub(r"\s+", " ", t).strip(" .")


def numbers(text: str) -> frozenset:
    return frozenset(n.replace(",", "") for n in NUMBER.findall(text))


def negated(text: str) -> bool:
    return bool(NEGATION.search(text))


def acronym_of(words: list[str]) -> str:
    return "".join(w[0] for w in words if w[:1].isupper())


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True

    def clusters(self) -> dict[int, list[int]]:
        out = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return out


# --- mentions ------------------------------------------------------------------

def extract_mentions(paragraphs: list[Paragraph], spacy_model: str, cache_path: Path) -> list[dict]:
    """Named-entity mentions with paragraph char offsets, cached."""
    if cache_path.exists():
        print(f"[cache] entity mentions {cache_path}")
        return json.loads(cache_path.read_text())
    import spacy

    nlp = spacy.load(spacy_model, exclude=["parser", "lemmatizer"])
    out = []
    for p, doc in zip(paragraphs, nlp.pipe([p.context for p in paragraphs], batch_size=64)):
        for ent in doc.ents:
            if ent.label_ in TYPE_GROUP and normalize(ent.text):
                out.append({"pid": p.id, "article": p.title, "s": ent.start_char, "e": ent.end_char,
                            "text": ent.text, "label": ent.label_})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out))
    print(f"[build] {len(out)} entity mentions -> {cache_path}")
    return out


# --- tiers ---------------------------------------------------------------------

def exact_tier(mentions: list[dict], uf: UnionFind) -> list[tuple[int, int]]:
    first: dict[tuple, int] = {}
    merges = []
    for i, m in enumerate(mentions):
        key = (normalize(m["text"]), TYPE_GROUP[m["label"]])
        if key in first:
            if uf.union(first[key], i):
                merges.append((first[key], i))
        else:
            first[key] = i
    return merges


def alias_tier(mentions: list[dict], uf: UnionFind) -> list[tuple[int, int, str]]:
    """Within-article aliases, merged only when exactly one longer form matches."""
    by_article = defaultdict(list)
    for i, m in enumerate(mentions):
        by_article[m["article"]].append(i)
    merges = []
    for idx in by_article.values():
        longs: dict[str, int] = {}          # distinct normalized long forms -> a mention
        for i in idx:
            n = normalize(mentions[i]["text"])
            if len(n.split()) > 1:
                longs.setdefault(n, i)
        for i in idx:
            m = mentions[i]
            n = normalize(m["text"])
            words = n.split()
            if len(words) != 1 and not m["text"].isupper():
                continue
            g = TYPE_GROUP[m["label"]]
            cands, rule = [], ""
            if m["text"].isupper() and 2 <= len(m["text"]) <= 6:      # acronym, any type
                cands = [j for ln, j in longs.items()
                         if acronym_of(mentions[j]["text"].split()) == m["text"]]
                rule = "acronym"
            elif g in ("person", "org", "place", "group", "thing"):
                # surname / first name for people, last word ("Broncos") otherwise
                cands = [j for ln, j in longs.items() if TYPE_GROUP[mentions[j]["label"]] == g
                         and (ln.split()[-1] == n or (g == "person" and ln.split()[0] == n))]
                rule = "short form"
            if len(cands) == 1 and uf.union(cands[0], i):
                merges.append((cands[0], i, rule))
    return merges


def meaning_tier(vectors: np.ndarray, texts: list[str], groups: list[str], tau: float,
                 top: int = 10, block: int = 2048) -> tuple[UnionFind, list[tuple[int, int, float]]]:
    """Merge items (one per existing node) with cosine >= tau, same group, identical
    numbers and negation; the two clusters' centroids must also be within tau, so a
    run of pairwise-similar items cannot chain into one node."""
    n = len(texts)
    nums = [numbers(t) for t in texts]
    neg = [negated(t) for t in texts]
    pairs = []
    for b0 in range(0, n, block):
        sims = vectors[b0:b0 + block] @ vectors.T
        for r, row in enumerate(sims):
            i = b0 + r
            row[i] = -1
            k = min(top, n - 1)
            for j in np.argpartition(-row, k)[:k] if k > 0 else []:
                if row[j] >= tau and i < j:
                    pairs.append((float(row[j]), i, int(j)))
    uf = UnionFind(n)
    members = {i: [i] for i in range(n)}
    merges = []
    for sim, i, j in sorted(pairs, reverse=True):
        if groups[i] != groups[j] or nums[i] != nums[j] or neg[i] != neg[j]:
            continue
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        ci, cj = vectors[members[ri]].mean(axis=0), vectors[members[rj]].mean(axis=0)
        if float(ci @ cj) / (np.linalg.norm(ci) * np.linalg.norm(cj)) < tau:
            continue
        uf.union(ri, rj)
        members[min(ri, rj)] = members.pop(ri) + members.pop(rj)
        merges.append((i, j, sim))
    return uf, merges


def embed(texts: list[str], cfg: dict) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg["dense_model"], cache_folder=cfg.get("model_cache"))
    return model.encode(texts, batch_size=128, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)


# --- report ----------------------------------------------------------------------

def cluster_stats(uf: UnionFind, texts: list[str]) -> dict:
    sizes = Counter(len(m) for m in uf.clusters().values())
    big = sorted(uf.clusters().values(), key=len, reverse=True)[:5]
    return {"items": len(texts), "nodes": len(uf.clusters()),
            "largest": [(len(m), Counter(texts[i] for i in m).most_common(4)) for m in big],
            "singletons": sizes.get(1, 0)}


def run_merge(cfg: dict, force: bool) -> None:
    """Build concept nodes for entities and cracked fragments and write a report with
    counts, sample merges per tier and threshold, largest clusters and hubs."""
    import random

    from ragsplit.crack import crack_units, extracted_spans, parse_sentences
    from ragsplit.retrieve import units_key
    from ragsplit.run import prepare_corpus, prepare_splits
    from ragsplit.units import sentence_units

    mc = cfg["merge"]
    paragraphs, questions = prepare_corpus(cfg, force=False)
    sp = prepare_splits(cfg, questions, force=False)
    q_by_id = {q.id: q for q in questions}
    cache = Path(cfg["data"]["data_dir"]) / "cache" / "merge"
    key = units_key(sentence_units(paragraphs))
    rng = random.Random(0)
    out, md = {}, ["# Node merging report", ""]

    def section(title, merges, fmt):
        md.extend([f"### {title}: {len(merges)} merges", ""])
        for m in rng.sample(merges, min(mc["samples"], len(merges))):
            md.append("- " + fmt(m))
        md.append("")

    # entities: exact, then alias, then meaning over the resulting nodes
    mentions = extract_mentions(paragraphs, cfg["data"]["spacy_model"], cache / f"mentions_{key}.json")
    uf = UnionFind(len(mentions))
    ex = exact_tier(mentions, uf)
    after_exact = len(uf.clusters())
    al = alias_tier(mentions, uf)
    clusters = uf.clusters()
    roots = sorted(clusters)
    rep = [Counter(mentions[i]["text"] for i in clusters[r]).most_common(1)[0][0] for r in roots]
    grp = [TYPE_GROUP[mentions[r]["label"]] for r in roots]
    vec = embed(rep, cfg["retrieval"])
    out["entities"] = {"mentions": len(mentions), "after_exact": after_exact, "after_alias": len(roots),
                       "alias_rules": dict(Counter(r for *_, r in al)), "meaning": {}}
    md += ["## Entities", "",
           f"{len(mentions)} mentions -> {after_exact} nodes after exact -> {len(roots)} after alias "
           f"({dict(Counter(r for *_, r in al))})", ""]
    t = lambda i: mentions[i]["text"]
    section("Alias merges (sample)", al, lambda m: f"{t(m[1])!r} -> {t(m[0])!r} ({m[2]}, {mentions[m[0]]['article']})")
    for tau in mc["entity_taus"]:
        muf, merges = meaning_tier(vec, rep, grp, tau)
        n_nodes = len(muf.clusters())
        out["entities"]["meaning"][str(tau)] = {"nodes": n_nodes, "merges": len(merges)}
        md += [f"Meaning tier at {tau}: {len(roots)} -> {n_nodes} nodes", ""]
        section(f"Entity meaning merges at {tau} (sample)", merges,
                lambda m: f"{rep[m[0]]!r} ~ {rep[m[1]]!r} ({m[2]:.3f}, {grp[m[0]]})")

    # hubs: entity nodes (after alias) by number of distinct paragraphs they link
    deg = sorted(((len({mentions[i]["pid"] for i in clusters[r]}), rep[n]) for n, r in enumerate(roots)), reverse=True)
    out["entities"]["hubs"] = deg[:15]
    md += ["### Biggest hubs (paragraphs linked)", ""] + [f"- {d}: {txt!r}" for d, txt in deg[:15]] + [""]

    # fragments: cracked clause/phrase pieces, exact then meaning
    parses = parse_sentences(paragraphs, cfg["data"]["spacy_model"], Path(cfg["data"]["data_dir"]) / "cache" / "crack" / f"parses_{key}.json")
    k = dict(zip(("sentence", "clause", "phrase"), mc["fragment_k"]))
    units = crack_units(paragraphs, extracted_spans(paragraphs, [q_by_id[q] for q in sp["past"]], parses, k))
    frags = [u.text.split(": ", 1)[1] for u in units if u.kind in ("piece-clause", "piece-phrase")]
    fuf = UnionFind(len(frags))
    fex = 0
    first = {}
    for i, f in enumerate(frags):
        n = normalize(f)
        if n in first:
            fex += fuf.union(first[n], i)
        else:
            first[n] = i
    fclusters = fuf.clusters()
    froots = sorted(fclusters)
    frep = [frags[r] for r in froots]
    fvec = embed(frep, cfg["retrieval"])
    out["fragments"] = {"pieces": len(frags), "after_exact": len(froots), "meaning": {}}
    md += ["## Fragments (cracked clause and phrase pieces)", "",
           f"{len(frags)} pieces at k={mc['fragment_k']} -> {len(froots)} nodes after exact", ""]
    for tau in mc["fragment_taus"]:
        muf, merges = meaning_tier(fvec, frep, ["fragment"] * len(frep), tau)
        out["fragments"]["meaning"][str(tau)] = {"nodes": len(muf.clusters()), "merges": len(merges)}
        md += [f"Meaning tier at {tau}: {len(froots)} -> {len(muf.clusters())} nodes", ""]
        section(f"Fragment meaning merges at {tau} (sample)", merges,
                lambda m: f"{frep[m[0]]!r} ~ {frep[m[1]]!r} ({m[2]:.3f})")

    results = Path(cfg.get("results_dir", "results"))
    (results / f"{cfg['name']}.json").write_text(json.dumps(out, indent=2) + "\n")
    (results / f"{cfg['name']}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
