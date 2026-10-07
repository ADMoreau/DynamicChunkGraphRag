"""Build the retrieval graph: text units, concept nodes and typed edges.

Front-loaded construction (all decisions made once, at build time):
  text nodes     cracked units (what the reader gets), each a span of one paragraph
  concept nodes  entities and fragments, merged where they are the same thing:
                   - identical named entities, and identical absolute dates, merge by string
                     (the reference labels found these almost always the same thing)
                   - bare numbers, ordinals, amounts and relative times never merge by string
                     ("300 patents" is not "300 cities"); only a judged SAME pair merges them
                   - any candidate pair the LLM judged SAME merges
edges            mentions   text unit <-> concept node it contains
                 adjacent   consecutive units of one paragraph (lets traversal rebuild
                            context that cracking separated)
                 related    concept <-> concept, from LLM RELATED verdicts (weaker, typed)
Common nodes are kept; limiting their influence is the traversal's job. Every node keeps
its degree so traversal can weigh it.
"""

from collections import Counter, defaultdict
import json
import re

from ragsplit.merge import TYPE_GROUP, UnionFind, normalize

NAMED = {"person", "org", "place", "group", "thing"}
ABSOLUTE_DATE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})s?\b|\b\d{1,2}(st|nd|rd|th) century\b", re.I)
RELATIVE = re.compile(r"\b(previous|next|following|last|that|this|same|earlier|later|ago|recent)\b", re.I)


def merges_by_string(mention: dict) -> bool:
    """May this mention merge with identical strings elsewhere without a judged pair?"""
    group = TYPE_GROUP[mention["label"]]
    if group in NAMED:
        return True
    if group == "time":
        return bool(ABSOLUTE_DATE.search(mention["text"])) and not RELATIVE.search(mention["text"])
    return False


def build_graph(units: list, mentions: list[dict], fragments: list[dict],
                candidates: dict[int, dict], verdicts: dict[int, str]) -> dict:
    """Nodes and edges as plain dicts (JSON-serializable).

    `fragments` are cracked clause/phrase pieces as {pid, s, e, text}; candidate pairs refer
    to mentions and fragments by (pid, s, e)."""
    m_index = {(m["pid"], m["s"], m["e"]): i for i, m in enumerate(mentions)}
    f_index = {(f["pid"], f["s"], f["e"]): len(mentions) + i for i, f in enumerate(fragments)}
    items = [{"kind": "entity", **m} for m in mentions] + [{"kind": "fragment", "label": None, **f} for f in fragments]
    uf = UnionFind(len(items))

    def item_of(side: dict, source: str):
        key = (side["pid"], side["s"], side["e"])
        return f_index.get(key) if source == "fragment" else m_index.get(key)

    # verdicts: SAME pairs are merge proposals; RELATED and DIFFERENT pairs are cannot-link
    # constraints, so a chain of SAME verdicts can never weld together two things the judge
    # said were related or different ("Luther" ~ "Reformation", two different parliaments)
    same, related, counts = [], [], Counter()
    cannot = defaultdict(set)
    for cid, label in verdicts.items():
        c = candidates[cid]
        a, b = item_of(c["a"], c["source"]), item_of(c["b"], c["source"])
        if a is None or b is None or label is None:
            counts["unmapped"] += 1
            continue
        counts[label] += 1
        if label == "SAME":
            same.append((-(c.get("cos") or 1.0), a, b))
        else:
            cannot[a].add(b)
            cannot[b].add(a)
            if label == "RELATED":
                related.append((a, b))
    members = {i: {i} for i in range(len(items))}
    blocked = {i: set(cannot[i]) for i in range(len(items))}
    # entity type per cluster (majority vote, robust to the odd NER mislabel): a person is
    # never the same thing as a company, nor a place the same thing as its nationality
    types = {i: Counter([TYPE_GROUP.get(it["label"], "fragment")]) for i, it in enumerate(items)}
    n_blocked = Counter()

    def try_union(a: int, b: int, why: str) -> None:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            return
        if members[ra] & blocked[rb] or members[rb] & blocked[ra]:
            n_blocked[why] += 1
            return
        if types[ra].most_common(1)[0][0] != types[rb].most_common(1)[0][0]:
            n_blocked[f"{why}: type"] += 1
            return
        uf.union(ra, rb)
        root, other = (ra, rb) if uf.find(ra) == ra else (rb, ra)
        members[root] |= members.pop(other)
        blocked[root] |= blocked.pop(other)
        types[root] += types.pop(other)

    # 1. identical strings that may merge without a judgement. A named-entity group is vetoed
    # when any judged pair inside it was DIFFERENT (e.g. "John", "SAT", "Barcelona"): only its
    # judged SAME pairs merge then. One-word person, organisation and "thing" names ("John",
    # "Parliament", "Treaty") merge by string only within an article.
    def string_key(m):
        key = (normalize(m["text"]), TYPE_GROUP[m["label"]])
        if key[1] in ("person", "org", "thing") and len(key[0].split()) == 1:
            key += (m["article"],)
        return key

    vetoed = set()
    for cid, label in verdicts.items():
        c = candidates[cid]
        if c["source"] == "exact" and label == "DIFFERENT" and c.get("group") in NAMED:
            ia, ib = item_of(c["a"], "exact"), item_of(c["b"], "exact")
            if ia is not None and ib is not None:
                vetoed.add(string_key(mentions[ia]))
                vetoed.add(string_key(mentions[ib]))
    first: dict = {}
    for i, m in enumerate(mentions):
        if merges_by_string(m) and string_key(m) not in vetoed:
            key = string_key(m)
            if key in first:
                try_union(first[key], i, "string")
            else:
                first[key] = i
    for i, f in enumerate(fragments):  # identical fragment text: same piece of text
        key = ("fragment", normalize(f["text"]))
        if key in first:
            try_union(first[key], len(mentions) + i, "string")
        else:
            first[key] = len(mentions) + i

    # 2. judged SAME pairs, most similar first
    for _, a, b in sorted(same):
        try_union(a, b, "verdict")

    # concept nodes = union-find clusters
    clusters = uf.clusters()
    concept_of = {}
    concepts = []
    for root, members in sorted(clusters.items()):
        cid = len(concepts)
        names = Counter(items[i]["text"] for i in members)
        concepts.append({"id": f"c{cid}", "name": names.most_common(1)[0][0],
                         "kind": items[root]["kind"], "group": TYPE_GROUP.get(items[root]["label"], "fragment"),
                         "aliases": [n for n, _ in names.most_common(8)], "n_mentions": len(members)})
        for i in members:
            concept_of[i] = cid

    # text units and their edges
    by_pid = defaultdict(list)
    for u in units:
        by_pid[u.paragraph_id].append(u)
    for us in by_pid.values():
        us.sort(key=lambda u: u.span[0])

    def unit_of(pid: str, s: int, e: int):
        """The unit holding the span's start (spans can straddle a cut; the start decides)."""
        for u in by_pid.get(pid, []):
            if u.span[0] <= s < u.span[1] or (s < u.span[0] < e):
                return u.id
        return None

    edges = []
    mention_pairs = set()
    for i, it in enumerate(items):
        uid = unit_of(it["pid"], it["s"], it["e"])
        if uid is not None and (uid, concept_of[i]) not in mention_pairs:
            mention_pairs.add((uid, concept_of[i]))
            edges.append({"type": "mentions", "a": uid, "b": f"c{concept_of[i]}"})
    for us in by_pid.values():
        for u, v in zip(us, us[1:]):
            edges.append({"type": "adjacent", "a": u.id, "b": v.id})
    seen = set()
    for a, b in related:
        ca, cb = concept_of[a], concept_of[b]
        if ca != cb and (min(ca, cb), max(ca, cb)) not in seen:
            seen.add((min(ca, cb), max(ca, cb)))
            edges.append({"type": "related", "a": f"c{ca}", "b": f"c{cb}"})

    degree = Counter()
    for e in edges:
        degree[e["a"]] += 1
        degree[e["b"]] += 1
    for c in concepts:
        c["degree"] = degree[c["id"]]
    text_nodes = [{"id": u.id, "pid": u.paragraph_id, "span": list(u.span), "kind": u.kind,
                   "text": u.text, "search_text": u.search_text, "degree": degree[u.id]} for u in units]
    return {"text_nodes": text_nodes, "concepts": concepts, "edges": edges,
            "verdict_counts": dict(counts), "vetoed_string_groups": len(vetoed),
            "merges_blocked_by_constraints": dict(n_blocked)}


def graph_report(g: dict, top: int = 15) -> list[str]:
    edge_counts = Counter(e["type"] for e in g["edges"])
    big = sorted(g["concepts"], key=lambda c: -c["n_mentions"])[:top]
    hubs = sorted(g["concepts"], key=lambda c: -c["degree"])[:top]
    kinds = Counter(c["kind"] for c in g["concepts"])
    lines = ["# Graph build report", "",
             f"- text nodes: {len(g['text_nodes'])}",
             f"- concept nodes: {len(g['concepts'])} ({dict(kinds)})",
             f"- edges: {dict(edge_counts)}",
             f"- LLM verdicts used: {g['verdict_counts']}",
             f"- identical-name groups vetoed from string merging: {g['vetoed_string_groups']}",
             f"- merges blocked (by RELATED/DIFFERENT verdicts, or by entity type): {g['merges_blocked_by_constraints']}", "",
             "## Largest merged concepts (check for chained merges)", ""]
    lines += [f"- {c['n_mentions']} mentions: {c['aliases']}" for c in big]
    lines += ["", "## Highest-degree concepts (hubs; traversal should damp these)", ""]
    lines += [f"- degree {c['degree']}: {c['name']!r} ({c['group']})" for c in hubs]
    return lines


def run_graph(cfg: dict, force: bool) -> None:
    from pathlib import Path

    from ragsplit.crack import crack_units, extracted_spans, parse_sentences
    from ragsplit.merge import extract_mentions
    from ragsplit.retrieve import units_key
    from ragsplit.run import prepare_corpus, prepare_splits, reference_qids
    from ragsplit.units import sentence_units

    gc = cfg["graph"]
    paragraphs, questions = prepare_corpus(cfg, force=False)
    sp = prepare_splits(cfg, questions, force=False)
    q_by_id = {q.id: q for q in questions}
    data = Path(cfg["data"]["data_dir"]) / "cache"
    key = units_key(sentence_units(paragraphs))
    parses = parse_sentences(paragraphs, cfg["data"]["spacy_model"], data / "crack" / f"parses_{key}.json")
    past = [q_by_id[q] for q in reference_qids(cfg, sp)]
    if gc.get("units") == "sentences":  # plain sentences (no learned cuts), as text nodes
        units = crack_units(paragraphs, {}, context_index=False, base="sentence")
    elif gc.get("units") == "sentence_base":  # sentences everywhere + learned finer cuts
        k = dict(zip(("sentence", "clause", "phrase"), gc["unit_k"]))
        units = crack_units(paragraphs, extracted_spans(paragraphs, past, parses, k), context_index=True,
                            base="sentence")
    else:
        k = dict(zip(("sentence", "clause", "phrase"), gc["unit_k"]))
        units = crack_units(paragraphs, extracted_spans(paragraphs, past, parses, k), context_index=True)
    mentions = extract_mentions(paragraphs, cfg["data"]["spacy_model"], data / "merge" / f"mentions_{key}.json")
    candidates = {c["id"]: c for c in map(json.loads, open(gc["candidates"]))}
    fragments, seen = [], set()
    for c in candidates.values():
        if c["source"] == "fragment":
            for side in (c["a"], c["b"]):
                k2 = (side["pid"], side["s"], side["e"])
                if k2 not in seen:
                    seen.add(k2)
                    fragments.append({"pid": side["pid"], "s": side["s"], "e": side["e"], "text": side["text"]})
    verdicts = {d["id"]: d["label"] for d in map(json.loads, open(gc["verdicts"]))}
    g = build_graph(units, mentions, fragments, candidates, verdicts)
    out = data / "graph" / f"{cfg['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(g))
    report = graph_report(g)
    (Path(cfg.get("results_dir", "results")) / f"{cfg['name']}.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nwrote {out}")
