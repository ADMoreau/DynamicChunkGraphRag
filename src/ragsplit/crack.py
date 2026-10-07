"""Hierarchical cracking: split text at sentence, clause and phrase level where past
questions' answers concentrate, every split replacing the node it cuts.

For each past question we find the answer's sentence, the smallest clause holding the
answer, and the smallest phrase (syntactic constituent) holding it. A span at level l
is extracted once at least k[l] distinct past questions pointed at it; k rises as spans
shrink, because a cut that small needs more evidence (user, 3 Oct 2026). Extracting a
span cuts whatever node holds it into before / span / after, so every paragraph stays
tiled by contiguous, non-overlapping units: a paragraph nobody asked about stays whole,
and repeated extraction cracks it into a chain, like database cracking.
"""

import json
from pathlib import Path

from ragsplit.data import Paragraph, Question, locate_answer, sentence_of
from ragsplit.units import Unit, title_prefix

LEVELS = ("sentence", "clause", "phrase")
CLAUSE_DEPS = {"ccomp", "advcl", "relcl", "conj", "xcomp", "acl", "csubj", "parataxis"}
EDGE_CHARS = " \t\n,;:"


# --- parse cache: per sentence, the token table the span logic needs ----------

def parse_sentences(paragraphs: list[Paragraph], spacy_model: str, cache_path: Path) -> dict:
    """{(pid, sent): [token dicts]} with paragraph-level char offsets, cached as JSON."""
    if cache_path.exists():
        print(f"[cache] sentence parses {cache_path}")
        raw = json.loads(cache_path.read_text())
        return {(k.split("|")[0], int(k.split("|")[1])): v for k, v in raw.items()}
    from ragsplit.data import load_spacy

    nlp = load_spacy(spacy_model)
    keys, texts = [], []
    for p in paragraphs:
        for i, (s, e) in enumerate(p.sents):
            keys.append((p.id, i, s))
            texts.append(p.context[s:e])
    out = {}
    for (pid, i, base), doc in zip(keys, nlp.pipe(texts, batch_size=256)):
        out[(pid, i)] = [{"s": base + t.idx, "e": base + t.idx + len(t.text), "dep": t.dep_,
                          "pos": t.pos_, "head": t.head.i, "punct": t.is_punct} for t in doc]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({f"{pid}|{i}": v for (pid, i), v in out.items()}))
    print(f"[build] parsed {len(out)} sentences -> {cache_path}")
    return out


# --- spans ------------------------------------------------------------------------

def trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s] in EDGE_CHARS:
        s += 1
    while e > s and text[e - 1] in EDGE_CHARS:
        e -= 1
    return s, e


def ancestors(toks: list[dict], i: int) -> list[int]:
    out = [i]
    while toks[i]["head"] != i:
        i = toks[i]["head"]
        out.append(i)
    return out


def subtree_span(toks: list[dict], i: int) -> tuple[int, int]:
    """Char range of token i's subtree (contiguous: spaCy's parses are projective)."""
    members = [j for j in range(len(toks)) if i in ancestors(toks, j)]
    return min(toks[j]["s"] for j in members), max(toks[j]["e"] for j in members)


def answer_spans(text: str, toks: list[dict], sent: tuple[int, int],
                 a_start: int, a_end: int) -> dict[str, tuple[int, int]]:
    """The answer's sentence, smallest clause and smallest phrase, as trimmed char ranges.
    A level is left out when it is no smaller than the level above it."""
    out = {"sentence": trim(text, *sent)}
    hit = [j for j, t in enumerate(toks) if t["s"] < a_end and t["e"] > a_start and not t["punct"]]
    if not hit:
        return out
    common = set(ancestors(toks, hit[0]))
    for j in hit[1:]:
        common &= set(ancestors(toks, j))
    lca = next((a for a in ancestors(toks, hit[0]) if a in common), None)
    if lca is None:  # the answer straddles two parse trees (the parser split the sentence)
        return out
    head = next((a for a in ancestors(toks, lca)
                 if toks[a]["head"] == a or (toks[a]["dep"] in CLAUSE_DEPS and toks[a]["pos"] in ("VERB", "AUX"))),
                lca)
    clause = trim(text, *subtree_span(toks, head))
    phrase = trim(text, *subtree_span(toks, lca))
    if clause != out["sentence"]:
        out["clause"] = clause
    if phrase not in (out["sentence"], out.get("clause")):
        out["phrase"] = phrase
    return out


def question_answer(p: Paragraph, q: Question) -> tuple[int, int] | None:
    """Char range of the first annotator's answer, repaired as in data.map_answer."""
    for a in q.answers:
        if a.start < 0:  # e.g. HotpotQA yes/no answers: not in the text
            continue
        if a.paragraph_id is not None and a.paragraph_id != p.id:  # a hop answer elsewhere
            continue
        loc = locate_answer(p.context, a.text, a.start)
        if loc is not None and a.text.strip():
            start = loc[0] + len(a.text) - len(a.text.lstrip())
            return start, start + len(a.text.strip())
    return None


def extracted_spans(paragraphs: list[Paragraph], past: list[Question], parses: dict,
                    k: dict[str, int]) -> dict[str, dict[tuple[int, int], str]]:
    """{pid: {span: level}} for spans at least k[level] distinct past questions pointed at."""
    bad = [lvl for lvl in LEVELS[1:] if k[lvl] < k[LEVELS[LEVELS.index(lvl) - 1]]]
    if bad:
        raise ValueError(f"k must not fall as spans shrink: {k}")
    para = {p.id: p for p in paragraphs}
    counts: dict[tuple, set] = {}
    for q in past:
        # multi-hop data: every supporting sentence is a sentence-level pointer (both hops)
        for pid, si in q.support:
            sp = para[pid]
            counts.setdefault((pid, "sentence", trim(sp.context, *sp.sents[si])), set()).add(q.id)
        p = para[q.paragraph_id]
        ans = question_answer(p, q)
        if ans is None:
            continue
        si = sentence_of(ans[0], p.sents)
        if si is None:
            continue
        for lvl, span in answer_spans(p.context, parses[(p.id, si)], p.sents[si], *ans).items():
            counts.setdefault((p.id, lvl, span), set()).add(q.id)
    out: dict[str, dict] = {}
    for (pid, lvl, span), qs in counts.items():
        if len(qs) >= k[lvl]:
            prev = out.setdefault(pid, {}).get(span)
            if prev is None or LEVELS.index(lvl) > LEVELS.index(prev):
                out[pid][span] = lvl  # the same span at two levels counts as the finer one
    return out


def crack_units(paragraphs: list[Paragraph], spans: dict[str, dict[tuple[int, int], str]],
                context_index: bool = False, base: str = "paragraph") -> list[Unit]:
    """Tile each paragraph by the cut points of its extracted spans.

    base "paragraph": text no past question touched stays as whole paragraphs / segments.
    base "sentence": every sentence boundary is also a cut, so untouched text is plain
    sentences and cracking only adds finer (clause / phrase) cuts where demand repeats.

    Units carry their char span; a unit equal to an extracted span is a piece of that
    level, anything else a segment. With `context_index`, a unit inside one sentence is
    indexed (and reranked) as "unit || its sentence", while the reader still gets only
    the unit."""
    out = []
    for p in paragraphs:
        ext = spans.get(p.id, {})
        cuts = {0, len(p.context)} | {c for span in ext for c in span}
        if base == "sentence":
            cuts |= {c for s, e in p.sents for c in (s, e)}
        cuts = sorted(cuts)
        title = title_prefix(p.title)
        for a, b in zip(cuts, cuts[1:]):
            s, e = trim(p.context, a, b)
            if not any(ch.isalnum() for ch in p.context[s:e]):
                continue
            lo, hi = sentence_of(s, p.sents), sentence_of(e - 1, p.sents)
            lo = 0 if lo is None else lo
            hi = lo if hi is None else hi
            kind = f"piece-{ext[(s, e)]}" if (s, e) in ext else "segment"
            text = title + p.context[s:e]
            index_text = None
            if context_index and lo == hi and (s, e) != trim(p.context, *p.sents[lo]):
                index_text = f"{text} || {p.context[slice(*p.sents[lo])]}"
            out.append(Unit(f"{p.id}c{s:04d}-{e:04d}", kind, p.id, lo, hi, text,
                            span=(s, e), index_text=index_text))
    return out


# --- evidence-span units (sentence base, rejoin and lock) ----------------------------

def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1, max(a[1], b[1]) - min(a[0], b[0]))


def evidence_groups(paragraphs: list[Paragraph], past: list[Question], k: int,
                    min_iou: float = 0.7) -> dict[str, list[tuple[tuple[int, int], int]]]:
    """{pid: [(span, n_questions)]} for evidence spans at least k distinct past questions used.
    Spans from different questions that overlap by IoU >= min_iou count as the same evidence
    (the first span seen represents the group)."""
    para = {p.id: p for p in paragraphs}
    groups: dict[str, list[list]] = {}
    for q in past:
        for a in q.answers:
            if a.start < 0:
                continue
            pid = a.paragraph_id or q.paragraph_id
            p = para[pid]
            span = trim(p.context, a.start, min(len(p.context), a.start + len(a.text)))
            if span[1] <= span[0]:
                continue
            for g in groups.setdefault(pid, []):
                if _iou(g[0], span) >= min_iou:
                    g[1].add(q.id)
                    break
            else:
                groups[pid].append([span, {q.id}])
    return {pid: [(g[0], len(g[1])) for g in gs if len(g[1]) >= k] for pid, gs in groups.items()}


def evidence_units(paragraphs: list[Paragraph], groups: dict, context_index: bool = True) -> list[Unit]:
    """Sentence units, except that each demanded evidence span becomes exactly one unit:
    its edges are cut (crack) and any sentence boundary inside it is removed (rejoin).
    The most-demanded spans go first and their interiors are locked, so a later span that
    would cut through one is skipped."""
    out = []
    for p in paragraphs:
        cuts = {0, len(p.context)} | {c for s, e in p.sents for c in (s, e)}
        locked: list[tuple[int, int]] = []
        for (s, e), _ in sorted(groups.get(p.id, []), key=lambda g: (-g[1], g[0][0] - g[0][1])):
            if any(not (e <= ls or s >= le) for ls, le in locked):
                continue  # overlaps a locked span: never cut through it
            cuts = {c for c in cuts if not (s < c < e)} | {s, e}
            locked.append((s, e))
        cuts = sorted(cuts)
        title = title_prefix(p.title)
        lockset = set(locked)
        for a, b in zip(cuts, cuts[1:]):
            s, e = trim(p.context, a, b)
            if not any(ch.isalnum() for ch in p.context[s:e]):
                continue
            lo, hi = sentence_of(s, p.sents), sentence_of(e - 1, p.sents)
            lo = 0 if lo is None else lo
            hi = lo if hi is None else hi
            kind = "evidence" if (s, e) in lockset or (a, b) in lockset else "segment"
            text = title + p.context[s:e]
            index_text = None
            if context_index and lo == hi and (s, e) != trim(p.context, *p.sents[lo]):
                index_text = f"{text} || {p.context[slice(*p.sents[lo])]}"
            out.append(Unit(f"{p.id}e{s:04d}-{e:04d}", kind, p.id, lo, hi, text,
                            span=(s, e), index_text=index_text))
    return out
