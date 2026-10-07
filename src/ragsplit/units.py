"""Retrieval units.

Every unit covers a contiguous run of sentences in one paragraph, and its text
is prefixed with the article title so it reads as self-contained.
"""

from dataclasses import dataclass

from ragsplit.data import Paragraph


@dataclass(frozen=True)
class Unit:
    id: str
    kind: str              # "paragraph", "sentence", "piece", ...
    paragraph_id: str
    sent_lo: int           # first sentence covered
    sent_hi: int           # last sentence covered, inclusive
    text: str              # title-prefixed text, as indexed and as sent to the reader
    span: tuple[int, int] | None = None   # char range in the paragraph (sub-sentence units)
    index_text: str | None = None         # what retrieval and reranking see, if not `text`

    @property
    def search_text(self) -> str:
        return self.index_text or self.text

    def covers(self) -> set:
        # sub-sentence units tile their paragraph, so a char span identifies them
        if self.span is not None:
            return {(self.paragraph_id, self.span)}
        return {(self.paragraph_id, i) for i in range(self.sent_lo, self.sent_hi + 1)}


def title_prefix(title: str) -> str:
    return title.replace("_", " ") + ": "


def paragraph_units(paragraphs: list[Paragraph]) -> list[Unit]:
    return [
        Unit(p.id, "paragraph", p.id, 0, len(p.sents) - 1, title_prefix(p.title) + p.context)
        for p in paragraphs
    ]


def sentence_units(paragraphs: list[Paragraph]) -> list[Unit]:
    return [
        Unit(f"{p.id}s{i:02d}", "sentence", p.id, i, i, title_prefix(p.title) + p.context[s:e])
        for p in paragraphs
        for i, (s, e) in enumerate(p.sents)
    ]


def piece_units(paragraphs: list[Paragraph], keys: set[tuple[str, int]]) -> list[Unit]:
    """Single-sentence pieces for the given (paragraph_id, sentence) keys.
    Pieces are added alongside their parents, never instead of them."""
    return [u for u in sentence_units(paragraphs) if (u.paragraph_id, u.sent_lo) in keys]


def split_units(paragraphs: list[Paragraph], keys: set[tuple[str, int]]) -> list[Unit]:
    """Split paragraphs on extracted sentences; the split node is replaced.

    Extracting sentence i from a node covering sentences lo..hi replaces it with
    lo..i-1, i, and i+1..hi (empty parts dropped). Repeated extraction gives the
    same result in any order: every extracted sentence becomes its own piece and
    each maximal run of other sentences becomes a segment. Paragraphs with no
    extracted sentence stay whole.
    """
    cuts: dict[str, list[int]] = {}
    for pid, i in keys:
        cuts.setdefault(pid, []).append(i)
    out = []
    for p in paragraphs:
        cut = sorted(cuts.get(p.id, []))
        if not cut:
            out.append(Unit(p.id, "paragraph", p.id, 0, len(p.sents) - 1,
                            title_prefix(p.title) + p.context))
            continue
        runs, lo = [], 0
        for i in cut:
            if lo <= i - 1:
                runs.append((lo, i - 1, "segment"))
            runs.append((i, i, "piece"))
            lo = i + 1
        if lo <= len(p.sents) - 1:
            runs.append((lo, len(p.sents) - 1, "segment"))
        for a, b, kind in runs:
            uid = f"{p.id}s{a:02d}" if kind == "piece" else f"{p.id}s{a:02d}-{b:02d}"
            text = p.context[p.sents[a][0]: p.sents[b][1]]
            out.append(Unit(uid, kind, p.id, a, b, title_prefix(p.title) + text))
    return out


def build_units(name: str, paragraphs: list[Paragraph],
                piece_keys: set[tuple[str, int]] | None = None) -> list[Unit]:
    if name == "paragraphs":
        return paragraph_units(paragraphs)
    if name == "sentences":
        return sentence_units(paragraphs)
    if name == "split":
        if piece_keys is None:
            raise ValueError("unit set 'split' needs piece_keys")
        return split_units(paragraphs, piece_keys)
    if name == "paragraphs+pieces":  # M3 oracle design: pieces added beside kept parents
        if piece_keys is None:
            raise ValueError("unit set 'paragraphs+pieces' needs piece_keys")
        return paragraph_units(paragraphs) + [
            Unit(u.id, "piece", u.paragraph_id, u.sent_lo, u.sent_hi, u.text)
            for u in piece_units(paragraphs, piece_keys)
        ]
    raise ValueError(f"unknown unit set {name!r}")
