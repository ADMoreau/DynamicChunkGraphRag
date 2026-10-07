"""SQuAD loading, sentence segmentation, and answer -> sentence mapping.

Sentences are stored as (start, end) character offsets into the paragraph,
end exclusive. Whitespace between sentences belongs to no sentence.
"""

from bisect import bisect_right
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

Span = tuple[int, int]


@dataclass
class Paragraph:
    id: str
    title: str
    context: str
    sents: list[Span] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    start: int            # character offset as given by SQuAD
    start_sent: int | None = None
    end_sent: int | None = None
    offset_fixed: bool = False  # start did not match text and was repaired
    paragraph_id: str | None = None  # where `start` points, if not the question's paragraph


@dataclass
class Question:
    id: str
    question: str
    paragraph_id: str
    answers: list[Answer]
    gold_sent: int | None = None  # majority sentence over annotators
    support: list = field(default_factory=list)  # (paragraph_id, sentence) pairs; multi-hop data only
    qtype: str | None = None                     # e.g. HotpotQA "bridge/hard"


# --- loading ---------------------------------------------------------------

def load_squad_rows(hf_id: str, split: str, cache_dir: str | Path) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(hf_id, split=split, cache_dir=str(cache_dir))
    return list(ds)


def build_corpus(rows: list[dict]) -> tuple[list[Paragraph], list[Question]]:
    """Group flat SQuAD rows into paragraphs and questions.

    Paragraph ids are a{article:02d}p{paragraph:03d}, numbered in order of
    first appearance, so they are stable for a given split.
    """
    article_idx: dict[str, int] = {}
    para_ids: dict[tuple[str, str], str] = {}
    paras_in_article: Counter = Counter()
    paragraphs: list[Paragraph] = []
    questions: list[Question] = []

    for row in rows:
        title, context = row["title"], row["context"]
        a = article_idx.setdefault(title, len(article_idx))
        key = (title, context)
        if key not in para_ids:
            pid = f"a{a:02d}p{paras_in_article[title]:03d}"
            paras_in_article[title] += 1
            para_ids[key] = pid
            paragraphs.append(Paragraph(pid, title, context))
        answers = [
            Answer(text=t, start=s)
            for t, s in zip(row["answers"]["text"], row["answers"]["answer_start"])
        ]
        questions.append(Question(row["id"], row["question"], para_ids[key], answers))
    return paragraphs, questions


# --- segmentation ----------------------------------------------------------

def load_spacy(model: str):
    import spacy

    return spacy.load(model, exclude=["ner", "lemmatizer"])


def segment(texts: list[str], nlp) -> list[list[Span]]:
    out = []
    for doc in nlp.pipe(texts, batch_size=64):
        out.append([(s.start_char, s.end_char) for s in doc.sents if s.text.strip()])
    return out


# --- answer -> sentence mapping ---------------------------------------------

def sentence_of(offset: int, sents: list[Span]) -> int | None:
    """Index of the sentence holding character `offset`.

    An offset in the whitespace before a sentence maps to that sentence.
    Offsets past the last sentence's end, or negative, map to None.
    """
    if offset < 0 or not sents or offset >= sents[-1][1]:
        return None
    i = bisect_right([s for s, _ in sents], offset) - 1
    if i < 0:
        return 0
    if offset >= sents[i][1]:
        return i + 1
    return i


def locate_answer(context: str, text: str, start: int) -> tuple[int, bool] | None:
    """Return (start, fixed): the answer's true start in `context`.

    Uses the given offset when the text matches there, otherwise the nearest
    occurrence of the text. None if the text does not occur at all.
    """
    if context[start:start + len(text)] == text:
        return start, False
    hits, i = [], context.find(text)
    while i != -1:
        hits.append(i)
        i = context.find(text, i + 1)
    if not hits:
        return None
    return min(hits, key=lambda h: abs(h - start)), True


def map_answer(context: str, sents: list[Span], ans: Answer) -> None:
    """Fill ans.start_sent / ans.end_sent in place (None if unmappable)."""
    loc = locate_answer(context, ans.text, ans.start)
    if loc is None:
        return
    start, ans.offset_fixed = loc
    stripped = ans.text.strip()
    if not stripped:
        return
    start += len(ans.text) - len(ans.text.lstrip())
    end = start + len(stripped)  # exclusive
    ans.start_sent = sentence_of(start, sents)
    ans.end_sent = sentence_of(end - 1, sents)


def gold_sentence(answers: list[Answer]) -> int | None:
    """Majority start sentence over annotators; ties go to the earliest answer."""
    sents = [a.start_sent for a in answers if a.start_sent is not None]
    if not sents:
        return None
    counts = Counter(sents)
    best = max(counts.values())
    return next(s for s in sents if counts[s] == best)


def map_all(paragraphs: list[Paragraph], questions: list[Question]) -> None:
    by_id = {p.id: p for p in paragraphs}
    for q in questions:
        p = by_id[q.paragraph_id]
        for a in q.answers:
            map_answer(p.context, p.sents, a)
        q.gold_sent = gold_sentence(q.answers)


def mapping_stats(paragraphs: list[Paragraph], questions: list[Question]) -> dict:
    answers = [a for q in questions for a in q.answers]
    n = len(answers)
    mapped = sum(a.start_sent is not None for a in answers)
    within = sum(a.start_sent is not None and a.start_sent == a.end_sent for a in answers)
    disagree = sum(
        len({a.start_sent for a in q.answers if a.start_sent is not None}) > 1
        for q in questions
    )
    return {
        "n_articles": len({p.title for p in paragraphs}),
        "n_paragraphs": len(paragraphs),
        "n_sentences": sum(len(p.sents) for p in paragraphs),
        "n_questions": len(questions),
        "n_answers": n,
        "answers_mapped": mapped,
        "answer_map_rate": mapped / n if n else 0.0,
        "answers_within_one_sentence": within,
        "answers_crossing_sentences": mapped - within,
        "answers_offset_fixed": sum(a.offset_fixed for a in answers),
        "questions_with_gold_sent": sum(q.gold_sent is not None for q in questions),
        "questions_annotators_disagree_on_sentence": disagree,
    }


# --- persistence -----------------------------------------------------------

def write_jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(asdict(it)) + "\n")


def read_paragraphs(path: Path) -> list[Paragraph]:
    out = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            d["sents"] = [tuple(s) for s in d["sents"]]
            out.append(Paragraph(**d))
    return out


def read_questions(path: Path) -> list[Question]:
    out = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            d["answers"] = [Answer(**a) for a in d["answers"]]
            d["support"] = [tuple(x) for x in d.get("support", [])]
            out.append(Question(**d))
    return out
