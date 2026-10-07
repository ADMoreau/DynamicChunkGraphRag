"""SyllabusQA (63 real course syllabi, 5,078 questions) as a corpus with repeated demand.

Syllabus text comes from PDF extraction with no reliable layout (some files are one long
line), so each syllabus is whitespace-normalized, sentence-split with spaCy and chunked
into "paragraphs" of up to `chunk_sents` consecutive sentences (`max_chars` at most).
Titles are the course names from the metadata, so every unit says which course it is from.

Only questions whose answer spans can all be found in the text are kept (spans are matched
word by word, ignoring case and punctuation). Each span becomes an Answer (text exactly as
in the document); `support` lists the sentences holding the spans, so "all supporting
sentences located" means every span reached the reader. The official split keeps syllabi
apart, so past questions would never touch test syllabi; this corpus is split by question
instead (prepare_splits), which is what learning from past questions needs.
"""

import csv
import glob
import json
import os
import re

from ragsplit.data import Answer, Paragraph, Question


def find_span(text: str, span: str) -> tuple[int, int] | None:
    words = re.findall(r"[A-Za-z0-9]+", span)
    if not words:
        return None
    m = re.search(r"\W+".join(map(re.escape, words)), text, flags=re.I)
    return (m.start(), m.end()) if m else None


def build_corpus(raw_dir: str, spacy_model: str, chunk_sents: int = 5,
                 max_chars: int = 700) -> tuple[list[Paragraph], list[Question]]:
    from ragsplit.data import load_spacy

    titles = {}
    with open(os.path.join(raw_dir, "meta.csv")) as f:
        for row in csv.DictReader(f):
            titles[row["name"]] = f"{row['course']} syllabus"
    rows = [r for s in ("train", "val", "test") for r in json.load(open(os.path.join(raw_dir, f"{s}.json")))]
    nlp = load_spacy(spacy_model)
    paragraphs, questions = [], []
    names = sorted({r["syllabus_name"] for r in rows})
    docs = {}
    for si, name in enumerate(names):
        raw = open(os.path.join(raw_dir, "text", f"{name}.txt"), errors="ignore").read()
        text = re.sub(r"\s+", " ", raw).strip()
        sents = [(s.start_char, s.end_char) for s in nlp(text).sents if s.text.strip()]
        chunks = []          # (pid, chunk start in text, [sentence spans in text])
        cur = []
        for s in sents:
            if cur and (len(cur) >= chunk_sents or s[1] - cur[0][0] > max_chars):
                chunks.append(cur)
                cur = []
            cur.append(s)
        if cur:
            chunks.append(cur)
        located = []
        for ci, ch in enumerate(chunks):
            pid = f"s{si:02d}p{ci:03d}"
            base = ch[0][0]
            p = Paragraph(pid, titles.get(name, name), text[base:ch[-1][1]],
                          [(a - base, b - base) for a, b in ch])
            paragraphs.append(p)
            located.append((base, ch[-1][1], p, ch))
        docs[name] = (text, located)

    def place(name: str, start: int):
        for base, end, p, ch in docs[name][1]:
            if base <= start < end:
                for k, (a, b) in enumerate(ch):
                    if a <= start < b:
                        return p, base, k
        return None

    for r in rows:
        spans = [r[f"answer_span_{i}"] for i in range(1, 6) if r[f"answer_span_{i}"]]
        if not spans:
            continue
        text = docs[r["syllabus_name"]][0]
        answers, support = [], []
        for span in spans:
            loc = find_span(text, span)
            if loc is None:
                break
            placed = place(r["syllabus_name"], loc[0])
            if placed is None:
                break
            p, base, k = placed
            end = min(loc[1], base + len(p.context))
            answers.append((p, Answer(text[loc[0]:end], loc[0] - base, start_sent=k, end_sent=k,
                                      paragraph_id=p.id)))
            if (p.id, k) not in support:
                support.append((p.id, k))
        else:
            first_p, first = answers[0]
            questions.append(Question(r["id"], r["question"], first_p.id, [a for _, a in answers],
                                      gold_sent=first.start_sent, support=support,
                                      qtype=r["question_type"]))
    return paragraphs, questions
