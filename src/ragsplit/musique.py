"""MuSiQue-Ans (dev, 2,417 questions with 2, 3 or 4 hops) as a pooled corpus.

Every paragraph shown with any question (supporting + distractors, 20 per question) joins one
corpus, deduplicated by (title, text). Paragraphs are sentence-split with spaCy. Each
question comes decomposed into hops, each with an intermediate answer and the paragraph
that supports it; those become the question's Answers (conjunctive: every hop's answer must
reach the reader from its own paragraph, measured by `spans_located`), and `support`
lists the sentences holding them. Questions whose hop answers cannot all be found in their
supporting paragraphs are dropped. `qtype` is the hop type (2hop, 3hop1, 4hop2, ...).
"""

import json

from ragsplit.data import Answer, Paragraph, Question, sentence_of

FILE = "musique_ans_v1.0_dev.jsonl"


def load_rows(hf_id: str, cache_dir: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(hf_id, FILE, repo_type="dataset", cache_dir=cache_dir)
    return [json.loads(line) for line in open(path)]


def build_corpus(rows: list[dict], spacy_model: str) -> tuple[list[Paragraph], list[Question]]:
    from ragsplit.data import load_spacy, segment

    paragraphs, key_to_p = [], {}
    for r in rows:
        for p in r["paragraphs"]:
            key = (p["title"], p["paragraph_text"])
            if key not in key_to_p:
                para = Paragraph(f"m{len(paragraphs):05d}", p["title"], p["paragraph_text"].strip())
                key_to_p[key] = para
                paragraphs.append(para)
    nlp = load_spacy(spacy_model)
    for p, sents in zip(paragraphs, segment([p.context for p in paragraphs], nlp)):
        p.sents = sents
    questions = []
    for r in rows:
        if not r.get("answerable", True):
            continue
        by_idx = {p["idx"]: key_to_p[(p["title"], p["paragraph_text"])] for p in r["paragraphs"]}
        answers, support, ok = [], [], True
        for step in r["question_decomposition"]:
            p = by_idx.get(step["paragraph_support_idx"])
            text = step["answer"].strip()
            at = p.context.find(text) if p is not None and text else -1
            if at < 0 and p is not None and text:
                low = p.context.lower().find(text.lower())
                at = low
            if at < 0:
                ok = False
                break
            si = sentence_of(at, p.sents)
            answers.append(Answer(p.context[at:at + len(text)], at, start_sent=si, end_sent=si, paragraph_id=p.id))
            if si is not None and (p.id, si) not in support:
                support.append((p.id, si))
        if not ok or not answers:
            continue
        last = answers[-1]
        questions.append(Question(r["id"], r["question"], last.paragraph_id, answers,
                                  gold_sent=last.start_sent, support=support, qtype=r["id"].split("__")[0]))
    return paragraphs, questions
