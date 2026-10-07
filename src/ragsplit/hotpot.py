"""HotpotQA (distractor setting) as a pooled corpus for multi-hop retrieval.

A seeded sample of questions is taken; every paragraph shown with them (2 gold + 8
distractors each) joins one shared corpus, deduplicated by Wikipedia title. HotpotQA
paragraphs come split into sentences, so no segmentation is needed. Each question keeps
its supporting sentences (`support`, two or more across two paragraphs). Its
`paragraph_id` / `gold_sent` point at the supporting sentence that holds the answer (the
first supporting sentence for yes/no answers, which never appear in the text).
"""

import random

from ragsplit.data import Answer, Paragraph, Question

PARQUET = "distractor/validation-00000-of-00001.parquet"


def load_rows(hf_id: str, cache_dir: str) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(hf_id, PARQUET, repo_type="dataset", cache_dir=cache_dir)
    return pd.read_parquet(path).to_dict("records")


def build_corpus(rows: list[dict], n_questions: int, seed: int) -> tuple[list[Paragraph], list[Question]]:
    rows = sorted(rows, key=lambda r: r["id"])
    rows = random.Random(seed).sample(rows, min(n_questions, len(rows)))
    paragraphs: list[Paragraph] = []
    by_title: dict[str, Paragraph] = {}
    by_id: dict[str, Paragraph] = {}
    questions: list[Question] = []
    for r in rows:
        for title, sentences in zip(r["context"]["title"], r["context"]["sentences"]):
            if title in by_title:
                continue
            context, spans, pos = "", [], 0
            for s in sentences:
                piece = s.strip()
                if not piece:
                    spans.append((pos, pos))
                    continue
                if context:
                    context += " "
                    pos += 1
                spans.append((pos, pos + len(piece)))
                context += piece
                pos += len(piece)
            p = Paragraph(f"h{len(paragraphs):05d}", title, context, spans)
            by_title[title] = by_id[p.id] = p
            paragraphs.append(p)
        support = []
        for title, sid in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]):
            p = by_title.get(title)
            if p is not None and 0 <= sid < len(p.sents) and p.sents[sid][1] > p.sents[sid][0]:
                support.append((p.id, int(sid)))
        if not support:
            continue
        answer = r["answer"].strip()
        holder, start = support[0], -1
        for pid, sid in support:
            p = by_id[pid]
            s, e = p.sents[sid]
            at = p.context.find(answer, s, e) if answer.lower() not in ("yes", "no") else -1
            if at >= 0:
                holder, start = (pid, sid), at
                break
        a = Answer(answer, start, start_sent=holder[1] if start >= 0 else None,
                   end_sent=holder[1] if start >= 0 else None)
        questions.append(Question(r["id"], r["question"], holder[0], [a], gold_sent=holder[1],
                                  support=support, qtype=f"{r['type']}/{r['level']}"))
    return paragraphs, questions
