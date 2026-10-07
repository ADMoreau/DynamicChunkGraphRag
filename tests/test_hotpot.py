from ragsplit.hotpot import build_corpus


def row(qid, question, answer, support, context):
    return {"id": qid, "question": question, "answer": answer, "type": "bridge", "level": "hard",
            "supporting_facts": {"title": [t for t, _ in support], "sent_id": [i for _, i in support]},
            "context": {"title": [t for t, _ in context], "sentences": [s for _, s in context]}}


def test_pooled_corpus_support_and_answer_location():
    ctx = [("Eiffel Tower", ["The Eiffel Tower is in Paris.", " It was designed by Gustave Eiffel."]),
           ("Gustave Eiffel", ["Gustave Eiffel was born in Dijon.", " He died in 1923."])]
    rows = [row("q1", "Where was the tower's designer born?", "Dijon", [("Eiffel Tower", 1), ("Gustave Eiffel", 0)], ctx),
            row("q2", "Is the tower in Paris?", "yes", [("Eiffel Tower", 0)], ctx)]
    paras, qs = build_corpus(rows, n_questions=2, seed=0)
    assert len(paras) == 2                                   # deduplicated by title
    p = {x.title: x for x in paras}
    assert p["Eiffel Tower"].context == "The Eiffel Tower is in Paris. It was designed by Gustave Eiffel."
    s, e = p["Eiffel Tower"].sents[1]
    assert p["Eiffel Tower"].context[s:e] == "It was designed by Gustave Eiffel."
    q = {x.id: x for x in qs}
    assert q["q1"].support == [(p["Eiffel Tower"].id, 1), (p["Gustave Eiffel"].id, 0)]
    assert q["q1"].paragraph_id == p["Gustave Eiffel"].id and q["q1"].gold_sent == 0   # where the answer is
    assert p["Gustave Eiffel"].context[q["q1"].answers[0].start:].startswith("Dijon")
    assert q["q2"].answers[0].start == -1                    # yes/no: not in the text
