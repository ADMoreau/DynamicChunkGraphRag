import pytest

from ragsplit.crack import answer_spans, crack_units, extracted_spans
from ragsplit.data import Answer, Paragraph, Question
from ragsplit.metrics import answer_located

# "The tower, which Eiffel designed, was completed in 1889. It is tall."
TEXT = "The tower, which Eiffel designed, was completed in 1889. It is tall."
SENTS = [(0, 56), (57, 68)]


def tok(s, word, dep, pos, head, punct=False):
    return {"s": s, "e": s + len(word), "dep": dep, "pos": pos, "head": head, "punct": punct}


# parse of sentence 0 (head indices are token positions within the sentence)
S0 = [tok(0, "The", "det", "DET", 1), tok(4, "tower", "nsubjpass", "NOUN", 8),
      tok(9, ",", "punct", "PUNCT", 1, True), tok(11, "which", "dobj", "PRON", 5),
      tok(17, "Eiffel", "nsubj", "PROPN", 5), tok(24, "designed", "relcl", "VERB", 1),
      tok(32, ",", "punct", "PUNCT", 1, True), tok(34, "was", "auxpass", "AUX", 8),
      tok(38, "completed", "ROOT", "VERB", 8), tok(48, "in", "prep", "ADP", 8),
      tok(51, "1889", "pobj", "NUM", 9), tok(55, ".", "punct", "PUNCT", 8, True)]
S1 = [tok(57, "It", "nsubj", "PRON", 1), tok(60, "is", "ROOT", "AUX", 1),
      tok(63, "tall", "acomp", "ADJ", 1), tok(67, ".", "punct", "PUNCT", 1, True)]
PARSES = {("p", 0): S0, ("p", 1): S1}
P = Paragraph("p", "Eiffel_Tower", TEXT, SENTS)


def span_text(span):
    return TEXT[span[0]:span[1]]


def test_answer_spans_levels():
    sp = answer_spans(TEXT, S0, SENTS[0], 51, 55)  # "1889"
    assert span_text(sp["sentence"]) == "The tower, which Eiffel designed, was completed in 1889."
    assert "clause" not in sp            # the answer's clause is the whole sentence
    assert span_text(sp["phrase"]) == "1889"
    sp = answer_spans(TEXT, S0, SENTS[0], 17, 23)  # "Eiffel", inside the relative clause
    assert span_text(sp["clause"]) == "which Eiffel designed"
    assert span_text(sp["phrase"]) == "Eiffel"


def q(qid, start, text):
    return Question(qid, "?", "p", [Answer(text, start)])


def test_k_thresholds_rise_and_cuts_replace():
    past = [q("a", 51, "1889"), q("b", 51, "1889"), q("c", 17, "Eiffel")]
    spans = extracted_spans([P], past, PARSES, {"sentence": 1, "clause": 2, "phrase": 2})
    levels = {span_text(s): lvl for s, lvl in spans["p"].items()}
    # "1889" had 2 questions (>= k phrase), "Eiffel"/its clause only 1 (< k)
    assert levels == {"The tower, which Eiffel designed, was completed in 1889.": "sentence", "1889": "phrase"}
    units = crack_units([P], spans)
    texts = [u.text.split(": ", 1)[1] for u in units]
    assert texts == ["The tower, which Eiffel designed, was completed in", "1889", "It is tall."]
    assert [u.kind for u in units] == ["segment", "piece-phrase", "segment"]
    with pytest.raises(ValueError):
        extracted_spans([P], past, PARSES, {"sentence": 2, "clause": 1, "phrase": 3})


def test_context_index_only_for_sub_sentence_units():
    spans = {"p": {(51, 55): "phrase"}}
    units = crack_units([P], spans, context_index=True)
    piece = next(u for u in units if u.kind == "piece-phrase")
    assert piece.text == "Eiffel Tower: 1889"
    assert piece.search_text == "Eiffel Tower: 1889 || " + TEXT[0:56]
    before = next(u for u in units if u.span[1] == 50)
    assert before.search_text.endswith("|| " + TEXT[0:56])  # also inside one sentence
    whole = crack_units([P], {"p": {(0, 56): "sentence"}}, context_index=True)
    assert all(u.index_text is None for u in whole)  # whole sentences need no context
    assert len({frozenset(u.covers()) for u in units}) == len(units)  # tiles never overlap


def test_answer_located_across_adjacent_pieces_only_in_gold_paragraph():
    parts = [("p", 51, "Eiffel Tower: 18"), ("x", 0, "Other: completed in 1889"), ("p", 38, "Eiffel Tower: completed in")]
    assert not answer_located(parts, "p", "Eiffel Tower: ", ["completed in 1889"])  # "18" != "1889"
    parts[0] = ("p", 51, "Eiffel Tower: 1889")
    assert answer_located(parts, "p", "Eiffel Tower: ", ["completed in 1889"])
    assert not answer_located(parts[1:2], "p", "Eiffel Tower: ", ["1889"])


def test_answer_across_two_parse_trees_keeps_sentence_only():
    # "It is tall." parsed as two trees: "It is" and "tall." (two roots)
    toks = [tok(57, "It", "nsubj", "PRON", 1), tok(60, "is", "ROOT", "AUX", 1),
            tok(63, "tall", "ROOT", "ADJ", 2), tok(67, ".", "punct", "PUNCT", 2, True)]
    sp = answer_spans(TEXT, toks, SENTS[1], 60, 67)   # "is tall" spans both trees
    assert set(sp) == {"sentence"}


def test_sentence_base_cracks_below_sentences_only():
    units = crack_units([P], {"p": {(51, 55): "phrase"}}, base="sentence")
    texts = [u.text.split(": ", 1)[1] for u in units]
    assert texts == ["The tower, which Eiffel designed, was completed in", "1889", "It is tall."]
    plain = crack_units([P], {}, base="sentence")
    assert [u.text.split(": ", 1)[1] for u in plain] == ["The tower, which Eiffel designed, was completed in 1889.", "It is tall."]


def test_evidence_units_rejoin_lock_and_k():
    from ragsplit.crack import evidence_groups, evidence_units

    # evidence crossing the sentence boundary ("1889. It is tall") is rejoined into one unit
    past = [Question("a", "?", "p", [Answer("completed in 1889. It is tall", 38)]),
            Question("b", "?", "p", [Answer("completed in 1889. It is tall.", 38)]),   # same evidence (IoU)
            Question("c", "?", "p", [Answer("Eiffel designed, was completed", 17)])]  # cuts through it
    groups = evidence_groups([P], past, k=2)
    assert [n for _, n in groups["p"]] == [2]                     # c's span alone is below k
    texts = [u.text.split(": ", 1)[1] for u in evidence_units([P], groups)]
    assert texts == ["The tower, which Eiffel designed, was", "completed in 1889. It is tall"]
    g1 = evidence_groups([P], past, k=1)
    texts1 = [u.text.split(": ", 1)[1] for u in evidence_units([P], g1)]
    assert "completed in 1889. It is tall" in texts1              # locked: c (1 question) can't cut it
