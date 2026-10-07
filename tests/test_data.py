import pytest

from ragsplit.data import (
    Answer,
    Paragraph,
    Question,
    build_corpus,
    gold_sentence,
    locate_answer,
    map_all,
    map_answer,
    sentence_of,
)

# Three sentences with single-space gaps at offsets 11 and 26.
CTX = "Cats purr. Dogs bark loud. Birds sing."
SENTS = [(0, 10), (11, 26), (27, 38)]


def test_fixture_spans():
    assert [CTX[s:e] for s, e in SENTS] == ["Cats purr.", "Dogs bark loud.", "Birds sing."]


@pytest.mark.parametrize(
    "offset, expected",
    [
        (0, 0),     # first char
        (9, 0),     # last char of sentence 0 (the period)
        (10, 1),    # whitespace gap -> following sentence
        (11, 1),    # first char of sentence 1
        (25, 1),
        (27, 2),
        (37, 2),    # last char of the paragraph
        (38, None), # past the end
        (-1, None),
    ],
)
def test_sentence_of(offset, expected):
    assert sentence_of(offset, SENTS) == expected


def test_sentence_of_leading_whitespace():
    assert sentence_of(0, [(2, 5)]) == 0


def test_sentence_of_empty():
    assert sentence_of(0, []) is None


def test_map_answer_inside_one_sentence():
    a = Answer("bark", CTX.index("bark"))
    map_answer(CTX, SENTS, a)
    assert (a.start_sent, a.end_sent, a.offset_fixed) == (1, 1, False)


def test_map_answer_crossing_sentences():
    a = Answer("loud. Birds", CTX.index("loud"))
    map_answer(CTX, SENTS, a)
    assert (a.start_sent, a.end_sent) == (1, 2)


def test_map_answer_strips_surrounding_whitespace():
    # " Dogs" starts in the gap but the answer itself starts in sentence 1.
    a = Answer(" Dogs ", 10)
    map_answer(CTX, SENTS, a)
    assert (a.start_sent, a.end_sent) == (1, 1)


def test_map_answer_repairs_bad_offset():
    a = Answer("sing", 3)  # wrong offset
    map_answer(CTX, SENTS, a)
    assert a.offset_fixed and a.start_sent == 2


def test_map_answer_text_missing():
    a = Answer("meow", 0)
    map_answer(CTX, SENTS, a)
    assert a.start_sent is None and a.end_sent is None


def test_locate_answer_picks_nearest_occurrence():
    ctx = "a x b x c x"  # x at 2, 6, 10
    assert locate_answer(ctx, "x", 5) == (6, True)
    assert locate_answer(ctx, "x", 2) == (2, False)


def test_gold_sentence_majority_and_ties():
    def ans(*sents):
        return [Answer("t", 0, start_sent=s) for s in sents]

    assert gold_sentence(ans(2, 1, 1)) == 1
    assert gold_sentence(ans(2, 1)) == 2          # tie -> earliest answer
    assert gold_sentence(ans(None, 3)) == 3
    assert gold_sentence(ans(None)) is None


def test_build_corpus_ids_and_dedup():
    def row(qid, title, ctx, text, start):
        return {"id": qid, "title": title, "context": ctx, "question": "?",
                "answers": {"text": [text], "answer_start": [start]}}

    rows = [
        row("q1", "A", "ctx one", "one", 4),
        row("q2", "A", "ctx one", "ctx", 0),
        row("q3", "A", "ctx two", "two", 4),
        row("q4", "B", "other", "other", 0),
    ]
    paras, qs = build_corpus(rows)
    assert [p.id for p in paras] == ["a00p000", "a00p001", "a01p000"]
    assert [q.paragraph_id for q in qs] == ["a00p000", "a00p000", "a00p001", "a01p000"]


def test_map_all_sets_gold():
    p = Paragraph("a00p000", "T", CTX, SENTS)
    q = Question("q", "?", p.id, [Answer("Birds", 27), Answer("sing", 33), Answer("bark", 16)])
    map_all([p], [q])
    assert q.gold_sent == 2


@pytest.mark.slow
def test_spacy_segmentation_offsets():
    spacy = pytest.importorskip("spacy")
    from ragsplit.data import load_spacy, segment

    try:
        nlp = load_spacy("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    (sents,) = segment([CTX], nlp)
    assert [CTX[s:e] for s, e in sents] == ["Cats purr.", "Dogs bark loud.", "Birds sing."]
