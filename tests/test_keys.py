import pytest

from ragsplit.data import Paragraph
from ragsplit.keys import build_keys, clause_keys, first_hits, triple_keys
from ragsplit.units import sentence_units

TEXT = ("The Eiffel Tower, which was designed by Gustave Eiffel, was completed in 1889 "
        "and attracts millions of visitors.")


@pytest.fixture(scope="module")
def sent():
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    return next(nlp(TEXT).sents)


def test_clauses_split_relative_and_conjoined_clauses(sent):
    keys = clause_keys(sent)
    assert any("designed by Gustave Eiffel" in k for k in keys)
    assert any("attracts millions" in k for k in keys)
    assert not any("designed" in k and "attracts" in k for k in keys)
    # every content word lands in exactly one clause
    words = [w for k in keys for w in k.split()]
    assert words.count("1889") == 1 and words.count("visitors") == 1


def test_triples_are_compact_relations(sent):
    keys = triple_keys(sent)
    assert "Eiffel Tower completed in 1889" in keys
    assert "Eiffel Tower designed by Gustave Eiffel" in keys  # relcl takes the noun it modifies
    assert "attracts millions" in " ".join(keys)


def test_build_keys_fine_only_and_targets():
    p = Paragraph("p1", "T", "A b c. D e f.", [(0, 6), (7, 13)])
    sents = sentence_units([p])
    parsed = {sents[0].id: {"clause": ["A b", "c"], "triple": ["A b c"]},
              sents[1].id: {"clause": ["D e f"], "triple": []}}
    keys, targets = build_keys(sents, [p], ["sentence", "clause", "triple"], parsed, fine_only={("p1", 0)})
    assert [k.text for k in keys] == ["T: A b c.", "T: A b", "T: c", "T: A b c", "T: D e f."]
    assert targets == [0, 0, 0, 0, 1]
    assert len({k.id for k in keys}) == len(keys)
    with pytest.raises(ValueError):
        build_keys(sents, [p], ["word"], parsed)


def test_first_hits_dedupes_in_rank_order():
    assert first_hits([3, 0, 1, 2, 4], targets=[0, 0, 1, 1, 2], depth=2) == [1, 0]
