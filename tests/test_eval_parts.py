from ragsplit.context import SEP, pack
from ragsplit.metrics import exact_match, f1, normalize_answer
from ragsplit.retrieve import rrf
from ragsplit.units import Unit


# Whitespace "tokenizer": one token per word, separator costs 0.
def count(text: str) -> int:
    return len(text.split())


def truncate(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def unit(uid, pid, lo, hi, n_words):
    return Unit(uid, "x", pid, lo, hi, " ".join(f"{uid}{i}" for i in range(n_words)))


def test_pack_fills_in_rank_order():
    us = [unit("a", "p1", 0, 2, 3), unit("b", "p2", 0, 1, 3), unit("c", "p3", 0, 0, 3)]
    out = pack(us, 6, count, truncate)
    assert out.unit_ids == ["a", "b"] and not out.truncated
    assert out.text == us[0].text + SEP + us[1].text


def test_pack_truncates_last_unit():
    us = [unit("a", "p1", 0, 2, 3), unit("b", "p2", 0, 1, 5)]
    out = pack(us, 5, count, truncate)
    assert out.unit_ids == ["a", "b"] and out.truncated
    assert count(out.text) == 5


def test_pack_skips_overlap_with_higher_ranked_unit():
    piece = unit("s", "p1", 1, 1, 2)
    parent = unit("p", "p1", 0, 3, 6)
    other = unit("o", "p2", 0, 0, 2)
    assert pack([piece, parent, other], 100, count, truncate).unit_ids == ["s", "o"]
    assert pack([parent, piece, other], 100, count, truncate).unit_ids == ["p", "o"]


def test_pack_zero_budget():
    assert pack([unit("a", "p", 0, 0, 3)], 0, count, truncate).unit_ids == []


def test_rrf():
    # 3: 1/63 + 1/61 = 0.032266 just beats 2: 2/62 = 0.032258
    assert rrf([[1, 2, 3], [3, 2]]) == [3, 2, 1]
    assert rrf([[1, 2, 3], [2]]) == [2, 1, 3]
    assert rrf([[1], [2]]) == [1, 2]  # ties broken by index


def test_squad_metrics():
    assert normalize_answer("The  Denver Broncos!") == "denver broncos"
    assert exact_match("the Denver Broncos", ["Denver Broncos", "Broncos"]) == 1.0
    assert f1("Denver", ["Denver Broncos"]) == 2 * (1 * 0.5) / 1.5
    assert f1("Panthers", ["Denver Broncos"]) == 0.0


def test_oracle_units_add_pieces_alongside_parents():
    from ragsplit.data import Paragraph
    from ragsplit.units import build_units

    paras = [Paragraph("a00p000", "Big_Cats", "Cats purr. Lions roar.", [(0, 10), (11, 22)]),
             Paragraph("a00p001", "Big_Cats", "Tigers swim.", [(0, 12)])]
    units = build_units("paragraphs+pieces", paras, {("a00p000", 1)})
    assert [(u.id, u.kind) for u in units] == [
        ("a00p000", "paragraph"), ("a00p001", "paragraph"), ("a00p000s01", "piece")]
    assert units[-1].text == "Big Cats: Lions roar."
    assert units[-1].covers() <= units[0].covers()


def _six_sentence_paragraph():
    from ragsplit.data import Paragraph

    ctx = "S0. S1. S2. S3. S4. S5."
    return Paragraph("a00p000", "T", ctx, [(i * 4, i * 4 + 3) for i in range(6)])


def test_split_replaces_node_with_before_sentence_after():
    from ragsplit.units import build_units

    p = _six_sentence_paragraph()
    units = build_units("split", [p], {("a00p000", 2)})
    assert [(u.id, u.kind, u.sent_lo, u.sent_hi) for u in units] == [
        ("a00p000s00-01", "segment", 0, 1),
        ("a00p000s02", "piece", 2, 2),
        ("a00p000s03-05", "segment", 3, 5),
    ]
    assert [u.text for u in units] == ["T: S0. S1.", "T: S2.", "T: S3. S4. S5."]


def test_split_edges_adjacent_and_untouched():
    from ragsplit.data import Paragraph
    from ragsplit.units import build_units

    p = _six_sentence_paragraph()
    other = Paragraph("a00p001", "T", "Whole.", [(0, 6)])
    units = build_units("split", [p, other], {("a00p000", 0), ("a00p000", 3), ("a00p000", 4)})
    assert [(u.sent_lo, u.sent_hi, u.kind) for u in units if u.paragraph_id == "a00p000"] == [
        (0, 0, "piece"), (1, 2, "segment"), (3, 3, "piece"), (4, 4, "piece"), (5, 5, "segment")]
    assert units[-1].id == "a00p001" and units[-1].kind == "paragraph"
    # the parts tile the paragraph exactly: no sentence lost or repeated
    covered = [i for u in units if u.paragraph_id == "a00p000" for i in range(u.sent_lo, u.sent_hi + 1)]
    assert covered == list(range(6))
