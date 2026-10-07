import numpy as np

from ragsplit.merge import (UnionFind, alias_tier, exact_tier, meaning_tier, negated, normalize,
                            numbers)


def m(text, label, article="A", pid="p"):
    return {"pid": pid, "article": article, "s": 0, "e": len(text), "text": text, "label": label}


def test_normalize_numbers_negation():
    assert normalize("The Denver Broncos'") == normalize("Denver Broncos") == "denver broncos"
    assert normalize("1,000 people") == "1000 people"
    assert numbers("won 24–10 in 2016") == {"24", "10", "2016"}
    assert negated("did not win") and negated("didn't win") and not negated("won the title")


def test_exact_and_alias_tiers():
    ms = [m("Denver Broncos", "ORG"), m("the Denver Broncos", "ORG"), m("Broncos", "ORG"),
          m("National Football League", "ORG"), m("NFL", "ORG"),
          m("Gustave Eiffel", "PERSON"), m("Eiffel", "PERSON"),
          m("Broncos", "ORG", article="B"),                        # other article: no alias
          m("Denver", "GPE")]                                      # different type group
    uf = UnionFind(len(ms))
    assert exact_tier(ms, uf) == [(0, 1), (2, 7)]                  # exact merges cross articles
    rules = {(a, b): r for a, b, r in alias_tier(ms, uf)}
    assert rules[(3, 4)] == "acronym" and rules[(5, 6)] == "short form"
    assert uf.find(2) == uf.find(0)                                # "Broncos" -> "Denver Broncos"
    assert uf.find(8) != uf.find(0)


def test_alias_needs_a_unique_long_form():
    ms = [m("John Smith", "PERSON"), m("Jane Smith", "PERSON"), m("Smith", "PERSON")]
    uf = UnionFind(3)
    assert alias_tier(ms, uf) == []


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_meaning_tier_guards_and_no_chaining():
    # a~b and b~c pass the threshold but a and c are far apart: only one merge may happen
    a, b, c = unit([1, 0]), unit([np.cos(0.5), np.sin(0.5)]), unit([np.cos(1.0), np.sin(1.0)])
    vecs = np.stack([a, b, c])
    uf, merges = meaning_tier(vecs, ["x", "y", "z"], ["g"] * 3, tau=0.85)
    assert len(merges) == 1 and len(uf.clusters()) == 2
    # identical vectors but different numbers, negation or group never merge
    v = np.stack([a, a, a, a])
    uf, merges = meaning_tier(v, ["won in 1889", "won in 1887", "did not win in 1889", "won in 1889"],
                              ["g", "g", "g", "h"], tau=0.9)
    assert merges == []
