from ragsplit.graph import build_graph, merges_by_string
from ragsplit.units import Unit


def m(text, label, pid, s):
    return {"pid": pid, "article": "A", "s": s, "e": s + len(text), "text": text, "label": label}


def test_string_merge_rules():
    assert merges_by_string(m("Denver Broncos", "ORG", "p", 0))
    assert merges_by_string(m("1994", "DATE", "p", 0)) and merges_by_string(m("the 1960s", "DATE", "p", 0))
    assert not merges_by_string(m("the previous year", "DATE", "p", 0))
    assert not merges_by_string(m("300", "CARDINAL", "p", 0)) and not merges_by_string(m("first", "ORDINAL", "p", 0))


def unit(pid, s, e, text):
    return Unit(f"{pid}c{s:04d}-{e:04d}", "segment", pid, 0, 0, text, span=(s, e))


def test_build_graph_merges_links_and_adjacency():
    units = [unit("p", 0, 20, "T: Denver Broncos won"), unit("p", 21, 40, "T: 300 games in 1994"),
             unit("q", 0, 30, "U: The Broncos had 300 fans")]
    mentions = [m("Denver Broncos", "ORG", "p", 0), m("300", "CARDINAL", "p", 21), m("1994", "DATE", "p", 34),
                m("The Broncos", "ORG", "q", 0), m("300", "CARDINAL", "q", 19)]
    cands = {0: {"source": "embed", "a": mentions[0], "b": mentions[3]},          # judged SAME
             1: {"source": "exact", "a": mentions[1], "b": mentions[4]},          # judged DIFFERENT
             2: {"source": "alias", "a": mentions[2], "b": mentions[1]}}          # judged RELATED (odd, but typed)
    g = build_graph(units, mentions, [], cands, {0: "SAME", 1: "DIFFERENT", 2: "RELATED"})
    names = {c["name"]: c for c in g["concepts"]}
    assert names["Denver Broncos"]["n_mentions"] == 2                  # merged by verdict
    assert sum(c["name"] == "300" for c in g["concepts"]) == 2         # numbers never merge by string
    types = [e["type"] for e in g["edges"]]
    assert types.count("adjacent") == 1 and types.count("related") == 1
    broncos = names["Denver Broncos"]["id"]
    linked = {e["a"] for e in g["edges"] if e["type"] == "mentions" and e["b"] == broncos}
    assert linked == {units[0].id, units[2].id}                        # one node links both paragraphs


def test_veto_and_lone_person_names():
    units = [unit("p", 0, 30, "T: John met SAT"), unit("q", 0, 30, "U: John took the SAT")]
    mentions = [m("John", "PERSON", "p", 0), m("SAT", "ORG", "p", 9), m("John", "PERSON", "q", 0), m("SAT", "ORG", "q", 16)]
    mentions[2]["article"] = "B"; mentions[3]["article"] = "B"
    cands = {0: {"source": "exact", "group": "org", "a": mentions[1], "b": mentions[3]}}
    g = build_graph(units, mentions, [], cands, {0: "DIFFERENT"})
    assert sum(c["name"] == "SAT" for c in g["concepts"]) == 2      # vetoed group stays apart
    assert sum(c["name"] == "John" for c in g["concepts"]) == 2     # lone names don't cross articles


def test_cannot_link_stops_chaining():
    units = [unit("p", 0, 60, "T: Luther led the Reformation, Martin Luther wrote")]
    mentions = [m("Luther", "PERSON", "p", 0), m("Martin Luther", "PERSON", "p", 30), m("Reformation", "EVENT", "p", 15)]
    cands = {0: {"source": "embed", "cos": 0.9, "a": mentions[0], "b": mentions[1]},   # SAME
             1: {"source": "embed", "cos": 0.85, "a": mentions[0], "b": mentions[2]},  # SAME (a chain link)
             2: {"source": "embed", "cos": 0.82, "a": mentions[1], "b": mentions[2]}}  # RELATED: blocks the chain
    g = build_graph(units, mentions, [], cands, {0: "SAME", 1: "SAME", 2: "RELATED"})
    names = sorted(tuple(sorted(c["aliases"])) for c in g["concepts"])
    assert names == [("Luther", "Martin Luther"), ("Reformation",)]
    assert g["merges_blocked_by_constraints"] == {"verdict": 1}


def test_type_guard_blocks_person_company_merge():
    units = [unit("p", 0, 60, "T: Tesla founded Tesla Electric Light & Manufacturing")]
    mentions = [m("Tesla", "PERSON", "p", 0), m("Tesla Electric Light & Manufacturing", "ORG", "p", 14)]
    cands = {0: {"source": "alias", "cos": None, "a": mentions[1], "b": mentions[0]}}
    g = build_graph(units, mentions, [], cands, {0: "SAME"})
    assert len(g["concepts"]) == 2 and g["merges_blocked_by_constraints"] == {"verdict: type": 1}
