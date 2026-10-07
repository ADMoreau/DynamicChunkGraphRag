import numpy as np

from ragsplit.traverse import GraphTraverser


class FakeRetriever:
    depth = 10

    def __init__(self, ranked):
        self.ranked = ranked

    def search(self, queries):
        return [self.ranked for _ in queries]


def graph(n_hub_links):
    # u0 is retrieved; u1 shares a specific concept with it; u2..: only linked through a hub
    text = [{"id": f"u{i}"} for i in range(3 + n_hub_links)]
    concepts = [{"id": "c_specific", "aliases": ["Kony Ealy"]}, {"id": "c_hub", "aliases": ["United States"]}]
    edges = [{"type": "mentions", "a": "u0", "b": "c_specific"}, {"type": "mentions", "a": "u1", "b": "c_specific"},
             {"type": "mentions", "a": "u0", "b": "c_hub"}]
    edges += [{"type": "mentions", "a": f"u{i}", "b": "c_hub"} for i in range(2, 3 + n_hub_links)]
    return {"text_nodes": text, "concepts": concepts, "edges": edges}


def test_specific_links_beat_hub_links_and_question_concepts_seed():
    g = graph(n_hub_links=20)
    ids = [t["id"] for t in g["text_nodes"]]
    t = GraphTraverser(g, ids, FakeRetriever([0]), alpha=0.5, concept_w=0.0)
    s = t.scores("who is it", [0])
    assert s[0] > s[1] > s[2]                       # the hub spreads its mass thin
    assert t.question_concepts("Which team did Kony Ealy play for?") == {len(ids)}
    t2 = GraphTraverser(g, ids, FakeRetriever([5]), alpha=0.5, concept_w=0.5)
    assert t2.scores("Kony Ealy", [5])[1] > GraphTraverser(g, ids, FakeRetriever([5]), concept_w=0.0).scores("Kony Ealy", [5])[1]
    assert t.search(["q"])[0][0] == 0


class FakeCE:
    """Scores a pair by word overlap between query and text."""

    def predict(self, pairs, **kw):
        return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs]


class FakeReranker:
    top_n = 30
    model = FakeCE()


def test_hop_expander_finds_the_bridge_paragraph():
    from ragsplit.traverse import HopExpander
    from ragsplit.units import Unit

    units = [Unit("u0", "paragraph", "p0", 0, 0, "Eiffel Tower designed by Gustave Eiffel"),
             Unit("u1", "paragraph", "p1", 0, 0, "Gustave Eiffel was born in Dijon"),
             Unit("u2", "paragraph", "p2", 0, 0, "France is a country in Europe")]
    units += [Unit(f"x{i}", "paragraph", f"px{i}", 0, 0, "unrelated text about France") for i in range(60)]
    concepts = [{"id": "c_eiffel", "aliases": ["Gustave Eiffel"]}, {"id": "c_france", "aliases": ["France"]}]
    edges = [{"type": "mentions", "a": "u0", "b": "c_eiffel"}, {"type": "mentions", "a": "u1", "b": "c_eiffel"},
             {"type": "mentions", "a": "u0", "b": "c_france"}, {"type": "mentions", "a": "u2", "b": "c_france"}]
    edges += [{"type": "mentions", "a": f"x{i}", "b": "c_france"} for i in range(60)]   # France: a hub
    g = {"text_nodes": [{"id": u.id} for u in units], "concepts": concepts, "edges": edges}
    r = FakeRetriever([0, 2])
    h = HopExpander(g, units, r, FakeReranker(), n_seeds=1, n_candidates=5, mode="interleave", max_paragraphs=50)
    assert "c_france" not in h.concept_units                  # hub skipped (61 paragraphs > 50)
    out = h.search(["Where was the tower designer born, Eiffel Tower?"])[0]
    assert out[:2] == [0, 1]                                    # seed, then its bridge-linked second hop


def test_beam_traversal_reassembles_and_hops():
    from ragsplit.traverse import BeamTraverser
    from ragsplit.units import Unit

    # p0 is cracked into three pieces; the answer needs the piece after the seed (adjacent)
    # and then a second passage linked by a rare concept (concept hop)
    units = [Unit("a", "s", "p0", 0, 0, "the tower was designed", span=(0, 22)),
             Unit("b", "s", "p0", 0, 0, "by Gustave Eiffel", span=(23, 40)),
             Unit("c", "s", "p0", 0, 0, "unrelated filler text", span=(41, 62)),
             Unit("d", "s", "p1", 0, 0, "Gustave Eiffel was born in Dijon", span=(0, 32)),
             Unit("e", "s", "p2", 0, 0, "a different tower story", span=(0, 23))]
    g = {"text_nodes": [{"id": u.id} for u in units], "concepts": [{"id": "c_ge", "aliases": ["Gustave Eiffel"]}],
         "edges": [{"type": "mentions", "a": "b", "b": "c_ge"}, {"type": "mentions", "a": "d", "b": "c_ge"}]}
    t = BeamTraverser(units, FakeRetriever([0, 4, 2]), FakeReranker(), g, beam=1, depth=3, n_candidates=5)
    out = t.search(["where was the tower designer born"])[0]
    assert out[:3] == [0, 1, 3]                      # seed, adjacent piece, concept hop
    assert t.step_log == {"adjacent": 1, "concept": 1}


def test_beam_flat_stop_only_takes_steps_that_beat_the_next_reranked_unit():
    from ragsplit.traverse import BeamTraverser
    from ragsplit.units import Unit

    # seed a; its neighbour b scores 0 against the question, but the next reranked unit e
    # scores higher, so the flat rule refuses the step and e is packed second
    units = [Unit("a", "s", "p0", 0, 0, "where was the tower designer born", span=(0, 19)),
             Unit("b", "s", "p0", 0, 0, "filler words only", span=(20, 37)),
             Unit("e", "s", "p2", 0, 0, "tower designer was born in Dijon", span=(0, 32))]
    t = BeamTraverser(units, FakeRetriever([0, 2, 1]), FakeReranker(), None, beam=1, depth=2,
                      n_candidates=5, stop="flat")
    assert t.search(["where was the tower designer born"])[0][:2] == [0, 2]
    assert dict(t.step_log) == {}
    loose = BeamTraverser(units, FakeRetriever([0, 2, 1]), FakeReranker(), None, beam=1, depth=2, n_candidates=5)
    assert loose.search(["where was the tower designer born"])[0][:2] == [0, 1]   # old rule takes the weak step
