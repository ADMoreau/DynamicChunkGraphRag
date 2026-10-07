import numpy as np

from ragsplit.data import Answer, Question
from ragsplit.prior import QueryPrior
from ragsplit.run import method_spec, piece_keys_for, reference_qids
from ragsplit.units import Unit


def q(qid, pid, sent, text="?"):
    return Question(qid, text, pid, [Answer("x", 0, sent, sent)], gold_sent=sent)


QS = {
    "a": q("a", "p1", 0), "b": q("b", "p1", 0), "c": q("c", "p1", 1),
    "d": q("d", "p2", 0), "t": q("t", "p3", 0), "f": q("f", "p9", 9),
}
SP = {"past": ["a", "b", "c", "d", "t"], "build": ["a", "b", "c", "d"], "tuning": ["t"], "future": ["f"]}


def cfg(questions):
    return {"eval": {"questions": questions}, "splits": {"seed": 13}, "prior": {"w": {"B4": 0.5, "L+P": 2.0}, "lp_k": 2}}


def test_reference_questions_never_include_future():
    assert reference_qids(cfg("pilot"), SP) == SP["past"]
    assert reference_qids(cfg("tuning"), SP) == SP["build"]


def test_piece_threshold_k():
    keys = lambda k, which: piece_keys_for(method_spec(f"L-k{k}", cfg(which)), cfg(which), SP, QS)
    assert keys(1, "pilot") == {("p1", 0), ("p1", 1), ("p2", 0), ("p3", 0)}
    assert keys(2, "pilot") == {("p1", 0)}
    assert keys(3, "pilot") == set()
    assert ("p3", 0) not in keys(1, "tuning")  # tuning question's own label is held out


def test_method_spec_reads_tuned_values():
    assert method_spec("B4", cfg("pilot"))["w"] == 0.5
    s = method_spec("L+P", cfg("pilot"))
    assert (s["w"], s["k"]) == (2.0, 2)


class FakeEncoder:
    """Embeds a question text '<x>,<y>' as the normalized vector (x, y)."""

    def encode(self, texts, **kw):
        v = np.array([[float(t) for t in s.split(",")] for s in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_prior_ranks_units_by_most_similar_attached_question(tmp_path):
    units = [
        Unit("p1", "paragraph", "p1", 0, 1, "t"),   # holds gold of q1 and q2
        Unit("p2", "paragraph", "p2", 0, 0, "t"),   # holds gold of q3
        Unit("p3", "paragraph", "p3", 0, 0, "t"),   # nothing attached
    ]
    past = [q("q1", "p1", 0, "1,0"), q("q2", "p1", 1, "0,1"), q("q3", "p2", 0, "1,1")]
    prior = QueryPrior(units, past, FakeEncoder(), tmp_path, "fake")
    assert prior.n_attached_units == 2
    assert prior.rank(np.array([0.0, 1.0])) == [0, 1]   # q2 matches exactly -> p1 first
    assert prior.rank(np.array([0.6, 0.8])) == [1, 0]   # closest to q3 -> p2 first


def test_prior_threshold_drops_weak_matches(tmp_path):
    units = [Unit("p1", "paragraph", "p1", 0, 0, "t"), Unit("p2", "paragraph", "p2", 0, 0, "t")]
    past = [q("q1", "p1", 0, "1,0"), q("q2", "p2", 0, "0,1")]
    prior = QueryPrior(units, past, FakeEncoder(), tmp_path, "fake")
    qe = np.array([0.96, 0.28])  # cos 0.96 with q1, 0.28 with q2
    assert prior.rank(qe) == [0, 1]
    assert prior.rank(qe, min_sim=0.9) == [0]
    assert prior.rank(qe, min_sim=0.99) == []



def test_b2_top_n_variant():
    from ragsplit.run import method_spec

    assert method_spec("B2-n5", {}) == {"units": "paragraphs", "prune": True, "prune_top_n": 5}
    assert "prune_top_n" not in method_spec("B2", {})


def test_ratio_ci_paired():
    import numpy as np

    from ragsplit.bench import ratio_ci

    a = np.array([10.0, 20.0, 30.0])
    out = ratio_ci(a, a / 4)
    assert out["ratio"] == 4.0 and out["ci_low"] == out["ci_high"] == 4.0
