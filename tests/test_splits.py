import random

from ragsplit.splits import overlap_buckets, pilot_slice, split_past_future, split_tuning

QIDS = [f"q{i:05d}" for i in range(1000)]


def test_past_future_sizes_and_disjoint():
    past, future = split_past_future(QIDS, 0.7, 13)
    assert len(past) == 700 and len(future) == 300
    assert set(past).isdisjoint(future)
    assert set(past) | set(future) == set(QIDS)


def test_past_future_deterministic_and_order_independent():
    shuffled = QIDS[:]
    random.Random(0).shuffle(shuffled)
    assert split_past_future(QIDS, 0.7, 13) == split_past_future(shuffled, 0.7, 13)


def test_seed_changes_split():
    assert split_past_future(QIDS, 0.7, 13)[0] != split_past_future(QIDS, 0.7, 14)[0]


def test_tuning_is_subset_of_past():
    past, future = split_past_future(QIDS, 0.7, 13)
    tuning, build = split_tuning(past, 0.1, 13)
    assert len(tuning) == 70 and len(build) == 630
    assert set(tuning).isdisjoint(build)
    assert set(tuning) | set(build) == set(past)
    assert set(tuning).isdisjoint(future)


def test_pilot_is_fixed_subset_of_future():
    _, future = split_past_future(QIDS, 0.7, 13)
    pilot = pilot_slice(future, 200, 13)
    assert len(pilot) == 200 and set(pilot) <= set(future)
    assert pilot == pilot_slice(list(reversed(future)), 200, 13)
    assert pilot_slice(future, 10_000, 13) == sorted(future)


def test_overlap_buckets():
    past_gold = [("p1", 0), ("p1", 2), None]
    future_gold = {"a": ("p1", 0), "b": ("p1", 1), "c": ("p2", 0), "d": None}
    assert overlap_buckets(future_gold, past_gold) == {
        "a": "covered", "b": "uncovered", "c": "uncovered", "d": "unmapped",
    }
