from ragsplit.syllabus import find_span


def test_find_span_ignores_case_punctuation_and_spacing():
    text = "Grading: Homework (30%), Midterm - 30% and Final 40%. Late work loses 10%."
    s, e = find_span(text, "homework 30% midterm 30%")
    assert text[s:e] == "Homework (30%), Midterm - 30"
    assert find_span(text, "attendance policy") is None
