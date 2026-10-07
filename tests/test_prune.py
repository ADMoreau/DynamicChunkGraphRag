import pytest

pytest.importorskip("nltk")

from ragsplit.prune import kept_original_text


def lossy_decode(sent: str, first: bool) -> str:
    """Stand-in for Provence's round trip: drops '½' like its tokenizer does."""
    return ("" if first else " ") + sent.replace("½", "")


def needs_punkt():
    from pathlib import Path

    import nltk

    nltk.data.path.insert(0, str(Path(__file__).parents[1] / "data" / "models" / "nltk"))
    try:
        nltk.sent_tokenize("A. B.")
    except LookupError:
        pytest.skip("nltk punkt_tab not installed")


CTX = "Miller had 2½ sacks. Denver won. Manning retired."


def test_recovers_original_text_of_kept_sentences():
    needs_punkt()
    pruned = "Miller had 2 sacks. Manning retired."
    assert kept_original_text(CTX, pruned, lossy_decode) == "Miller had 2½ sacks. Manning retired."


def test_nothing_kept():
    needs_punkt()
    assert kept_original_text(CTX, "", lossy_decode) == ""
