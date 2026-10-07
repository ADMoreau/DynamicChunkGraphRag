import pytest

from ragsplit.config import load_config


def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("stage: prepare\nsplits:\n  seed: 13\n")
    cfg = load_config(p)
    assert cfg["stage"] == "prepare" and cfg["splits"]["seed"] == 13


def test_load_config_requires_stage(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("splits: {}\n")
    with pytest.raises(ValueError, match="stage"):
        load_config(p)


def test_repo_configs_load():
    from pathlib import Path

    for path in Path(__file__).parents[1].glob("configs/*.yaml"):
        load_config(path)


def test_document_scope_keeps_only_the_questions_document():
    from types import SimpleNamespace

    from ragsplit.run import doc_of, scoped
    assert doc_of("s04p021") == "s04" and doc_of("a00p000") == "a00" and doc_of("h00012") == "h00012"
    units = [SimpleNamespace(paragraph_id=p) for p in ("s04p001", "s07p002", "s04p009")]
    q = SimpleNamespace(paragraph_id="s04p021")
    assert [u.paragraph_id for u in scoped(units, q, {"eval": {"scope": "document"}})] == ["s04p001", "s04p009"]
    assert len(scoped(units, q, {"eval": {}})) == 3
