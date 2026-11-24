"""Tiny offline optimizer and artifact checks, not a toxicity benchmark."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")


def test_actual_training_updates_weights_and_predicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math

    import torch
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertConfig,
        DistilBertForSequenceClassification,
    )

    from toxic_comments.cli import classify
    from toxic_comments.training import run_training, training_config

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    base = tmp_path / "base"
    base.mkdir()
    vocab = base / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nkind\nrude\ncomment\n"
        + "\n".join(str(i) for i in range(20))
        + "\n"
    )
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    tokenizer.save_pretrained(base)
    model = DistilBertForSequenceClassification(
        DistilBertConfig(
            vocab_size=len(tokenizer),
            n_layers=1,
            dim=16,
            hidden_dim=32,
            n_heads=2,
            max_position_embeddings=128,
            dropout=0.0,
            attention_dropout=0.0,
            **training_config(),
        )
    )
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.save_pretrained(base, safe_serialization=True)
    rows = [
        {"comment_text": f"{'kind' if i % 2 == 0 else 'rude'} comment {i}", "toxic": i % 2}
        for i in range(20)
    ]
    output = tmp_path / "trained"
    report = run_training(rows, base, output, epochs=1, batch_size=4)
    assert math.isfinite(report["eval_loss"])
    trained = DistilBertForSequenceClassification.from_pretrained(output, local_files_only=True)
    assert any(not torch.equal(before[k], v) for k, v in trained.state_dict().items())
    assert (output / "model.safetensors").is_file()
    prediction = classify("hello kind comment", output)
    assert prediction["label"] in {"toxic", "non_toxic"}
    assert 0 <= prediction["score"] <= 1
