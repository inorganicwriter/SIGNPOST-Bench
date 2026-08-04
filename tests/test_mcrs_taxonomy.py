import json

from evaluation.mcrs import load_taxonomy_targets


def test_load_taxonomy_targets_uses_sample_adversarial_text(tmp_path):
    dataset_dir = tmp_path / "sample"
    dataset_dir.mkdir()
    rows = [
        {"base_id": "a", "adversarial_text": " Tokyo "},
        {"base_id": "b", "adversarial_text": ""},
        {"base_id": "c", "adversarial_text": "Rue de Rivoli"},
    ]
    path = dataset_dir / "taxonomy_labels.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    assert load_taxonomy_targets(tmp_path, "sample") == {
        "a": "tokyo",
        "c": "rue de rivoli",
    }


def test_load_taxonomy_targets_missing_file_is_empty(tmp_path):
    assert load_taxonomy_targets(tmp_path, "missing") == {}
