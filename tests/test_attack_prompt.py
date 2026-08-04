from data_collector.generate_attacks import build_attack_prompt
from evaluation.prompts import (
    ATTACK_GENERATION_PROMPT,
    build_attack_generation_prompt,
)


def test_construction_uses_shared_attack_prompt_template():
    location = {"city": "Paris", "province": "Ile-de-France", "country": "France"}

    assert build_attack_prompt(location) == build_attack_generation_prompt(location)


def test_paper_prompt_card_discloses_dynamic_ground_truth_fields():
    assert "Ground-truth location:" in ATTACK_GENERATION_PROMPT
    assert "<city>" in ATTACK_GENERATION_PROMPT
    assert "different language/script" in ATTACK_GENERATION_PROMPT
    assert "different continent" in ATTACK_GENERATION_PROMPT
