from analysis.audit_geocode_rebuild import audit_flags
from analysis.rebuild_geocode_cache import clean_query, normalize_target


def test_target_normalization_matches_cache_key_convention():
    assert normalize_target("  New York  ") == "new york"
    assert clean_query("St. John's (London)") == "St. Johns London"


def test_audit_flags_separate_empty_and_reviewable_results():
    assert audit_flags("missing place", None) == ["no_result"]

    flags = audit_flags(
        "021",
        {
            "category": "amenity",
            "place_rank": 30,
            "importance": 0.0001,
        },
    )
    assert "rank30_poi" in flags
    assert "very_low_importance" in flags
    assert "numeric_query" in flags
