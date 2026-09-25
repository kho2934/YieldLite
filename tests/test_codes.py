"""Unit tests for household-code parsing. No data needed."""

import unicodedata

import pytest

from yieldlite.io.codes import (
    looks_like_household_code,
    normalize_text,
    parse_household_code,
)


def test_regular_code():
    code = parse_household_code("01-OM-Long Hưng")
    assert (code.seq, code.district, code.commune) == (1, "OM", "Long Hưng")
    assert code.flag == "certain"
    assert code.district_name == "Ô Môn"


@pytest.mark.parametrize(
    "raw", ["93- VT - Thạnh Lợi", "93-VT-Thạnh Lợi", " 93 - VT-Thạnh  Lợi "]
)
def test_irregular_spacing_is_normalized(raw):
    code = parse_household_code(raw)
    assert (code.seq, code.district, code.commune) == (93, "VT", "Thạnh Lợi")


def test_decomposed_unicode_matches_composed():
    composed = "05-OM-Phước Thới"
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    assert parse_household_code(decomposed).commune == "Phước Thới"


def test_alias_is_applied_and_flagged():
    code = parse_household_code("15-TN-Trung Kiên")
    assert code.commune == "Trung Nhứt"
    assert code.flag == "inferred"
    assert "Report 131" in code.note


def test_unknown_district_is_flagged_not_guessed():
    code = parse_household_code("07-XX-Somewhere")
    assert code.flag == "unresolved"
    assert code.district_name is None


def test_malformed_code_fails_loudly():
    with pytest.raises(ValueError, match="Malformed"):
        parse_household_code("12 OM Long Hưng")


@pytest.mark.parametrize(
    "value", ["Tổng cộng", "BQ p. Long Hưng", "Mã số", "", None, 12]
)
def test_non_household_rows_are_skipped(value):
    assert not looks_like_household_code(value)


def test_household_rows_are_recognized():
    assert looks_like_household_code("93- VT - Thạnh Lợi")


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  Tổng CP phân  ") == "Tổng CP phân"
