"""Tests for the plain-language messages."""

import dataclasses
import math
import string

import pytest

from yieldlite.messages import (
    CATALOG,
    LANGUAGES,
    RUNG_NAMES,
    render,
    render_reasons,
)
from yieldlite.profile import DataProfile
from yieldlite.selection import PanelGain, SelectionPolicy

# One set of example values for every message.
_SAMPLE = {
    "too_few_panels": {"panels": 3, "needed": 4},
    "too_few_panels_unknown": {"panels": 3},
    "unknown_district": {"district": "XX"},
    "season_unresolved": {},
    "npk_out_of_range": {"npk": 900, "low": 100, "high": 300},
    "npk_unit_unresolved": {},
    "level_unsupported": {"level": 90, "needed": 9, "panels": 5},
    "interval_uninformative": {"width": 2.3, "usual": 1.1},
    "variety_pooled": {"variety": "OM18"},
    "history_thin": {"district": "CD", "season": "DX"},
    "rung_selected": {
        "rung": "m1",
        "simpler": "m0",
        "gain": 0.04,
        "se": 0.01,
        "compared": 8,
        "panels": 8,
    },
    "rung_selected_simplest": {"rung": "m0", "panels": 8},
    "no_gain": {"rung": "m1", "simpler": "m0", "gain": 0.01, "se": 0.01},
    "gain_not_assessable": {"rung": "m1", "simpler": "m0", "compared": 1},
    "rung_not_evaluated": {"rung": "m2"},
    "rung_not_reached": {"rung": "m3"},
    "rung_needs_more_panels": {"rung": "m1", "have": 3, "needed": 5},
    "rung_needs_more_households": {"rung": "m1", "have": 12.5, "needed": 20},
    "rung_needs_more_contrasts": {"rung": "m2", "have": 2, "needed": 5},
    "rung_needs_more_rows": {"rung": "m3", "have": 900, "needed": 2000},
    "rung_blocked": {"rung": "m2", "needs": "m1"},
}


def _fields(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_the_sample_values_cover_every_message():
    assert set(_SAMPLE) == set(CATALOG)


def test_every_message_exists_in_both_languages():
    for code, entry in CATALOG.items():
        assert set(entry) == set(LANGUAGES), code
        assert all(entry[lang].strip() for lang in LANGUAGES), code


def test_both_languages_use_the_same_placeholders():
    for code, entry in CATALOG.items():
        assert _fields(entry["en"]) == _fields(entry["vi"]), code


def test_the_sample_values_fill_every_placeholder():
    for code, entry in CATALOG.items():
        assert _fields(entry["en"]) <= set(_SAMPLE[code]), code


def test_every_message_renders_cleanly_in_both_languages():
    for code, params in _SAMPLE.items():
        for lang in LANGUAGES:
            text = render(code, lang, **params)

            assert text != code
            assert "{" not in text and "}" not in text
            assert text[0].isupper()


def test_the_two_languages_give_different_text():
    for code, params in _SAMPLE.items():
        assert render(code, "en", **params) != render(code, "vi", **params), code


def test_vietnamese_text_keeps_its_diacritics():
    text = render("too_few_panels", "vi", panels=3, needed=4)

    assert text == (
        "Chỉ có 3 vụ dữ liệu, trong khi cần ít nhất 4 vụ để dự đoán đáng "
        "tin cậy, nên không đưa ra dự đoán."
    )


def test_english_is_the_default_language():
    assert render("season_unresolved") == render("season_unresolved", "en")


def test_an_unknown_language_falls_back_to_english():
    assert render("season_unresolved", "fr") == render("season_unresolved", "en")


def test_model_codes_become_readable_names():
    text = render("rung_not_evaluated", "en", rung="m2")

    assert " m2 " not in text
    assert RUNG_NAMES["m2"]["en"][1:] in text

    assert RUNG_NAMES["m2"]["vi"] in render("rung_not_evaluated", "vi", rung="m2")


def test_numbers_are_formatted_in_the_sentence():
    text = render("no_gain", "en", **_SAMPLE["no_gain"])

    assert "0.01 t/ha" in text
    assert "standard error 0.01" in text


def test_a_refusal_without_a_known_requirement_has_its_own_wording():
    known = render("too_few_panels", "en", panels=3, needed=4)
    unknown = render("too_few_panels", "en", panels=3, needed=None)

    assert "at least 4" in known
    assert "None" not in unknown
    assert "3 seasons" in unknown


def test_the_simplest_model_has_its_own_wording():
    text = render("rung_selected", "en", rung="m0", panels=8)

    assert "simplest" in text
    assert "8 seasons" in text


def test_an_unknown_code_returns_the_code():
    assert render("no_such_code", "en") == "no_such_code"
    assert render("no_such_code", "vi", panels=3) == "no_such_code"


@pytest.mark.parametrize("lang", ["en", "vi"])
def test_missing_values_never_raise(lang):
    for code in CATALOG:
        text = render(code, lang)

        assert isinstance(text, str)
        assert text


def test_values_of_the_wrong_type_never_raise():
    text = render("no_gain", "en", rung="m1", simpler="m0", gain="big", se=None)

    assert isinstance(text, str)


def test_reasons_are_written_in_order():
    class _Reason:
        def __init__(self, code, params):
            self.code = code
            self.params = params

    reasons = [
        _Reason("rung_not_evaluated", {"rung": "m3"}),
        _Reason("rung_selected", {"rung": "m0", "panels": 8}),
    ]

    assert render_reasons(reasons, "en") == [
        render("rung_not_evaluated", "en", rung="m3"),
        render("rung_selected", "en", rung="m0", panels=8),
    ]


# Every reason the selection policy can produce has a message.

_WIN = PanelGain(mean=0.04, se=0.01, n_panels=8)
_TIE = PanelGain(mean=0.01, se=0.01, n_panels=8)
_UNKNOWN = PanelGain(mean=math.nan, se=math.nan, n_panels=0)


def _document(**changes):
    entries = {
        "P0": {"status": "reached", "value": 4},
        "P1": {"status": "reached", "value": 3},
        "H1": {"status": "reached", "value": 20},
        "W2": {"status": "not_evaluated", "value": None},
        "N3": {"status": "not_evaluated", "value": None},
        "P3": {"status": "not_evaluated", "value": None},
    }
    entries.update(changes)
    return {"thresholds": entries}


def _profile(**changes):
    profile = DataProfile(
        n_rows=1327,
        n_panels=8,
        n_seasons=3,
        panels_per_season={"DX": 3, "HT": 2, "TD": 3},
        median_households_per_cell=40.0,
        within_season_contrasts=5,
        rows_per_variety={"pooled": 100},
        flagged_share=0.0,
    )
    return dataclasses.replace(profile, **changes)


def _reached(value):
    return {"status": "reached", "value": value}


def _gain_by(results):
    return lambda better, baseline: results[(better, baseline)]


def test_every_selection_reason_has_a_message():
    scenarios = [
        (_document(), _profile(n_panels=3), None),
        (
            _document(P0={"status": "not_reached", "value": None}),
            _profile(),
            None,
        ),
        (_document(), _profile(), _gain_by({("m1", "m0"): _WIN})),
        (_document(), _profile(), _gain_by({("m1", "m0"): _TIE})),
        (_document(), _profile(), _gain_by({("m1", "m0"): _UNKNOWN})),
        (_document(), _profile(median_households_per_cell=5.0), None),
        (_document(P1=_reached(9)), _profile(), None),
        (
            _document(P1={"status": "not_reached", "value": None}),
            _profile(),
            None,
        ),
        (
            _document(W2=_reached(10)),
            _profile(),
            _gain_by({("m1", "m0"): _WIN}),
        ),
        (
            _document(P1=_reached(9), W2=_reached(5)),
            _profile(),
            None,
        ),
        (
            _document(N3=_reached(5000), P3=_reached(20)),
            _profile(),
            _gain_by({("m1", "m0"): _WIN}),
        ),
    ]

    seen = set()

    for document, profile, gain in scenarios:
        selection = SelectionPolicy(document).select(profile, gain)

        for reason in selection.reasons:
            seen.add(reason.code)

            for lang in LANGUAGES:
                text = render(reason.code, lang, **reason.params)

                assert text != reason.code, reason
                assert "{" not in text, reason

    assert {
        "too_few_panels",
        "rung_selected",
        "no_gain",
        "gain_not_assessable",
        "rung_not_evaluated",
        "rung_not_reached",
        "rung_needs_more_panels",
        "rung_needs_more_households",
        "rung_needs_more_contrasts",
        "rung_needs_more_rows",
        "rung_blocked",
    } <= seen
