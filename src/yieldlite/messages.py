"""Turn yieldlite decisions into clear English and Vietnamese messages.

Yieldlite stores each decision as a short code together with the values used
to make it. This module turns that information into an explanation that a
farmer, extension worker, or researcher can understand.

Every message in ``CATALOG`` has an English and a Vietnamese version. Both
versions use the same placeholders so they always report the same result.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

Lang = Literal["en", "vi"]

LANGUAGES = ("en", "vi")
DEFAULT_LANG = "en"

# Readable names for the models. These replace codes such as "m1" so the
# explanation never expects the user to know the internal model names.
RUNG_NAMES: dict[str, dict[str, str]] = {
    "m0": {
        "en": "the season-and-district average (M0)",
        "vi": "mức năng suất trung bình theo vụ và huyện (M0)",
    },
    "m1": {
        "en": "the fertilizer-and-variety regression (M1)",
        "vi": "mô hình hồi quy dựa trên phân bón và giống lúa (M1)",
    },
    "m2": {
        "en": "the weather model (M2)",
        "vi": "mô hình có sử dụng dữ liệu thời tiết (M2)",
    },
    "m3": {
        "en": "the boosted-tree model (M3)",
        "vi": "mô hình cây tăng cường (M3)",
    },
}

_RUNG_FIELDS = ("rung", "simpler", "needs")

CATALOG: dict[str, dict[Lang, str]] = {
    "result_estimate": {
        "en": (
            "Expected yield: {estimate:.1f} t/ha, most likely between "
            "{lower:.1f} and {upper:.1f} t/ha ({level}% range)."
        ),
        "vi": (
            "Năng suất dự kiến: {estimate:.1f} t/ha, với khoảng dự đoán "
            "từ {lower:.1f} đến {upper:.1f} t/ha (mức {level}%)."
        ),
    },
    "result_history": {
        "en": (
            "Past yields for {district} in {season} were usually between "
            "{lower:.1f} and {upper:.1f} t/ha "
            "({level}% of past records)."
        ),
        "vi": (
            "Năng suất trước đây tại huyện {district} trong vụ {season} "
            "thường nằm trong khoảng từ {lower:.1f} đến {upper:.1f} t/ha "
            "(bao gồm {level}% số liệu lịch sử)."
        ),
    },
    # Reasons for refusing or placing a warning on a prediction.
    "too_few_panels": {
        "en": (
            "There are only {panels} seasons of data, and at least {needed} "
            "are needed for a reliable prediction, so no prediction is made."
        ),
        "vi": (
            "Chỉ có {panels} vụ dữ liệu, trong khi cần ít nhất {needed} vụ "
            "để dự đoán đáng tin cậy, nên không đưa ra dự đoán."
        ),
    },
    "too_few_panels_unknown": {
        "en": (
            "There are only {panels} seasons of data. That is not enough for "
            "a reliable prediction, so no prediction is made."
        ),
        "vi": (
            "Hiện chỉ có dữ liệu của {panels} vụ, chưa đủ để đưa ra dự đoán "
            "đáng tin cậy. Vì vậy, hệ thống không thực hiện dự đoán."
        ),
    },
    "unknown_district": {
        "en": (
            "District {district} does not appear in the training data, so no "
            "prediction can be made for it."
        ),
        "vi": (
            "Dữ liệu huấn luyện không có huyện {district}, vì vậy hệ thống "
            "không thể đưa ra dự đoán cho huyện này."
        ),
    },
    "season_unresolved": {
        "en": "The information provided is not enough to identify the season.",
        "vi": "Thông tin được cung cấp chưa đủ để xác định vụ lúa.",
    },
    "npk_out_of_range": {
        "en": (
            "The fertilizer amount, {npk} kg/ha, is outside the range found "
            "in the training data ({low} to {high} kg/ha). The estimate may "
            "therefore be unreliable."
        ),
        "vi": (
            "Tổng lượng phân bón là {npk} kg/ha, nằm ngoài khoảng đã xuất "
            "hiện trong dữ liệu huấn luyện ({low} đến {high} kg/ha). Vì vậy, "
            "kết quả ước tính có thể không đáng tin cậy."
        ),
    },
    "npk_unit_unresolved": {
        "en": (
            "The fertilizer units could not be identified. Enter N, P2O5, "
            "and K2O as kilograms of pure nutrient per hectare."
        ),
        "vi": (
            "Không xác định được đơn vị của các giá trị phân bón. Vui lòng "
            "nhập N, P2O5 và K2O theo kg dưỡng chất nguyên chất trên mỗi hecta."
        ),
    },
    "level_unsupported": {
        "en": (
            "A {level}% prediction interval needs at least {needed} seasons "
            "of data, but only {panels} are available. Choose a lower level "
            "or add more data."
        ),
        "vi": (
            "Khoảng dự đoán {level}% cần dữ liệu của ít nhất {needed} vụ, "
            "nhưng hiện chỉ có {panels} vụ. Hãy chọn mức thấp hơn hoặc bổ "
            "sung thêm dữ liệu."
        ),
    },
    "interval_uninformative": {
        "en": (
            "The prediction interval is {width} t/ha wide, which is no "
            "narrower than the usual historical range of {usual} t/ha. The "
            "historical range is shown instead."
        ),
        "vi": (
            "Khoảng dự đoán rộng {width} t/ha, không hẹp hơn khoảng biến động "
            "năng suất thường thấy trước đây là {usual} t/ha. Vì vậy, hệ "
            "thống hiển thị khoảng năng suất lịch sử thay cho dự đoán này."
        ),
    },
    "variety_pooled": {
        "en": (
            "There are too few records for variety {variety}, so it was "
            "combined with other varieties."
        ),
        "vi": (
            "Giống {variety} có quá ít dữ liệu riêng, nên được gộp vào nhóm "
            "các giống khác."
        ),
    },
    "history_thin": {
        "en": (
            "There are too few past records for {district} during {season} "
            "to estimate its usual yield range."
        ),
        "vi": (
            "Không có đủ dữ liệu năng suất trước đây của huyện {district} "
            "trong vụ {season} để xác định khoảng năng suất thường thấy."
        ),
    },

    # Explanations from the model-selection policy.
    "rung_selected": {
        "en": (
            "Using {rung}. When each of the {compared} seasons was held out "
            "in turn, it predicted an average of {gain:.2f} t/ha better than "
            "{simpler} (standard error {se:.2f})."
        ),
        "vi": (
            "Chọn {rung}. Khi lần lượt giữ lại từng vụ trong tổng số "
            "{compared} vụ để kiểm tra, mô hình này dự đoán tốt hơn {simpler} "
            "trung bình {gain:.2f} t/ha, với sai số chuẩn là {se:.2f}."
        ),
    },
    "rung_selected_simplest": {
        "en": (
            "Using {rung}, the simplest model. With {panels} seasons of data, "
            "none of the more complex models was clearly better."
        ),
        "vi": (
            "Chọn {rung}, là mô hình đơn giản nhất. Với dữ liệu của {panels} "
            "vụ, chưa có mô hình phức tạp hơn nào cho kết quả tốt hơn một "
            "cách rõ ràng."
        ),
    },
    "no_gain": {
        "en": (
            "{rung} was not used. It improved on {simpler} by only "
            "{gain:.2f} t/ha (standard error {se:.2f}), so the improvement "
            "was not clear enough."
        ),
        "vi": (
            "Không chọn {rung}. So với {simpler}, mô hình này chỉ cải thiện "
            "{gain:.2f} t/ha, với sai số chuẩn {se:.2f}. Mức cải thiện này "
            "chưa đủ rõ ràng để sử dụng mô hình phức tạp hơn."
        ),
    },
    "gain_not_assessable": {
        "en": (
            "{rung} could not be compared reliably with {simpler}. Only "
            "{compared} seasons were available for testing, and at least 2 "
            "are needed."
        ),
        "vi": (
            "Chưa thể so sánh đáng tin cậy giữa {rung} và {simpler}. Chỉ có "
            "{compared} vụ phù hợp để kiểm tra, trong khi cần ít nhất 2 vụ."
        ),
    },
    "rung_not_evaluated": {
        "en": (
            "{rung} was not used because the amount of data it needs has not "
            "yet been evaluated."
        ),
        "vi": (
            "Không chọn {rung} vì chưa có nghiên cứu xác định mô hình này "
            "cần bao nhiêu dữ liệu để hoạt động đáng tin cậy."
        ),
    },
    "rung_not_reached": {
        "en": (
            "{rung} was not used. Within the tested data sizes, it did not "
            "clearly outperform the simpler models."
        ),
        "vi": (
            "Không chọn {rung}. Trong phạm vi lượng dữ liệu đã thử nghiệm, "
            "mô hình này chưa cho kết quả tốt hơn rõ ràng so với các mô hình "
            "đơn giản hơn."
        ),
    },
    "rung_needs_more_panels": {
        "en": (
            "{rung} was not used. It needs at least {needed} seasons of data, "
            "but only {have} are available."
        ),
        "vi": (
            "Không chọn {rung}. Mô hình này cần dữ liệu của ít nhất {needed} "
            "vụ, nhưng hiện chỉ có {have} vụ."
        ),
    },
    "rung_needs_more_households": {
        "en": (
            "{rung} was not used. It needs at least {needed} households in "
            "each district-season group, while a typical group has {have}."
        ),
        "vi": (
            "Không chọn {rung}. Mô hình này cần ít nhất {needed} hộ trong mỗi "
            "nhóm huyện–vụ, nhưng một nhóm điển hình hiện chỉ có {have} hộ."
        ),
    },
    "rung_needs_more_contrasts": {
        "en": (
            "{rung} was not used. It needs at least {needed} year-to-year "
            "comparisons within the same season, but only {have} are available."
        ),
        "vi": (
            "Không chọn {rung}. Mô hình này cần ít nhất {needed} phép so sánh "
            "giữa các năm trong cùng một vụ, nhưng hiện chỉ có {have}."
        ),
    },
    "rung_needs_more_rows": {
        "en": (
            "{rung} was not used. It needs at least {needed} records, but only "
            "{have} are available."
        ),
        "vi": (
            "Không chọn {rung}. Mô hình này cần ít nhất {needed} bản ghi dữ "
            "liệu, nhưng hiện chỉ có {have}."
        ),
    },
    "rung_blocked": {
        "en": (
            "{rung} was not used because it builds on {needs}, which the "
            "current data cannot support."
        ),
        "vi": (
            "Không chọn {rung} vì mô hình này được xây dựng dựa trên {needs}, "
            "trong khi dữ liệu hiện tại chưa đủ để sử dụng mô hình đó."
        ),
    },
}


def _variant(code: str, params: dict[str, Any]) -> str:
    """Choose the right message variant for the available values."""
    if code == "too_few_panels" and params.get("needed") is None:
        return "too_few_panels_unknown"

    if code == "rung_selected" and "simpler" not in params:
        return "rung_selected_simplest"

    return code


def _display(params: dict[str, Any], lang: str) -> dict[str, Any]:
    """Replace internal model codes with readable model names."""
    values = dict(params)

    for name in _RUNG_FIELDS:
        value = values.get(name)
        if isinstance(value, str) and value in RUNG_NAMES:
            values[name] = RUNG_NAMES[value][lang]

    return values


def _capitalize(text: str) -> str:
    """Make sure the rendered message starts with a capital letter."""
    return text[:1].upper() + text[1:]


def render(
    code: str,
    lang: str = DEFAULT_LANG,
    **params: Any,
) -> str:
    """Turn one decision code into a readable message.

    Unknown languages fall back to English. If the message cannot be filled
    because a required value is missing or invalid, the function tries the
    English version. If that also fails, it returns the original code instead
    of raising an error.

    Parameters
    ----------
    code : str
        Message code, such as ``"no_gain"``.
    lang : {"en", "vi"}, default "en"
        Language used for the message.
    **params
        Values inserted into the message. Percentage levels should be passed
        as whole percentages, such as 80 instead of 0.8.

    Returns
    -------
    str
        The completed message, or the original code if it cannot be rendered.
    """
    entry = CATALOG.get(_variant(code, params))
    if entry is None:
        return str(code)

    language = lang if lang in LANGUAGES else DEFAULT_LANG

    for candidate in (language, DEFAULT_LANG):
        try:
            text = entry[candidate].format(
                **_display(params, candidate)
            )
        except (
            KeyError,
            IndexError,
            ValueError,
            TypeError,
            AttributeError,
        ):
            continue

        return _capitalize(text)

    return str(code)


def render_reasons(
    reasons: Iterable[Any],
    lang: str = DEFAULT_LANG,
) -> list[str]:
    """Render several reasons in the order they were recorded.

    Each item must have a ``code`` attribute and a ``params`` attribute, as
    provided by ``yieldlite.selection.Reason``.
    """
    return [
        render(reason.code, lang, **reason.params)
        for reason in reasons
    ]
