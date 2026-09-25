"""Parse household codes from the ĐTGT survey workbooks.

A typical code looks like ``"01-OM-Long Hưng"`` and contains a sequence
number, district code, and commune name. The sequence number only identifies
a row within one survey file. It should not be treated as a permanent
household ID because the same number may represent different households in
different seasons.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

Flag = Literal["certain", "inferred", "unresolved"]

#: These district codes were confirmed in the ĐX 2019-20 and HT 2020 files.
#: Thới Lai appears in later surveys, so its code can be added after checking
#: the 2021-2024 files. Unknown district codes are flagged instead of guessed.
KNOWN_DISTRICTS: dict[str, str] = {
    "OM": "Ô Môn",
    "CD": "Cờ Đỏ",
    "VT": "Vĩnh Thạnh",
    "TN": "Thốt Nốt",
}

#: Some commune names are written differently across the original files.
#: Each key contains the district code and commune name as written, while the
#: value contains the standardized survey commune and the evidence for the change.
COMMUNE_ALIASES: dict[tuple[str, str], tuple[str, str]] = {
    ("TN", "Trung Kiên"): (
        "Trung Nhứt",
        "1 household per file; merging gives Trung Nhứt 15 households like "
        "every commune and reproduces Report 131 (Thốt Nốt 7.10 t/ha)",
    ),
}

_CODE_PATTERN = re.compile(r"^(?P<seq>\d+)-(?P<district>[A-Za-zĐđ]+)-(?P<commune>.+)$")


def normalize_text(text: str) -> str:
    """Normalize Unicode and clean up extra whitespace.

    Vietnamese characters may be stored differently depending on which
    program saved the workbook. Two commune names can look identical on the
    screen but still compare as different strings. Converting everything to
    Unicode NFC prevents that problem. This function also replaces repeated
    whitespace with one space and removes spaces from both ends.

    Parameters
    ----------
    text : str
        Text read directly from a workbook cell.

    Returns
    -------
    str
        Cleaned text stored in Unicode NFC form.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def looks_like_household_code(value: object) -> bool:
    """Check whether a cell appears to contain a household code.

    Household codes begin with a sequence number. Summary rows such as
    ``"Tổng cộng"``, commune averages such as ``"BQ p. Long Hưng"``, headers,
    and blank cells do not begin with a number and are ignored.

    Any text beginning with a digit is sent to
    :func:`parse_household_code`. This makes malformed codes produce a clear
    error instead of disappearing silently during data processing.

    Parameters
    ----------
    value : object
        Value read from the household-code column.

    Returns
    -------
    bool
        True when the value should be treated as a household code.
    """
    return isinstance(value, str) and value.strip()[:1].isdigit()


@dataclass(frozen=True)
class HouseholdCode:
    """Store the information extracted from one ĐTGT household code.

    Attributes
    ----------
    raw : str
        Original cell text exactly as it appeared in the workbook.
    seq : int
        Household sequence number within the survey file.
    district : str
        Short district code, such as ``"OM"``.
    commune : str
        Cleaned commune name after applying any confirmed alias.
    flag : {"certain", "inferred", "unresolved"}
        ``"certain"`` means the code was recognized without changes.
        ``"inferred"`` means a documented commune alias was applied.
        ``"unresolved"`` means the district code is not recognized.
    note : str
        Explanation for an inferred or unresolved value. It is empty when
        the code is certain.
    """

    raw: str
    seq: int
    district: str
    commune: str
    flag: Flag
    note: str = ""

    @property
    def district_name(self) -> str | None:
        """Return the full district name, or None for an unknown code."""
        return KNOWN_DISTRICTS.get(self.district)


def parse_household_code(raw: str) -> HouseholdCode:
    """Parse a household code such as ``"01-OM-Long Hưng"``.

    Spacing around the dashes is inconsistent in the original workbooks. For
    example, a code may appear as ``"93- VT - Thạnh Lợi"``. Cleaning this
    spacing before matching is important because a strict pattern would miss
    all 15 Thạnh Lợi households in the ĐX 2019-20 file.

    Parameters
    ----------
    raw : str
        Household-code text read from the workbook.

    Returns
    -------
    HouseholdCode
        Parsed household information with a data-quality flag and note.

    Raises
    ------
    ValueError
        If the text does not follow the expected
        ``<number>-<district>-<commune>`` structure.
    """
    text = re.sub(r"\s*-\s*", "-", normalize_text(raw))
    match = _CODE_PATTERN.match(text)
    if match is None:
        raise ValueError(f"Malformed household code: {raw!r}")

    district = match["district"].upper()
    commune = match["commune"]
    flag: Flag = "certain"
    note = ""

    if (district, commune) in COMMUNE_ALIASES:
        commune, note = COMMUNE_ALIASES[(district, commune)]
        flag = "inferred"
    if district not in KNOWN_DISTRICTS:
        flag, note = "unresolved", f"unknown district code {district!r}"

    return HouseholdCode(
        raw=raw,
        seq=int(match["seq"]),
        district=district,
        commune=commune,
        flag=flag,
        note=note,
    )
