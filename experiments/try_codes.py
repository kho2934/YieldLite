"""Scratch space: not part of the tested package or test suite.

Use this file (or copies of it) to try things out on real data without
touching tests/ or src/.
"""

from yieldlite.io.codes import parse_household_code

code = parse_household_code("93- VT - Thạnh Lợi")
print(code)
print("district:", code.district_name)
