"""Rule-based grouping for AmE/BrE variant pairs."""

from __future__ import annotations

import re
from typing import Tuple


def lev_dist(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


GROUP_DEFINITIONS = {
    "Group 1 (-or vs. -our)": ("Group 1", "ends in -or vs. -our", "Orthographic/Spelling"),
    "Group 2 (-ize vs. -ise)": ("Group 2", "ends in -ize vs. -ise", "Orthographic/Spelling"),
    "Group 3 (-er vs. -re)": ("Group 3", "ends in -er vs. -re", "Orthographic/Spelling"),
    "Group 4 (-og vs. -ogue)": ("Group 4", "ends in -og vs. -ogue", "Orthographic/Spelling"),
    "Group 5 (single l vs. double l)": ("Group 5", "single l vs. double l", "Orthographic/Spelling"),
    "Group 6 (same length, small edit)": ("Group 6", "sublexical spelling variation", "Orthographic/Spelling"),
    "Group 7 (different words)": ("Group 7", "different lexical items entirely", "Vocabulary"),
    "Group 8 (miscellaneous)": ("Group 8", "other", "Uncategorized"),
    "Group 9 (-ense vs. -ence)": ("Group 9", "ends in -ense vs. -ence", "Orthographic/Spelling"),
    "Group 10 (-ae vs. -e)": ("Group 10", "e vs. ae", "Orthographic/Spelling"),
}


def categorize(us: str, uk: str) -> str:
    us_l = us.strip().lower()
    uk_l = uk.strip().lower()
    if re.search(r"or$", us_l) and re.search(r"our$", uk_l):
        return "Group 1 (-or vs. -our)"
    if re.search(r"ize$", us_l) and re.search(r"ise$", uk_l):
        return "Group 2 (-ize vs. -ise)"
    if re.search(r"er$", us_l) and re.search(r"re$", uk_l):
        return "Group 3 (-er vs. -re)"
    if re.search(r"og$", us_l) and re.search(r"ogue$", uk_l):
        return "Group 4 (-og vs. -ogue)"
    if re.search(r"ense$", us_l) and re.search(r"ence$", uk_l):
        return "Group 9 (-ense vs. -ence)"
    if "ae" in uk_l and uk_l.replace("ae", "e") == us_l:
        return "Group 10 (-ae vs. -e)"
    if "l" in us_l and "ll" in uk_l and len(uk_l) == len(us_l) + 1:
        return "Group 5 (single l vs. double l)"
    dist = lev_dist(us_l, uk_l)
    if len(us_l) == len(uk_l):
        if 1 <= dist <= 2:
            return "Group 6 (same length, small edit)"
        return "Group 7 (different words)"
    return "Group 7 (different words)"


def classify(us: str, uk: str) -> Tuple[str, str, str]:
    return GROUP_DEFINITIONS.get(categorize(us, uk), GROUP_DEFINITIONS["Group 8 (miscellaneous)"])

