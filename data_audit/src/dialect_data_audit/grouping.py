"""Rule-based grouping for AmE/BrE variant pairs."""

from __future__ import annotations

import re
from typing import Tuple


def lev_dist(a: str, b: str) -> int:
    """Compute Levenshtein edit distance."""

    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


GROUP_DEFINITIONS = {
    "Group 1 (-or vs. -our)": {
        "group": "Group 1",
        "type": "ends in -or vs. -our",
        "category": "Orthographic/Spelling",
    },
    "Group 2 (-ize vs. -ise)": {
        "group": "Group 2",
        "type": "ends in -ize vs. -ise",
        "category": "Orthographic/Spelling",
    },
    "Group 3 (-er vs. -re)": {
        "group": "Group 3",
        "type": "ends in -er vs. -re",
        "category": "Orthographic/Spelling",
    },
    "Group 4 (-og vs. -ogue)": {
        "group": "Group 4",
        "type": "ends in -og vs. -ogue",
        "category": "Orthographic/Spelling",
    },
    "Group 5 (single l vs. double l)": {
        "group": "Group 5",
        "type": "single l vs. double l",
        "category": "Orthographic/Spelling",
    },
    "Group 6 (same length, small edit)": {
        "group": "Group 6",
        "type": "sublexical spelling variation",
        "category": "Orthographic/Spelling",
    },
    "Group 7 (different words)": {
        "group": "Group 7",
        "type": "different lexical items entirely",
        "category": "Vocabulary",
    },
    "Group 8 (miscellaneous)": {
        "group": "Group 8",
        "type": "other",
        "category": "Uncategorized",
    },
    "Group 9 (-ense vs. -ence)": {
        "group": "Group 9",
        "type": "ends in -ense vs. -ence",
        "category": "Orthographic/Spelling",
    },
    "Group 10 (-ae vs. -e)": {
        "group": "Group 10",
        "type": "e vs. ae",
        "category": "Orthographic/Spelling",
    },
}


def categorize(us: str, uk: str) -> str:
    """Assign a pair to the same rule families used in the paper analyses."""

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
    """Return group, type, and category labels for a pair."""

    key = categorize(us, uk)
    meta = GROUP_DEFINITIONS.get(key, GROUP_DEFINITIONS["Group 8 (miscellaneous)"])
    return meta["group"], meta["type"], meta["category"]

