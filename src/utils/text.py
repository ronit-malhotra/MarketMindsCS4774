"""
Utility functions for text cleaning used across notebooks.

We keep this in src/ so the pipeline stays consistent and reproducible.
"""

import re
from typing import Optional


# A small list of boilerplate patterns commonly seen in headlines.
# Keep conservative: remove only obvious clutter to avoid deleting signal words.
_BOILERPLATE_PREFIXES = [
    r"^breaking:\s*",
    r"^update\s*\d*:\s*",
    r"^exclusive:\s*",
    r"^reuters:\s*",
    r"^wsj:\s*",
    r"^cnbc:\s*",
    r"^bloomberg:\s*",
]


def clean_headline(text: Optional[str]) -> str:
    """
    Clean a raw news headline into a normalized string suitable for NLP features.

    Steps:
      1) Handle None values safely
      2) Lowercase
      3) Remove boilerplate prefixes (conservative)
      4) Remove URLs
      5) Remove non-alphanumeric characters (keep spaces)
      6) Normalize whitespace

    Args:
        text: Raw headline string.

    Returns:
        Cleaned headline string (possibly empty).
    """
    if text is None:
        return ""

    # Normalize and lowercase first
    s = str(text).strip().lower()

    # Remove boilerplate prefixes
    for pat in _BOILERPLATE_PREFIXES:
        s = re.sub(pat, "", s)

    # Remove URLs
    s = re.sub(r"http\S+|www\.\S+", "", s)

    # Keep letters, numbers, and spaces. Remove everything else.
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    # Collapse repeated whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s
