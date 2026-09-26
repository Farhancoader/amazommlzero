"""
Normalization module for entity resolution.

Provides comprehensive normalization for:
- Business names
- Addresses
- Countries

Preserves multiple representations: original, normalized, tokenized, compact.
No external data dependencies for abbreviation resolution.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


# ── Legal suffix patterns (US/India common) ─────────────────────────
# Multi-word phrases must come before single words
LEGAL_SUFFIX_PHRASES = [
    "limited liability company",
    "limited liability partnership",
    "private limited",
    "pvt limited",
    "ltd co",
    "inc co",
]

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "llp", "ltd", "limited", "corp", "corporation",
    "co", "company", "pllc", "plc", "lp", "l.l.c.", "l.l.c", "l.l.p.", "l.l.p",
    "p.c.", "p.c", "pc", "pa", "p.a.", "lllp", "gp",
}

# Sort by length descending to match longer first
ALL_SUFFIXES = sorted(LEGAL_SUFFIX_PHRASES + list(LEGAL_SUFFIXES), key=len, reverse=True)
LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in ALL_SUFFIXES) + r")\b\.?",
    re.IGNORECASE
)

# ── Address abbreviation map (no external data) ─────────────────────
ADDRESS_ABBREVIATIONS = {
    # Street types
    "street": "st", "avenue": "ave", "road": "rd", "drive": "dr",
    "lane": "ln", "court": "ct", "circle": "cir", "boulevard": "blvd",
    "parkway": "pkwy", "highway": "hwy", "expressway": "expy",
    "place": "pl", "terrace": "ter", "way": "way", "trail": "trl",
    "square": "sq", "loop": "loop", "crescent": "cres",
    # Directional
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    # Building/unit
    "apartment": "apt", "suite": "ste", "unit": "unit", "floor": "fl",
    "building": "bldg", "department": "dept", "room": "rm",
    # Common
    "highway": "hwy", "freeway": "fwy", "turnpike": "tpke",
    "junction": "jct", "crossing": "xing", "center": "ctr",
    "park": "pk", "plaza": "plz", "mall": "mall",
    # India-specific
    "nagar": "nagar", "colony": "colony", "vihar": "vihar",
    "puram": "puram", "pura": "pura", "enclave": "enclave",
    "extension": "extn", "phase": "phase", "block": "blk",
    "sector": "sec", "zone": "zone",
}

# Compile abbreviation regex (word boundaries)
ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ADDRESS_ABBREVIATIONS.keys(), key=len, reverse=True)) + r")\b\.?",
    re.IGNORECASE
)

# ── Regex patterns ──────────────────────────────────────────────────
MULTI_WHITESPACE = re.compile(r"\s+")
PUNCTUATION = re.compile(r"[^\w\s&-]")  # Keep & and - for business names
COMPACT_PUNCT = re.compile(r"[^\w]")     # Remove all non-alphanumeric for compact
DIGIT_SEQUENCE = re.compile(r"\d+")
POSTAL_CODE_US = re.compile(r"\b\d{5}(?:-\d{4})?\b")
POSTAL_CODE_INDIA = re.compile(r"\b\d{6}\b")
POSTAL_CODE_FRANCE = re.compile(r"\b\d{5}\b")
HOUSE_NUMBER = re.compile(r"^\s*(\d+[A-Za-z]?)\b")


@dataclass
class NormalizedField:
    """Container for all representations of a normalized field."""
    original: str
    normalized: str
    tokenized: List[str]
    compact: str


# ── Core normalization utilities ────────────────────────────────────

def _unicode_normalize(text: str) -> str:
    """NFD normalization + strip combining characters + strip."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", str(text))
    # Remove combining characters (accents, etc.)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.strip()


def _case_fold(text: str) -> str:
    """Aggressive case folding for matching."""
    return text.casefold()


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace to single space."""
    return MULTI_WHITESPACE.sub(" ", text).strip()


def _normalize_punctuation(text: str, keep_ampersand: bool = True) -> str:
    """Normalize punctuation. Optionally keep & and -."""
    if keep_ampersand:
        # Replace various dash/hyphen types with standard hyphen
        text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
        # Keep & and -
        text = PUNCTUATION.sub(" ", text)
    else:
        text = COMPACT_PUNCT.sub(" ", text)
    return _collapse_whitespace(text)


def _tokenize(text: str) -> List[str]:
    """Split into tokens (alphanumeric sequences)."""
    if not text:
        return []
    return [t for t in text.split() if t]


def _compact(text: str) -> str:
    """Remove all non-alphanumeric, lowercase."""
    if not text:
        return ""
    return COMPACT_PUNCT.sub("", text.casefold())


# ── Business Name Normalization ────────────────────────────────────

def _normalize_and_safe(text: str) -> str:
    """Safe & <-> and normalization. Only when standalone."""
    # Replace & with 'and' when surrounded by spaces or at boundaries
    text = re.sub(r"\s+&\s+", " and ", text)
    text = re.sub(r"^&\s+", "and ", text)
    text = re.sub(r"\s+&$", " and", text)
    text = re.sub(r"^&$", "and", text)
    return text


def _normalize_legal_suffixes(text: str) -> str:
    """Normalize legal entity suffixes to canonical form."""
    def repl(match):
        suffix = match.group(0).lower().rstrip(".")
        # Map to shortest canonical form
        canonical = {
            "incorporated": "inc",
            "inc.": "inc",
            "corporation": "corp",
            "corp.": "corp",
            "company": "co",
            "co.": "co",
            "limited": "ltd",
            "ltd.": "ltd",
            "private limited": "pvt ltd",
            "pvt limited": "pvt ltd",
            "limited liability company": "llc",
            "limited liability partnership": "llp",
        }
        return canonical.get(suffix, suffix)
    return LEGAL_SUFFIX_PATTERN.sub(repl, text)


def _normalize_digits(text: str) -> str:
    """Normalize digit sequences to a token."""
    # Replace digit sequences with a placeholder to preserve structure
    return DIGIT_SEQUENCE.sub(" num ", text)


def normalize_business_name(name: str) -> NormalizedField:
    """
    Full business name normalization pipeline.

    Steps:
    1. Unicode normalization (NFKC)
    2. Case folding
    3. Whitespace normalization
    4. Punctuation normalization (keep & and -)
    5. Safe & <-> and normalization
    6. Legal suffix normalization
    7. Token normalization
    8. Digit normalization
    """
    if not name or pd_isna(name):
        return NormalizedField("", "", [], "")

    original = str(name)

    # Step 1: Unicode
    text = _unicode_normalize(original)

    # Step 2: Case fold
    text = _case_fold(text)

    # Step 3: Whitespace
    text = _collapse_whitespace(text)

    # Step 4: Punctuation (keep & -)
    text = _normalize_punctuation(text, keep_ampersand=True)

    # Step 5: Safe & <-> and
    text = _normalize_and_safe(text)

    # Step 6: Legal suffixes
    text = _normalize_legal_suffixes(text)

    # Step 7: Token normalization (already done via whitespace collapse)

    # Step 8: Digit normalization
    text = _normalize_digits(text)

    normalized = _collapse_whitespace(text)
    tokenized = _tokenize(normalized)
    compact = _compact(normalized)

    return NormalizedField(original, normalized, tokenized, compact)


# ── Address Normalization ────────────────────────────────────────────

def _normalize_address_abbreviations(text: str) -> str:
    """Expand/normalize common address abbreviations."""
    def repl(match):
        word = match.group(0).lower().rstrip(".")
        return ADDRESS_ABBREVIATIONS.get(word, word)
    return ABBREV_PATTERN.sub(repl, text)


def _extract_postal_code(text: str, country: str = "") -> Optional[str]:
    """Extract postal code based on country pattern."""
    if not text:
        return None
    c = country.upper() if country else ""
    if c in ("US", "USA", "UNITED STATES"):
        m = POSTAL_CODE_US.search(text)
    elif c in ("IN", "INDIA"):
        m = POSTAL_CODE_INDIA.search(text)
    elif c in ("FR", "FRA", "FRANCE"):
        m = POSTAL_CODE_FRANCE.search(text)
    else:
        # Try all patterns
        m = POSTAL_CODE_US.search(text) or POSTAL_CODE_INDIA.search(text) or POSTAL_CODE_FRANCE.search(text)
    return m.group(0) if m else None


def _extract_house_number(text: str) -> Optional[str]:
    """Extract leading house/building number."""
    if not text:
        return None
    m = HOUSE_NUMBER.match(text)
    return m.group(1) if m else None


def _normalize_digits_address(text: str) -> str:
    """Normalize digits in address context."""
    # Keep numbers but separate them for token matching
    return DIGIT_SEQUENCE.sub(" num ", text)


def normalize_address(address: str, country: str = "") -> NormalizedField:
    """
    Full address normalization pipeline.

    Steps:
    1. Unicode normalization (NFKC)
    2. Case folding
    3. Whitespace normalization
    4. Punctuation normalization
    5. Address abbreviation normalization
    6. Token normalization
    7. Digit normalization
    """
    if not address or pd_isna(address):
        return NormalizedField("", "", [], "")

    original = str(address)

    # Step 1: Unicode
    text = _unicode_normalize(original)

    # Step 2: Case fold
    text = _case_fold(text)

    # Step 3: Whitespace
    text = _collapse_whitespace(text)

    # Step 4: Punctuation (remove all for address)
    text = _normalize_punctuation(text, keep_ampersand=False)

    # Step 5: Address abbreviations
    text = _normalize_address_abbreviations(text)

    # Step 6: Token normalization (whitespace)
    text = _collapse_whitespace(text)

    # Step 7: Digit normalization
    text = _normalize_digits_address(text)

    normalized = _collapse_whitespace(text)
    tokenized = _tokenize(normalized)
    compact = _compact(normalized)

    return NormalizedField(original, normalized, tokenized, compact)


# ── Country Normalization ────────────────────────────────────────────

def normalize_country(country: str) -> NormalizedField:
    """
    Country normalization - minimal, preserves arbitrary string labels.
    Steps:
    1. Unicode normalization
    2. Whitespace normalization
    3. Case normalization (title case for display, lower for matching)
    """
    if not country or pd_isna(country):
        return NormalizedField("", "", [], "")

    original = str(country)

    # Step 1: Unicode
    text = _unicode_normalize(original)

    # Step 2: Whitespace
    text = _collapse_whitespace(text)

    # Step 3: Case - use title case for display consistency
    text = text.title()

    normalized = text
    tokenized = _tokenize(normalized)
    compact = _compact(normalized).upper()

    return NormalizedField(original, normalized, tokenized, compact)


# ── Batch normalization for DataFrames ──────────────────────────────

try:
    import pandas as pd
    pd_isna = pd.isna
except ImportError:
    def pd_isna(x):
        return x is None or (isinstance(x, float) and x != x)


def add_normalized_columns(df, name_col: str = "business_name",
                           addr_col: str = "business_address",
                           country_col: str = "country") -> None:
    """
    Add normalized columns to DataFrame in-place.

    Creates:
    - {col}_norm: normalized string
    - {col}_tokens: token list (as space-joined string for storage)
    - {col}_compact: compact form
    - For address: _postal_code, _house_number
    """
    # Business name
    if name_col in df.columns:
        results = df[name_col].apply(normalize_business_name)
        df[f"{name_col}_norm"] = results.apply(lambda x: x.normalized)
        df[f"{name_col}_tokens"] = results.apply(lambda x: " ".join(x.tokenized))
        df[f"{name_col}_compact"] = results.apply(lambda x: x.compact)

    # Address
    if addr_col in df.columns and country_col in df.columns:
        results = df.apply(lambda row: normalize_address(row[addr_col], row[country_col]), axis=1)
        df[f"{addr_col}_norm"] = results.apply(lambda x: x.normalized)
        df[f"{addr_col}_tokens"] = results.apply(lambda x: " ".join(x.tokenized))
        df[f"{addr_col}_compact"] = results.apply(lambda x: x.compact)
        df[f"{addr_col}_postal_code"] = df.apply(
            lambda row: _extract_postal_code(row[addr_col], row[country_col]), axis=1
        )
        df[f"{addr_col}_house_number"] = df[addr_col].apply(_extract_house_number)
    elif addr_col in df.columns:
        results = df[addr_col].apply(lambda x: normalize_address(x, ""))
        df[f"{addr_col}_norm"] = results.apply(lambda x: x.normalized)
        df[f"{addr_col}_tokens"] = results.apply(lambda x: " ".join(x.tokenized))
        df[f"{addr_col}_compact"] = results.apply(lambda x: x.compact)
        df[f"{addr_col}_postal_code"] = df[addr_col].apply(_extract_postal_code)
        df[f"{addr_col}_house_number"] = df[addr_col].apply(_extract_house_number)

    # Country
    if country_col in df.columns:
        results = df[country_col].apply(normalize_country)
        df[f"{country_col}_norm"] = results.apply(lambda x: x.normalized)
        df[f"{country_col}_tokens"] = results.apply(lambda x: " ".join(x.tokenized))
        df[f"{country_col}_compact"] = results.apply(lambda x: x.compact)


# ── Token utilities for blocking ────────────────────────────────────

def get_tokens(text: str) -> List[str]:
    """Get normalized tokens from already-normalized text."""
    return _tokenize(text) if text else []


def jaccard_tokens(tokens1: List[str], tokens2: List[str]) -> float:
    """Jaccard similarity on token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    set1, set2 = set(tokens1), set(tokens2)
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


if __name__ == "__main__":
    # Quick smoke test
    test_names = [
        "ABC & Co. Inc.",
        "XYZ Corporation",
        "Smith & Associates LLC",
        "Pvt Ltd Company",
        "Bajaj Auto Ltd.",
        "Google LLC",
        "A & B Trading Co",
    ]

    print("=== Business Name Normalization ===")
    for name in test_names:
        nf = normalize_business_name(name)
        print(f"  Original:  {nf.original}")
        print(f"  Normalized: {nf.normalized}")
        print(f"  Tokens:    {nf.tokenized}")
        print(f"  Compact:   {nf.compact}")
        print()

    test_addrs = [
        "123 Main Street, Apt 4B, New York, NY 10001",
        "456 Oak Avenue Suite 200, Bangalore 560001",
        "789 Elm Road, Paris 75001, France",
        "Plot 12, Sector 15, Gurgaon 122001",
    ]

    print("=== Address Normalization ===")
    for addr in test_addrs:
        nf = normalize_address(addr)
        postal = _extract_postal_code(addr)
        house = _extract_house_number(addr)
        print(f"  Original:     {nf.original}")
        print(f"  Normalized:   {nf.normalized}")
        print(f"  Tokens:       {nf.tokenized}")
        print(f"  Compact:      {nf.compact}")
        print(f"  Postal code:  {postal}")
        print(f"  House number: {house}")
        print()

    test_countries = ["US", "india", "  france  ", "United States", "IN"]
    print("=== Country Normalization ===")
    for c in test_countries:
        nf = normalize_country(c)
        print(f"  '{c}' -> '{nf.normalized}' (compact: {nf.compact})")