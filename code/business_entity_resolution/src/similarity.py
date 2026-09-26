"""
String similarity features for entity resolution.

Computes comprehensive pairwise similarity features for:
- Business names
- Addresses
- General (country, source, missingness)
"""

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from normalization import get_tokens, jaccard_tokens, _compact


# ── Pre-compiled vectorizers (fit on demand) ────────────────────────

_name_char_vectorizer: Optional[TfidfVectorizer] = None
_name_token_vectorizer: Optional[TfidfVectorizer] = None
_addr_char_vectorizer: Optional[TfidfVectorizer] = None
_addr_token_vectorizer: Optional[TfidfVectorizer] = None


def _get_name_char_vectorizer() -> TfidfVectorizer:
    global _name_char_vectorizer
    if _name_char_vectorizer is None:
        _name_char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
        )
    return _name_char_vectorizer


def _get_name_token_vectorizer() -> TfidfVectorizer:
    global _name_token_vectorizer
    if _name_token_vectorizer is None:
        _name_token_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
    return _name_token_vectorizer


def _get_addr_char_vectorizer() -> TfidfVectorizer:
    global _addr_char_vectorizer
    if _addr_char_vectorizer is None:
        _addr_char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 6),
            min_df=1,
            sublinear_tf=True,
        )
    return _addr_char_vectorizer


def _get_addr_token_vectorizer() -> TfidfVectorizer:
    global _addr_token_vectorizer
    if _addr_token_vectorizer is None:
        _addr_token_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
    return _addr_token_vectorizer


# ── Core similarity functions ───────────────────────────────────────

def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (1 - distance/max_len)."""
    if not s1 or not s2:
        return 0.0
    dist = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - dist / max_len if max_len > 0 else 1.0


def jaccard_similarity(s1: str, s2: str) -> float:
    """Character-level Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1)
    set2 = set(s2)
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def dice_similarity(s1: str, s2: str) -> float:
    """Character-level Dice coefficient."""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1)
    set2 = set(s2)
    inter = len(set1 & set2)
    return 2 * inter / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0.0


def tfidf_cosine_similarity(text1: str, text2: str, vectorizer: TfidfVectorizer) -> float:
    """TF-IDF cosine similarity using provided vectorizer."""
    if not text1 or not text2:
        return 0.0
    try:
        # Fit on both texts
        vec = vectorizer.fit_transform([text1, text2])
        # Cosine similarity
        sim = (vec[0] @ vec[1].T).toarray()[0, 0]
        return float(sim)
    except Exception:
        return 0.0


# ── Business Name Similarities ──────────────────────────────────────

def name_exact_match(norm1: str, norm2: str) -> int:
    """Exact match on normalized names."""
    return int(norm1 == norm2 and norm1 != "")


def name_rapidfuzz_ratio(norm1: str, norm2: str) -> float:
    """RapidFuzz ratio (0-100 normalized to 0-1)."""
    if not norm1 or not norm2:
        return 0.0
    return fuzz.ratio(norm1, norm2) / 100.0


def name_wratio(norm1: str, norm2: str) -> float:
    """RapidFuzz WRatio (handles transpositions)."""
    if not norm1 or not norm2:
        return 0.0
    return fuzz.WRatio(norm1, norm2) / 100.0


def name_qratio(norm1: str, norm2: str) -> float:
    """RapidFuzz QRatio (fast)."""
    if not norm1 or not norm2:
        return 0.0
    return fuzz.QRatio(norm1, norm2) / 100.0


def name_token_sort_ratio(norm1: str, norm2: str) -> float:
    """Token sort ratio (order-independent)."""
    if not norm1 or not norm2:
        return 0.0
    return fuzz.token_sort_ratio(norm1, norm2) / 100.0


def name_token_set_ratio(norm1: str, norm2: str) -> float:
    """Token set ratio (handles extra tokens)."""
    if not norm1 or not norm2:
        return 0.0
    return fuzz.token_set_ratio(norm1, norm2) / 100.0


def name_partial_ratio(norm1: str, norm2: str) -> float:
    """Partial ratio (substring matching)."""
    if not norm1 or not norm2:
        return 0.0
    return fuzz.partial_ratio(norm1, norm2) / 100.0


def name_length_diff(norm1: str, norm2: str) -> int:
    """Absolute length difference."""
    return abs(len(norm1) - len(norm2))


def name_length_ratio(norm1: str, norm2: str) -> float:
    """Length ratio (min/max)."""
    l1, l2 = len(norm1), len(norm2)
    if max(l1, l2) == 0:
        return 1.0
    return min(l1, l2) / max(l1, l2)


def name_token_count_diff(norm1: str, norm2: str) -> int:
    """Token count difference."""
    t1 = len(get_tokens(norm1))
    t2 = len(get_tokens(norm2))
    return abs(t1 - t2)


def name_shared_token_count(norm1: str, norm2: str) -> int:
    """Number of shared tokens."""
    t1 = set(get_tokens(norm1))
    t2 = set(get_tokens(norm2))
    return len(t1 & t2)


def name_jaccard_tokens(norm1: str, norm2: str) -> float:
    """Jaccard on tokens."""
    return jaccard_tokens(get_tokens(norm1), get_tokens(norm2))


def name_tfidf_char_cosine(norm1: str, norm2: str) -> float:
    """Character TF-IDF cosine."""
    return tfidf_cosine_similarity(norm1, norm2, _get_name_char_vectorizer())


def name_tfidf_token_cosine(norm1: str, norm2: str) -> float:
    """Token TF-IDF cosine."""
    return tfidf_cosine_similarity(norm1, norm2, _get_name_token_vectorizer())


# ── Address Similarities ────────────────────────────────────────────

def addr_exact_match(norm1: str, norm2: str) -> int:
    """Exact match on normalized addresses."""
    return int(norm1 == norm2 and norm1 != "")


def addr_rapidfuzz_ratio(norm1: str, norm2: str) -> float:
    if not norm1 or not norm2:
        return 0.0
    return fuzz.ratio(norm1, norm2) / 100.0


def addr_wratio(norm1: str, norm2: str) -> float:
    if not norm1 or not norm2:
        return 0.0
    return fuzz.WRatio(norm1, norm2) / 100.0


def addr_qratio(norm1: str, norm2: str) -> float:
    if not norm1 or not norm2:
        return 0.0
    return fuzz.QRatio(norm1, norm2) / 100.0


def addr_token_sort_ratio(norm1: str, norm2: str) -> float:
    if not norm1 or not norm2:
        return 0.0
    return fuzz.token_sort_ratio(norm1, norm2) / 100.0


def addr_token_set_ratio(norm1: str, norm2: str) -> float:
    if not norm1 or not norm2:
        return 0.0
    return fuzz.token_set_ratio(norm1, norm2) / 100.0


def addr_partial_ratio(norm1: str, norm2: str) -> float:
    if not norm1 or not norm2:
        return 0.0
    return fuzz.partial_ratio(norm1, norm2) / 100.0


def addr_levenshtein(norm1: str, norm2: str) -> float:
    return levenshtein_similarity(norm1, norm2)


def addr_jaccard(norm1: str, norm2: str) -> float:
    return jaccard_similarity(norm1, norm2)


def addr_dice(norm1: str, norm2: str) -> float:
    return dice_similarity(norm1, norm2)


def addr_length_diff(norm1: str, norm2: str) -> int:
    return abs(len(norm1) - len(norm2))


def addr_length_ratio(norm1: str, norm2: str) -> float:
    l1, l2 = len(norm1), len(norm2)
    if max(l1, l2) == 0:
        return 1.0
    return min(l1, l2) / max(l1, l2)


def addr_token_count_diff(norm1: str, norm2: str) -> int:
    t1 = len(get_tokens(norm1))
    t2 = len(get_tokens(norm2))
    return abs(t1 - t2)


def addr_shared_token_count(norm1: str, norm2: str) -> int:
    t1 = set(get_tokens(norm1))
    t2 = set(get_tokens(norm2))
    return len(t1 & t2)


def addr_jaccard_tokens(norm1: str, norm2: str) -> float:
    return jaccard_tokens(get_tokens(norm1), get_tokens(norm2))


def addr_tfidf_char_cosine(norm1: str, norm2: str) -> float:
    return tfidf_cosine_similarity(norm1, norm2, _get_addr_char_vectorizer())


def addr_tfidf_token_cosine(norm1: str, norm2: str) -> float:
    return tfidf_cosine_similarity(norm1, norm2, _get_addr_token_vectorizer())


# ── Address-specific features ───────────────────────────────────────

DIGIT_RE = re.compile(r"\d+")


def extract_digits(text: str) -> List[str]:
    """Extract all digit sequences from text."""
    return DIGIT_RE.findall(text) if text else []


def postal_code_match(pc1: Optional[str], pc2: Optional[str]) -> int:
    """Postal code exact match."""
    if not pc1 or not pc2:
        return 0
    return int(pc1 == pc2)


def house_number_match(hn1: Optional[str], hn2: Optional[str]) -> int:
    """House number exact match."""
    if not hn1 or not hn2:
        return 0
    return int(hn1 == hn2)


def digit_overlap(text1: str, text2: str) -> int:
    """Count of shared digit sequences."""
    d1 = set(extract_digits(text1))
    d2 = set(extract_digits(text2))
    return len(d1 & d2)


def digit_jaccard(text1: str, text2: str) -> float:
    """Jaccard on digit sequences."""
    d1 = set(extract_digits(text1))
    d2 = set(extract_digits(text2))
    if not d1 and not d2:
        return 1.0
    if not d1 or not d2:
        return 0.0
    return len(d1 & d2) / len(d1 | d2)


def shared_numeric_tokens(norm1: str, norm2: str) -> int:
    """Count of shared tokens that are purely numeric."""
    t1 = {t for t in get_tokens(norm1) if t.isdigit()}
    t2 = {t for t in get_tokens(norm2) if t.isdigit()}
    return len(t1 & t2)


def address_token_overlap(norm1: str, norm2: str) -> int:
    """Total shared tokens in address."""
    return len(set(get_tokens(norm1)) & set(get_tokens(norm2)))


# ── General Features ────────────────────────────────────────────────

def country_exact_match(c1: str, c2: str) -> int:
    """Exact match on country (original)."""
    return int(c1 == c2 and c1 != "")


def country_norm_match(norm1: str, norm2: str) -> int:
    """Exact match on normalized country."""
    return int(norm1 == norm2 and norm1 != "")


def name_missing(norm: str) -> int:
    """Indicator: name is empty."""
    return int(not norm or norm == "")


def addr_missing(norm: str) -> int:
    """Indicator: address is empty."""
    return int(not norm or norm == "")


def source_pair_indicator(source: str) -> int:
    """S1-S2 = 0, S1-S3 = 1."""
    return 1 if source == "S3" else 0


# ── Batch feature computation ───────────────────────────────────────

def compute_name_features(
    df: pd.DataFrame,
    s1_col: str = "business_name_norm",
    cand_col: str = "business_name_norm",
) -> pd.DataFrame:
    """Compute all name similarity features."""
    out = pd.DataFrame(index=df.index)

    out["name_exact"] = df.apply(lambda r: name_exact_match(r[s1_col], r[cand_col]), axis=1)
    out["name_ratio"] = df.apply(lambda r: name_rapidfuzz_ratio(r[s1_col], r[cand_col]), axis=1)
    out["name_wratio"] = df.apply(lambda r: name_wratio(r[s1_col], r[cand_col]), axis=1)
    out["name_qratio"] = df.apply(lambda r: name_qratio(r[s1_col], r[cand_col]), axis=1)
    out["name_token_sort"] = df.apply(lambda r: name_token_sort_ratio(r[s1_col], r[cand_col]), axis=1)
    out["name_token_set"] = df.apply(lambda r: name_token_set_ratio(r[s1_col], r[cand_col]), axis=1)
    out["name_partial"] = df.apply(lambda r: name_partial_ratio(r[s1_col], r[cand_col]), axis=1)
    out["name_levenshtein"] = df.apply(lambda r: levenshtein_similarity(r[s1_col], r[cand_col]), axis=1)
    out["name_jaccard_char"] = df.apply(lambda r: jaccard_similarity(r[s1_col], r[cand_col]), axis=1)
    out["name_dice"] = df.apply(lambda r: dice_similarity(r[s1_col], r[cand_col]), axis=1)
    out["name_len_diff"] = df.apply(lambda r: name_length_diff(r[s1_col], r[cand_col]), axis=1)
    out["name_len_ratio"] = df.apply(lambda r: name_length_ratio(r[s1_col], r[cand_col]), axis=1)
    out["name_tok_diff"] = df.apply(lambda r: name_token_count_diff(r[s1_col], r[cand_col]), axis=1)
    out["name_shared_toks"] = df.apply(lambda r: name_shared_token_count(r[s1_col], r[cand_col]), axis=1)
    out["name_jaccard_tok"] = df.apply(lambda r: name_jaccard_tokens(r[s1_col], r[cand_col]), axis=1)
    out["name_tfidf_char"] = df.apply(lambda r: name_tfidf_char_cosine(r[s1_col], r[cand_col]), axis=1)
    out["name_tfidf_tok"] = df.apply(lambda r: name_tfidf_token_cosine(r[s1_col], r[cand_col]), axis=1)

    return out


def compute_address_features(
    df: pd.DataFrame,
    s1_col: str = "business_address_norm",
    cand_col: str = "business_address_norm",
    s1_pc_col: str = "business_address_postal_code",
    cand_pc_col: str = "business_address_postal_code",
    s1_hn_col: str = "business_address_house_number",
    cand_hn_col: str = "business_address_house_number",
) -> pd.DataFrame:
    """Compute all address similarity features."""
    out = pd.DataFrame(index=df.index)

    out["addr_exact"] = df.apply(lambda r: addr_exact_match(r[s1_col], r[cand_col]), axis=1)
    out["addr_ratio"] = df.apply(lambda r: addr_rapidfuzz_ratio(r[s1_col], r[cand_col]), axis=1)
    out["addr_wratio"] = df.apply(lambda r: addr_wratio(r[s1_col], r[cand_col]), axis=1)
    out["addr_qratio"] = df.apply(lambda r: addr_qratio(r[s1_col], r[cand_col]), axis=1)
    out["addr_token_sort"] = df.apply(lambda r: addr_token_sort_ratio(r[s1_col], r[cand_col]), axis=1)
    out["addr_token_set"] = df.apply(lambda r: addr_token_set_ratio(r[s1_col], r[cand_col]), axis=1)
    out["addr_partial"] = df.apply(lambda r: addr_partial_ratio(r[s1_col], r[cand_col]), axis=1)
    out["addr_levenshtein"] = df.apply(lambda r: addr_levenshtein(r[s1_col], r[cand_col]), axis=1)
    out["addr_jaccard_char"] = df.apply(lambda r: addr_jaccard(r[s1_col], r[cand_col]), axis=1)
    out["addr_dice"] = df.apply(lambda r: addr_dice(r[s1_col], r[cand_col]), axis=1)
    out["addr_len_diff"] = df.apply(lambda r: addr_length_diff(r[s1_col], r[cand_col]), axis=1)
    out["addr_len_ratio"] = df.apply(lambda r: addr_length_ratio(r[s1_col], r[cand_col]), axis=1)
    out["addr_tok_diff"] = df.apply(lambda r: addr_token_count_diff(r[s1_col], r[cand_col]), axis=1)
    out["addr_shared_toks"] = df.apply(lambda r: addr_shared_token_count(r[s1_col], r[cand_col]), axis=1)
    out["addr_jaccard_tok"] = df.apply(lambda r: addr_jaccard_tokens(r[s1_col], r[cand_col]), axis=1)
    out["addr_tfidf_char"] = df.apply(lambda r: addr_tfidf_char_cosine(r[s1_col], r[cand_col]), axis=1)
    out["addr_tfidf_tok"] = df.apply(lambda r: addr_tfidf_token_cosine(r[s1_col], r[cand_col]), axis=1)

    # Address-specific
    out["postal_match"] = df.apply(lambda r: postal_code_match(r[s1_pc_col], r[cand_pc_col]), axis=1)
    out["house_num_match"] = df.apply(lambda r: house_number_match(r[s1_hn_col], r[cand_hn_col]), axis=1)
    out["digit_overlap"] = df.apply(lambda r: digit_overlap(r[s1_col], r[cand_col]), axis=1)
    out["digit_jaccard"] = df.apply(lambda r: digit_jaccard(r[s1_col], r[cand_col]), axis=1)
    out["shared_numeric_toks"] = df.apply(lambda r: shared_numeric_tokens(r[s1_col], r[cand_col]), axis=1)
    out["addr_tok_overlap"] = df.apply(lambda r: address_token_overlap(r[s1_col], r[cand_col]), axis=1)

    return out


def compute_general_features(
    df: pd.DataFrame,
    s1_country_col: str = "country_norm",
    cand_country_col: str = "country_norm",
    s1_name_col: str = "business_name_norm",
    cand_name_col: str = "business_name_norm",
    s1_addr_col: str = "business_address_norm",
    cand_addr_col: str = "business_address_norm",
    source_col: str = "candidate_source",
) -> pd.DataFrame:
    """Compute general features."""
    out = pd.DataFrame(index=df.index)

    out["country_exact"] = df.apply(lambda r: country_exact_match(r[s1_country_col], r[cand_country_col]), axis=1)
    out["country_norm_match"] = df.apply(lambda r: country_norm_match(r[s1_country_col], r[cand_country_col]), axis=1)
    out["name_missing"] = df[s1_name_col].apply(name_missing)
    out["cand_name_missing"] = df[cand_name_col].apply(name_missing)
    out["addr_missing"] = df[s1_addr_col].apply(addr_missing)
    out["cand_addr_missing"] = df[cand_addr_col].apply(addr_missing)
    out["source_s2_s3"] = df[source_col].apply(source_pair_indicator)

    return out


def compute_all_features(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all pairwise features for candidate pairs.

    Args:
        candidates_df: DataFrame with s1_id, candidate_id, candidate_source
        s1_df, s2_df, s3_df: Normalized source DataFrames (must have _norm columns)

    Returns:
        DataFrame with all features + identifying columns
    """
    # Merge candidate pairs with source data
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)

    # Merge S1 data
    merged = candidates_df.merge(
        s1_df.add_suffix("_s1"),
        left_on="s1_id",
        right_on="entity_id_s1",
        how="left",
    )

    # Merge candidate data
    merged = merged.merge(
        s23_df.add_suffix("_cand"),
        left_on="candidate_id",
        right_on="entity_id_cand",
        how="left",
    )

    # Compute features
    name_feats = compute_name_features(
        merged,
        s1_col="business_name_norm_s1",
        cand_col="business_name_norm_cand",
    )
    addr_feats = compute_address_features(
        merged,
        s1_col="business_address_norm_s1",
        cand_col="business_address_norm_cand",
        s1_pc_col="business_address_postal_code_s1",
        cand_pc_col="business_address_postal_code_cand",
        s1_hn_col="business_address_house_number_s1",
        cand_hn_col="business_address_house_number_cand",
    )
    gen_feats = compute_general_features(
        merged,
        s1_country_col="country_norm_s1",
        cand_country_col="country_norm_cand",
        s1_name_col="business_name_norm_s1",
        cand_name_col="business_name_norm_cand",
        s1_addr_col="business_address_norm_s1",
        cand_addr_col="business_address_norm_cand",
        source_col="candidate_source",
    )

    # Combine
    feature_df = pd.concat([name_feats, addr_feats, gen_feats], axis=1)

    # Add identifying columns
    feature_df["s1_id"] = candidates_df["s1_id"].values
    feature_df["candidate_id"] = candidates_df["candidate_id"].values
    feature_df["candidate_source"] = candidates_df["candidate_source"].values
    if "label" in candidates_df.columns:
        feature_df["label"] = candidates_df["label"].values
    if "blocking_methods" in candidates_df.columns:
        feature_df["blocking_methods"] = candidates_df["blocking_methods"].values

    return feature_df


if __name__ == "__main__":
    # Quick test
    import pandas as pd
    df = pd.DataFrame({
        "s1_name": ["abc corp", "google llc", ""],
        "cand_name": ["abc corporation", "google inc", "test"],
        "s1_addr": ["123 main st", "1600 amphitheatre pkwy", ""],
        "cand_addr": ["123 main street", "1600 amphitheatre parkway", "456 oak ave"],
        "s1_country": ["us", "us", ""],
        "cand_country": ["us", "us", ""],
        "candidate_source": ["S2", "S2", "S3"],
    })

    name_f = compute_name_features(df.rename(columns={"s1_name": "business_name_norm", "cand_name": "business_name_norm"}))
    addr_f = compute_address_features(df.rename(columns={"s1_addr": "business_address_norm", "cand_addr": "business_address_norm",
                                                         "s1_country": "country_norm", "cand_country": "country_norm"}))
    gen_f = compute_general_features(df.rename(columns={"s1_country": "country_norm", "cand_country": "country_norm",
                                                         "s1_name": "business_name_norm", "cand_name": "business_name_norm",
                                                         "s1_addr": "business_address_norm", "cand_addr": "business_address_norm"}))

    print("Name features:")
    print(name_f)
    print("\nAddress features:")
    print(addr_f)
    print("\nGeneral features:")
    print(gen_f)