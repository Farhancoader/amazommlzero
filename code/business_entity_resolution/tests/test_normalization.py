"""
Unit tests for normalization module.
"""

import pytest
from src.normalization import (
    normalize_business_name,
    normalize_address,
    normalize_country,
    NormalizedField,
    _normalize_and_safe,
    _normalize_legal_suffixes,
    _extract_postal_code,
    _extract_house_number,
    jaccard_tokens,
)


class TestBusinessNameNormalization:
    """Tests for business name normalization."""

    def test_basic_normalization(self):
        nf = normalize_business_name("ABC Corporation")
        assert nf.normalized == "abc corp"
        assert nf.tokenized == ["abc", "corp"]
        assert nf.compact == "abccorp"

    def test_unicode_normalization(self):
        nf = normalize_business_name("Café & Co.")
        assert "cafe" in nf.normalized

    def test_case_folding(self):
        nf = normalize_business_name("GOOGLE LLC")
        assert nf.normalized == "google llc"

    def test_whitespace_normalization(self):
        nf = normalize_business_name("  ABC    Corp  ")
        assert nf.normalized == "abc corp"

    def test_punctuation_normalization(self):
        nf = normalize_business_name("A.B.C. Corp!")
        assert nf.normalized == "a b c corp"

    def test_ampersand_normalization(self):
        nf = normalize_business_name("Smith & Associates")
        assert "and" in nf.normalized
        assert "&" not in nf.normalized

    def test_ampersand_edge_cases(self):
        # Standalone &
        nf = normalize_business_name("A & B")
        assert nf.normalized == "a and b"

        # & at start/end - Company gets normalized to co as legal suffix
        nf = normalize_business_name("& Company")
        assert nf.normalized == "and co"

        nf = normalize_business_name("Company &")
        assert nf.normalized == "co and"

    def test_legal_suffixes(self):
        test_cases = [
            ("ABC Incorporated", "abc inc"),
            ("XYZ Corporation", "xyz corp"),
            ("Test Company Ltd", "test co ltd"),
            ("PQR Limited Liability Company", "pqr llc"),
            ("LMN Limited Liability Partnership", "lmn llp"),
            ("Private Limited Company", "pvt ltd co"),
            ("Bajaj Auto Ltd.", "bajaj auto ltd"),
        ]
        for original, expected in test_cases:
            nf = normalize_business_name(original)
            assert nf.normalized == expected, f"Failed for {original}: got {nf.normalized}"

    def test_digit_normalization(self):
        nf = normalize_business_name("Office 365 Solutions")
        assert "num" in nf.normalized.lower()

        nf = normalize_business_name("24/7 Services")
        assert "num" in nf.normalized.lower()

    def test_empty_handling(self):
        nf = normalize_business_name("")
        assert nf.normalized == ""
        assert nf.tokenized == []
        assert nf.compact == ""

        nf = normalize_business_name(None)
        assert nf.normalized == ""

    def test_preserves_original(self):
        original = "ABC Corp."
        nf = normalize_business_name(original)
        assert nf.original == original


class TestAddressNormalization:
    """Tests for address normalization."""

    def test_basic_address(self):
        nf = normalize_address("123 Main Street")
        assert "st" in nf.normalized
        assert "main" in nf.normalized

    def test_abbreviations(self):
        test_cases = [
            ("123 Main Street", "st"),
            ("456 Oak Avenue", "ave"),
            ("789 Elm Road", "rd"),
            ("321 Pine Drive", "dr"),
            ("555 Maple Lane", "ln"),
            ("777 Cedar Court", "ct"),
        ]
        for addr, expected_abbrev in test_cases:
            nf = normalize_address(addr)
            assert expected_abbrev in nf.normalized, f"Failed for {addr}: got {nf.normalized}"

    def test_directional_abbreviations(self):
        nf = normalize_address("123 North Main Street")
        assert "n" in nf.normalized

        nf = normalize_address("456 Southwest Oak Ave")
        assert "sw" in nf.normalized

    def test_unit_abbreviations(self):
        nf = normalize_address("123 Main St Apartment 4B")
        assert "apt" in nf.normalized

        nf = normalize_address("456 Oak Ave Suite 200")
        assert "ste" in nf.normalized

    def test_india_addresses(self):
        nf = normalize_address("Plot 12, Sector 15, Gurgaon 122001", "India")
        assert "sec" in nf.normalized
        assert "num" in nf.normalized  # digit normalization

    def test_postal_code_extraction(self):
        # US
        postal = _extract_postal_code("123 Main St, New York, NY 10001", "US")
        assert postal == "10001"

        postal = _extract_postal_code("123 Main St, NY 10001-1234", "US")
        assert postal == "10001-1234"

        # India
        postal = _extract_postal_code("Bangalore 560001", "India")
        assert postal == "560001"

        # France
        postal = _extract_postal_code("Paris 75001", "France")
        assert postal == "75001"

    def test_house_number_extraction(self):
        hn = _extract_house_number("123 Main Street")
        assert hn == "123"

        hn = _extract_house_number("456A Oak Avenue")
        assert hn == "456A"

        hn = _extract_house_number("Main Street")  # No number
        assert hn is None

    def test_digit_normalization(self):
        nf = normalize_address("123 Main St")
        assert "num" in nf.normalized

    def test_empty_handling(self):
        nf = normalize_address("")
        assert nf.normalized == ""

        nf = normalize_address(None)
        assert nf.normalized == ""


class TestCountryNormalization:
    """Tests for country normalization."""

    def test_basic_normalization(self):
        nf = normalize_country("US")
        assert nf.normalized == "Us"
        assert nf.compact == "US"

    def test_case_normalization(self):
        nf = normalize_country("india")
        assert nf.normalized == "India"

        nf = normalize_country("FRANCE")
        assert nf.normalized == "France"

    def test_whitespace_trimming(self):
        nf = normalize_country("  france  ")
        assert nf.normalized == "France"

    def test_arbitrary_strings(self):
        # Should not assume fixed vocabulary
        nf = normalize_country("SomeCountry")
        assert nf.normalized == "Somecountry"

        nf = normalize_country("UNITED KINGDOM")
        assert nf.normalized == "United Kingdom"

    def test_empty_handling(self):
        nf = normalize_country("")
        assert nf.normalized == ""

        nf = normalize_country(None)
        assert nf.normalized == ""


class TestJaccardTokens:
    """Tests for token Jaccard similarity."""

    def test_identical(self):
        assert jaccard_tokens(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_disjoint(self):
        assert jaccard_tokens(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self):
        assert jaccard_tokens(["a", "b", "c"], ["b", "c", "d"]) == 0.5

    def test_empty(self):
        assert jaccard_tokens([], ["a", "b"]) == 0.0
        assert jaccard_tokens([], []) == 0.0


class TestNormalizedField:
    """Tests for NormalizedField dataclass."""

    def test_dataclass_structure(self):
        nf = NormalizedField("orig", "norm", ["tok1", "tok2"], "compact")
        assert nf.original == "orig"
        assert nf.normalized == "norm"
        assert nf.tokenized == ["tok1", "tok2"]
        assert nf.compact == "compact"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])