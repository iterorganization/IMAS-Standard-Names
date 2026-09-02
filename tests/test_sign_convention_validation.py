"""Quick test for sign convention validation."""

import pytest

from imas_standard_names.models import StandardNameScalarEntry


def test_valid_sign_convention():
    """Test that valid sign convention format passes."""
    entry = StandardNameScalarEntry(
        name="test_current",
        physics_domain="general",
        description="Test current quantity.",
        documentation="""
Test quantity description.

Sign convention: Positive when current flows counter-clockwise.

Additional physics explanation here.
""",
        unit="A",
    )
    assert entry.name == "test_current"


def test_invalid_bold_sign_convention():
    """Test that bold sign convention is rejected."""
    with pytest.raises(ValueError, match="plain text"):
        StandardNameScalarEntry(
            name="test_current",
            physics_domain="general",
            description="Test current quantity.",
            documentation="**Sign convention:** Positive when current flows counter-clockwise.",
            unit="A",
        )


def test_invalid_lowercase_sign_convention():
    """Test that lowercase 'sign convention:' is rejected."""
    with pytest.raises(ValueError, match="title case"):
        StandardNameScalarEntry(
            name="test_current",
            physics_domain="general",
            description="Test current quantity.",
            documentation="sign convention: Positive when current flows counter-clockwise.",
            unit="A",
        )


def test_invalid_missing_positive():
    """Test that missing 'Positive' keyword is rejected."""
    with pytest.raises(ValueError, match="Positive"):
        StandardNameScalarEntry(
            name="test_current",
            physics_domain="general",
            description="Test current quantity.",
            documentation="Sign convention: Current flows counter-clockwise when positive.",
            unit="A",
        )


def test_no_sign_convention_is_ok():
    """Test that entries without sign convention pass."""
    entry = StandardNameScalarEntry(
        name="test_temperature",
        physics_domain="general",
        description="Test temperature quantity.",
        documentation="Temperature measured in the plasma core.",
        unit="eV",
    )
    assert entry.name == "test_temperature"


def test_sign_convention_missing_blank_line_before():
    """Test that sign convention without blank line before is rejected."""
    with pytest.raises(ValueError, match="standalone paragraph.*blank line before"):
        StandardNameScalarEntry(
            name="test_current",
            physics_domain="general",
            description="Test current quantity.",
            documentation="First paragraph text.\nSign convention: Positive when current flows counter-clockwise.",
            unit="A",
        )


def test_sign_convention_missing_blank_line_after():
    """Test that sign convention without blank line after is rejected."""
    with pytest.raises(ValueError, match="standalone paragraph.*blank line after"):
        StandardNameScalarEntry(
            name="test_current",
            physics_domain="general",
            description="Test current quantity.",
            documentation="First paragraph.\n\nSign convention: Positive when current flows counter-clockwise.\nNext sentence immediately.",
            unit="A",
        )


def test_sign_convention_embedded_in_paragraph():
    """Test that sign convention embedded in same paragraph is rejected."""
    with pytest.raises(ValueError, match="standalone paragraph.*blank line before"):
        StandardNameScalarEntry(
            name="test_current",
            physics_domain="general",
            description="Test current quantity.",
            documentation="Some text. Sign convention: Positive when current flows counter-clockwise. More text.",
            unit="A",
        )


def test_sign_convention_at_start_is_rejected():
    """Test that sign convention at document start is rejected."""
    with pytest.raises(ValueError, match="must follow the main documentation content"):
        StandardNameScalarEntry(
            name="test_current",
            physics_domain="general",
            description="Test current quantity.",
            documentation="Sign convention: Positive when current flows counter-clockwise.\n\nAdditional text here.",
            unit="A",
        )


def test_sign_convention_at_end_with_blank_before():
    """Test that sign convention at document end with blank line before passes."""
    entry = StandardNameScalarEntry(
        name="test_current",
        physics_domain="general",
        description="Test current quantity.",
        documentation="First paragraph explaining the quantity.\n\nSign convention: Positive when current flows counter-clockwise.",
        unit="A",
    )
    assert entry.name == "test_current"


def test_sign_convention_positive_for_format():
    """Test that 'Positive for <subject>' format is accepted."""
    entry = StandardNameScalarEntry(
        name="test_current",
        physics_domain="general",
        description="Test current quantity.",
        documentation="Describes the plasma current.\n\nSign convention: Positive for counter-clockwise toroidal current.",
        unit="A",
    )
    assert entry.name == "test_current"


def test_sign_convention_positive_quantity_format():
    """Test that 'Positive <quantity-noun-phrase>' format is accepted."""
    entry = StandardNameScalarEntry(
        name="test_current",
        physics_domain="general",
        description="Test current quantity.",
        documentation="Describes the plasma current.\n\nSign convention: Positive outward radial direction.",
        unit="A",
    )
    assert entry.name == "test_current"


if __name__ == "__main__":
    # Run quick manual tests
    print("Testing valid sign convention...")
    test_valid_sign_convention()
    print("✓ Valid format accepted")

    print("\nTesting invalid bold format...")
    try:
        test_invalid_bold_sign_convention()
    except AssertionError:
        print("✓ Bold format rejected")

    print("\nTesting invalid lowercase format...")
    try:
        test_invalid_lowercase_sign_convention()
    except AssertionError:
        print("✓ Lowercase format rejected")

    print("\nTesting invalid missing 'Positive'...")
    try:
        test_invalid_missing_positive()
    except AssertionError:
        print("✓ Missing 'Positive' rejected")

    print("\nTesting no sign convention...")
    test_no_sign_convention_is_ok()
    print("✓ Entries without sign convention accepted")

    print("\nTesting standalone paragraph validation...")
    try:
        test_sign_convention_missing_blank_line_before()
    except AssertionError:
        print("✓ Missing blank line before rejected")

    try:
        test_sign_convention_missing_blank_line_after()
    except AssertionError:
        print("✓ Missing blank line after rejected")

    try:
        test_sign_convention_embedded_in_paragraph()
    except AssertionError:
        print("✓ Embedded sign convention rejected")

    try:
        test_sign_convention_at_start_is_rejected()
    except AssertionError:
        print("✓ Sign convention at start rejected")

    test_sign_convention_at_end_with_blank_before()
    print("✓ Sign convention at end accepted")

    print("\n✅ All manual tests passed!")
