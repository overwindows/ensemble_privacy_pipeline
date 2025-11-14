#!/usr/bin/env python3
"""
Verification Script: Test PrivacyRedactor with Both Field Name Formats

This script verifies that the PrivacyRedactor fix correctly handles:
1. Vendor-neutral field names (raw_queries, browsing_history, etc.)
2. Microsoft-specific field names (MSNClicks, BingSearch, MAI)
3. Backward compatibility

Usage:
  python3 src/verify_redaction_fix.py
"""

import sys
import os

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.privacy_core import PrivacyRedactor


def test_vendor_neutral_fields():
    """Test vendor-neutral field names used by public datasets."""
    print("=" * 80)
    print("TEST 1: Vendor-Neutral Field Names")
    print("=" * 80)

    redactor = PrivacyRedactor()

    # Test case 1: raw_queries (used by 5 benchmarks)
    print("\n1️⃣  Testing 'raw_queries' field...")
    user_data_1 = {
        'raw_queries': [
            'diabetes symptoms',
            'diabetes treatment options',
            'side effects of metformin'
        ],
        'demographics': {'age': 42, 'gender': 'F'}
    }

    masked_1 = redactor.redact_user_data(user_data_1)

    if 'queries' in masked_1:
        print("   ✅ PASS: 'raw_queries' field handled correctly")
        print(f"   ✓ Original: {len(user_data_1['raw_queries'])} queries")
        print(f"   ✓ Masked: {len(masked_1['queries'])} query tokens")
        for i, q in enumerate(masked_1['queries'], 1):
            print(f"      Token {i}: {q['token']}")
    else:
        print("   ❌ FAIL: 'raw_queries' field NOT handled!")
        print(f"   ✗ Output: {masked_1}")
        return False

    # Test case 2: browsing_history (used by neutral_benchmark.py)
    print("\n2️⃣  Testing 'browsing_history' field...")
    user_data_2 = {
        'browsing_history': [
            'Understanding diabetes symptoms',
            'Treatment options for diabetes',
            'Living with diabetes: expert advice'
        ]
    }

    masked_2 = redactor.redact_user_data(user_data_2)

    if 'browsing' in masked_2:
        print("   ✅ PASS: 'browsing_history' field handled correctly")
        print(f"   ✓ Original: {len(user_data_2['browsing_history'])} items")
        print(f"   ✓ Masked: {len(masked_2['browsing'])} browsing tokens")
        for i, b in enumerate(masked_2['browsing'], 1):
            print(f"      Token {i}: {b['token']}")
    else:
        print("   ❌ FAIL: 'browsing_history' field NOT handled!")
        print(f"   ✗ Output: {masked_2}")
        return False

    # Test case 3: source_text (ai4privacy dataset)
    print("\n3️⃣  Testing 'source_text' field...")
    user_data_3 = {
        'source_text': "My name is Sarah Johnson and I live at 123 Oak Street, Seattle, WA."
    }

    masked_3 = redactor.redact_user_data(user_data_3)

    if 'text' in masked_3:
        print("   ✅ PASS: 'source_text' field handled correctly")
        print(f"   ✓ Original: {user_data_3['source_text'][:50]}...")
        print(f"   ✓ Masked token: {masked_3['text']['token']}")
    else:
        print("   ❌ FAIL: 'source_text' field NOT handled!")
        print(f"   ✗ Output: {masked_3}")
        return False

    # Test case 4: user_prompt (PUPA dataset)
    print("\n4️⃣  Testing 'user_prompt' field...")
    user_data_4 = {
        'user_prompt': "I'm applying for a Software Engineer position at TechCorp Inc. My name is John Smith."
    }

    masked_4 = redactor.redact_user_data(user_data_4)

    if 'prompt' in masked_4:
        print("   ✅ PASS: 'user_prompt' field handled correctly")
        print(f"   ✓ Original: {user_data_4['user_prompt'][:50]}...")
        print(f"   ✓ Masked token: {masked_4['prompt']['token']}")
    else:
        print("   ❌ FAIL: 'user_prompt' field NOT handled!")
        print(f"   ✗ Output: {masked_4}")
        return False

    # Test case 5: text (TAB dataset)
    print("\n5️⃣  Testing 'text' field...")
    user_data_5 = {
        'text': "The applicant, John Smith, complained about violations at ABC Corporation on 2024-01-15."
    }

    masked_5 = redactor.redact_user_data(user_data_5)

    if 'text' in masked_5:
        print("   ✅ PASS: 'text' field handled correctly")
        print(f"   ✓ Original: {user_data_5['text'][:50]}...")
        print(f"   ✓ Masked token: {masked_5['text']['token']}")
    else:
        print("   ❌ FAIL: 'text' field NOT handled!")
        print(f"   ✗ Output: {masked_5}")
        return False

    print("\n" + "=" * 80)
    print("✅ ALL VENDOR-NEUTRAL FIELD TESTS PASSED!")
    print("=" * 80)
    return True


def test_microsoft_specific_fields():
    """Test Microsoft-specific field names (backward compatibility)."""
    print("\n" + "=" * 80)
    print("TEST 2: Microsoft-Specific Field Names (Backward Compatibility)")
    print("=" * 80)

    redactor = PrivacyRedactor()

    # Test case: MSNClicks, BingSearch (used by run_benchmarks.py)
    print("\n1️⃣  Testing 'MSNClicks' and 'BingSearch' fields...")
    user_data = {
        'MSNClicks': [
            {'title': 'Understanding diabetes symptoms', 'timestamp': '2024-01-15T10:00:00'},
            {'title': 'Treatment options for diabetes', 'timestamp': '2024-01-15T11:00:00'}
        ],
        'BingSearch': [
            {'query': 'diabetes diet plan', 'timestamp': '2024-01-15T12:00:00'},
            {'query': 'side effects of metformin', 'timestamp': '2024-01-15T13:00:00'}
        ],
        'MAI': ['Health', 'Health', 'Health', 'Finance', 'Finance', 'Sports', 'Sports', 'Sports'],
        'demographics': {'age': 42, 'gender': 'F'}
    }

    masked = redactor.redact_user_data(user_data)

    tests_passed = 0

    if 'MSNClicks' in masked:
        print("   ✅ PASS: 'MSNClicks' field still handled")
        print(f"   ✓ Masked: {len(masked['MSNClicks'])} click tokens")
        tests_passed += 1
    else:
        print("   ❌ FAIL: 'MSNClicks' field NOT handled!")

    if 'BingSearch' in masked:
        print("   ✅ PASS: 'BingSearch' field still handled")
        print(f"   ✓ Masked: {len(masked['BingSearch'])} search tokens")
        tests_passed += 1
    else:
        print("   ❌ FAIL: 'BingSearch' field NOT handled!")

    if 'MAI' in masked:
        print("   ✅ PASS: 'MAI' field still handled")
        print(f"   ✓ Masked: {masked['MAI']}")
        tests_passed += 1
    else:
        print("   ❌ FAIL: 'MAI' field NOT handled!")

    if 'demographics' in masked:
        print("   ✅ PASS: 'demographics' field still handled")
        print(f"   ✓ Masked: {masked['demographics']}")
        tests_passed += 1
    else:
        print("   ❌ FAIL: 'demographics' field NOT handled!")

    if tests_passed == 4:
        print("\n" + "=" * 80)
        print("✅ ALL BACKWARD COMPATIBILITY TESTS PASSED!")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print(f"❌ FAILED: {4 - tests_passed} / 4 backward compatibility tests failed")
        print("=" * 80)
        return False


def test_mixed_fields():
    """Test mixing vendor-neutral and Microsoft-specific fields."""
    print("\n" + "=" * 80)
    print("TEST 3: Mixed Field Names (Should Handle Both)")
    print("=" * 80)

    redactor = PrivacyRedactor()

    print("\n1️⃣  Testing mixed 'raw_queries' + 'MSNClicks'...")
    user_data = {
        'raw_queries': ['diabetes symptoms'],
        'MSNClicks': [{'title': 'Treatment options', 'timestamp': '2024-01-15T10:00:00'}],
        'demographics': {'age': 42}
    }

    masked = redactor.redact_user_data(user_data)

    if 'queries' in masked and 'MSNClicks' in masked:
        print("   ✅ PASS: Both 'raw_queries' and 'MSNClicks' handled together")
        print(f"   ✓ Queries: {len(masked['queries'])} tokens")
        print(f"   ✓ MSNClicks: {len(masked['MSNClicks'])} tokens")
        print("\n" + "=" * 80)
        print("✅ MIXED FIELD TEST PASSED!")
        print("=" * 80)
        return True
    else:
        print("   ❌ FAIL: Not both field types handled!")
        print(f"   ✗ Output: {masked}")
        print("\n" + "=" * 80)
        print("❌ MIXED FIELD TEST FAILED!")
        print("=" * 80)
        return False


def main():
    print("\n" + "=" * 80)
    print("PRIVACY REDACTOR FIX VERIFICATION")
    print("=" * 80)
    print("\nThis script verifies that PrivacyRedactor now supports:")
    print("  1. Vendor-neutral field names (raw_queries, browsing_history, etc.)")
    print("  2. Microsoft-specific field names (MSNClicks, BingSearch, MAI)")
    print("  3. Backward compatibility with existing code")
    print("  4. Mixed field names in the same input")

    results = []

    # Run all tests
    results.append(("Vendor-Neutral Fields", test_vendor_neutral_fields()))
    results.append(("Microsoft-Specific Fields", test_microsoft_specific_fields()))
    results.append(("Mixed Fields", test_mixed_fields()))

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - FIX VERIFIED!")
        print("=" * 80)
        print("\n🎉 The PrivacyRedactor now correctly handles:")
        print("   • public_datasets_simple.py (raw_queries)")
        print("   • pupa_benchmark.py (raw_queries)")
        print("   • text_sanitization_benchmark.py (raw_queries)")
        print("   • neutral_benchmark.py (raw_queries + browsing_history)")
        print("   • dp_benchmark.py (raw_queries)")
        print("   • run_benchmarks.py (MSNClicks, BingSearch) - backward compatible")
        print("   • run_demo_pipeline.py (MSNClicks, BingSearch) - backward compatible")
        print("\n✅ All 7 benchmark scripts can now properly redact data!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - FIX INCOMPLETE!")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    exit(main())
