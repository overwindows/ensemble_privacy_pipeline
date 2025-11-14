"""
Quick Test Script for Public Benchmark Evaluation

This script runs a quick sanity check to ensure all benchmark loaders
and evaluation components are working correctly.

Usage:
    python test_benchmarks.py
"""

import sys
import json
from typing import Dict, Any


def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*70)
    print("  TEST 1: Checking Imports")
    print("="*70)

    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False

    try:
        from evaluation_framework import (
            PrivacyEvaluator,
            UtilityEvaluator
        )
        print("✓ evaluation_framework imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluation_framework: {e}")
        return False

    try:
        from ensemble_privacy_pipeline import (
            PrivacyRedactor,
            MockLLMEvaluator,
            ConsensusAggregator
        )
        print("✓ ensemble_privacy_pipeline imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ensemble_privacy_pipeline: {e}")
        return False

    try:
        from benchmark_public_datasets import (
            PublicBenchmarkLoader,
            PublicBenchmarkEvaluator
        )
        print("✓ benchmark_public_datasets imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import benchmark_public_datasets: {e}")
        return False

    # Optional: Check Hugging Face datasets
    try:
        import datasets
        print("✓ datasets (Hugging Face) imported successfully")
        hf_available = True
    except ImportError:
        print("⚠  datasets (Hugging Face) not available - using fallback synthetic data")
        hf_available = False

    print("\n✓ All required imports successful")
    return True, hf_available


def test_benchmark_loaders(hf_available: bool):
    """Test that benchmark loaders work correctly."""
    print("\n" + "="*70)
    print("  TEST 2: Testing Benchmark Loaders")
    print("="*70)

    from benchmark_public_datasets import PublicBenchmarkLoader

    loader = PublicBenchmarkLoader()

    # Test 1: ai4privacy dataset
    print("\nTesting ai4privacy loader (10 samples)...")
    try:
        dataset = loader.load_ai4privacy_dataset(num_samples=10)
        if dataset:
            print(f"✓ Loaded {len(dataset)} samples from ai4privacy")
            print(f"  Sample structure: {list(dataset[0].keys())}")
        else:
            print("✗ Failed to load ai4privacy dataset")
            return False
    except Exception as e:
        print(f"✗ Error loading ai4privacy: {e}")
        return False

    # Test 2: PII-Bench dataset
    print("\nTesting PII-Bench loader (10 samples)...")
    try:
        dataset = loader.load_pii_bench_dataset(num_samples=10)
        if dataset:
            print(f"✓ Generated {len(dataset)} PII-Bench samples")
            print(f"  Sample structure: {list(dataset[0].keys())}")
        else:
            print("✗ Failed to generate PII-Bench dataset")
            return False
    except Exception as e:
        print(f"✗ Error generating PII-Bench: {e}")
        return False

    # Test 3: PrivacyXray dataset
    print("\nTesting PrivacyXray loader (10 samples)...")
    try:
        dataset = loader.load_privacyxray_dataset(num_samples=10)
        if dataset:
            print(f"✓ Generated {len(dataset)} PrivacyXray samples")
            print(f"  Sample structure: {list(dataset[0].keys())}")
        else:
            print("✗ Failed to generate PrivacyXray dataset")
            return False
    except Exception as e:
        print(f"✗ Error generating PrivacyXray: {e}")
        return False

    print("\n✓ All benchmark loaders working correctly")
    return True


def test_privacy_evaluation():
    """Test privacy evaluation metrics."""
    print("\n" + "="*70)
    print("  TEST 3: Testing Privacy Evaluation")
    print("="*70)

    from evaluation_framework import PrivacyEvaluator

    evaluator = PrivacyEvaluator()

    # Test PII detection
    print("\nTesting PII detection...")
    test_outputs = [
        "User email is john.doe@example.com",  # Should detect email
        "Call me at 555-123-4567",              # Should detect phone
        "I have diabetes",                       # Should detect medical condition
        "Generic score: 0.85"                    # Should detect nothing
    ]

    try:
        results = evaluator.detect_pii_leakage(test_outputs)
        print(f"✓ PII Detection Results:")
        print(f"  - Outputs with PII: {results['outputs_with_pii']}/{results['total_outputs']}")
        print(f"  - Leakage rate: {results['leakage_rate']*100:.1f}%")
        print(f"  - PII types found: {dict(results['pii_by_type'])}")
    except Exception as e:
        print(f"✗ Error in PII detection: {e}")
        return False

    # Test reconstruction attack
    print("\nTesting reconstruction attack evaluation...")
    outputs = [
        "User searched for: diabetes symptoms, insulin treatment",
        "Generic evidence: MSNClicks+BingSearch"
    ]
    ground_truth = [
        ["diabetes symptoms", "insulin treatment"],
        ["diabetes symptoms", "insulin treatment"]
    ]

    try:
        results = evaluator.evaluate_reconstruction_attack(outputs, ground_truth)
        print(f"✓ Reconstruction Attack Results:")
        print(f"  - Successful: {results['successful_reconstructions']}")
        print(f"  - Partial: {results['partial_reconstructions']}")
        print(f"  - Failed: {results['failed_reconstructions']}")
        print(f"  - Success rate: {results['reconstruction_rate']*100:.1f}%")
    except Exception as e:
        print(f"✗ Error in reconstruction attack: {e}")
        return False

    print("\n✓ Privacy evaluation working correctly")
    return True


def test_end_to_end_evaluation():
    """Test end-to-end benchmark evaluation."""
    print("\n" + "="*70)
    print("  TEST 4: Testing End-to-End Evaluation")
    print("="*70)

    from benchmark_public_datasets import PublicBenchmarkEvaluator

    evaluator = PublicBenchmarkEvaluator()

    # Run quick evaluation (10 samples only)
    print("\nRunning quick evaluation on PII-Bench (10 samples)...")
    try:
        results = evaluator.evaluate_on_benchmark(
            benchmark_name='pii-bench',
            num_samples=10
        )

        if results:
            print("\n✓ End-to-end evaluation completed successfully")
            print(f"\nQuick Results Summary:")
            pii = results['privacy_metrics']['pii_leakage']
            recon = results['privacy_metrics']['reconstruction_attack']
            print(f"  PII Leakage Improvement: {pii['improvement_pct']:.1f}%")
            print(f"  Reconstruction Prevention: {recon['improvement_pct']:.1f}%")
        else:
            print("✗ End-to-end evaluation returned empty results")
            return False

    except Exception as e:
        print(f"✗ Error in end-to-end evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ End-to-end evaluation working correctly")
    return True


def test_pipeline_integration():
    """Test that privacy pipeline integrates correctly with benchmarks."""
    print("\n" + "="*70)
    print("  TEST 5: Testing Pipeline Integration")
    print("="*70)

    from ensemble_privacy_pipeline import PrivacyRedactor

    redactor = PrivacyRedactor()

    # Test data
    test_user_data = {
        "BingSearch": [
            {"query": "diabetes symptoms", "timestamp": "2024-01-15"},
            {"query": "insulin treatment", "timestamp": "2024-01-14"}
        ],
        "MSNClicks": [
            {"title": "Managing diabetes effectively", "timestamp": "2024-01-15"}
        ],
        "demographics": {
            "age": 42,
            "gender": "F"
        }
    }

    print("\nTesting redaction on sample data...")
    try:
        masked_data = redactor.redact_user_data(test_user_data)
        print("✓ Redaction successful")
        print(f"  Original queries: {len(test_user_data['BingSearch'])}")
        print(f"  Masked tokens: {len(masked_data['BingSearch'])}")

        # Verify no raw queries in masked data
        masked_str = json.dumps(masked_data)
        if "diabetes" in masked_str or "insulin" in masked_str:
            print("✗ WARNING: Raw queries still present in masked data!")
            return False
        else:
            print("✓ Verified: No raw queries in masked data")

    except Exception as e:
        print(f"✗ Error in redaction: {e}")
        return False

    print("\n✓ Pipeline integration working correctly")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  BENCHMARK EVALUATION - QUICK TEST SUITE")
    print("="*70)

    all_passed = True

    # Test 1: Imports
    result = test_imports()
    if isinstance(result, tuple):
        passed, hf_available = result
        if not passed:
            print("\n✗ Import test failed. Please install required dependencies:")
            print("  pip install numpy scikit-learn")
            print("  pip install datasets huggingface_hub  # For ai4privacy dataset")
            return False
    else:
        hf_available = False

    # Test 2: Benchmark loaders
    if not test_benchmark_loaders(hf_available):
        print("\n✗ Benchmark loader test failed")
        all_passed = False

    # Test 3: Privacy evaluation
    if not test_privacy_evaluation():
        print("\n✗ Privacy evaluation test failed")
        all_passed = False

    # Test 4: Pipeline integration
    if not test_pipeline_integration():
        print("\n✗ Pipeline integration test failed")
        all_passed = False

    # Test 5: End-to-end
    if not test_end_to_end_evaluation():
        print("\n✗ End-to-end evaluation test failed")
        all_passed = False

    # Final summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)

    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("\nYou can now run full benchmark evaluations:")
        print("  python benchmark_public_datasets.py --benchmark ai4privacy --num_samples 1000")
        print("  python benchmark_public_datasets.py --benchmark all --num_samples 500")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease check the errors above and ensure all dependencies are installed.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
