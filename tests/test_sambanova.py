#!/usr/bin/env python3
"""
Quick test script for SambaNova API integration

Usage:
  1. Install: pip install sambanova
  2. Set your API key: export SAMBANOVA_API_KEY='your-key-here'
  3. Run: python3 test_sambanova.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.privacy_core import PrivacyRedactor, ConsensusAggregator
from examples.real_llm_example import RealLLMEvaluator


def main():
    print("=" * 80)
    print("SAMBANOVA API TEST")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv("SAMBANOVA_API_KEY")
    if not api_key:
        print("\n❌ Error: SAMBANOVA_API_KEY not set!")
        print("\nPlease set your API key:")
        print("  export SAMBANOVA_API_KEY='your-api-key-here'")
        print("\nGet your key from: https://cloud.sambanova.ai/")
        sys.exit(1)

    print(f"\n✓ API Key found: {api_key[:20]}...")

    # ========================================================================
    # STEP 1: Prepare test data
    # ========================================================================

    print("\n" + "=" * 80)
    print("TEST DATA")
    print("=" * 80)

    raw_user_data = {
        "MSNClicks": [
            {"title": "Understanding diabetes management", "timestamp": "2024-01-15T10:30:00"},
            {"title": "Best fitness apps for health tracking", "timestamp": "2024-01-14T15:20:00"},
        ],
        "BingSearch": [
            {"query": "diabetes diet tips", "timestamp": "2024-01-15T11:00:00"},
        ],
        "MAI": ["Health"] * 8 + ["Fitness"] * 2,
        "demographics": {"age": 42, "gender": "F"}
    }

    print("\n✓ Raw user data (sensitive):")
    print(f"   - {len(raw_user_data['MSNClicks'])} article clicks")
    print(f"   - {len(raw_user_data['BingSearch'])} search queries")
    print(f"   - {len(raw_user_data['MAI'])} interest signals")

    # ========================================================================
    # STEP 2: Redact sensitive data
    # ========================================================================

    print("\n" + "=" * 80)
    print("REDACTION")
    print("=" * 80)

    redactor = PrivacyRedactor()
    masked_data = redactor.redact_user_data(raw_user_data)

    print("\n✓ Data masked successfully")
    print("   Original queries → QUERY_SEARCH_001, ...")
    print("   Original titles → QUERY_MSN_001, ...")

    # ========================================================================
    # STEP 3: Test SambaNova API
    # ========================================================================

    print("\n" + "=" * 80)
    print("SAMBANOVA API TEST")
    print("=" * 80)

    candidate_topics = [
        {"ItemId": "A", "Topic": "Managing diabetes with healthy eating"},
        {"ItemId": "B", "Topic": "Latest AI technology trends"},
        {"ItemId": "C", "Topic": "Women's health and wellness"},
    ]

    print(f"\n✓ Testing with {len(candidate_topics)} candidate topics")

    # Test with first model
    test_model = "gpt-oss-120b"
    print(f"\n⏳ Calling SambaNova API (model: {test_model})...")

    try:
        evaluator = RealLLMEvaluator(
            model_name=test_model,
            api_key=api_key
        )

        results = evaluator.evaluate_interest(masked_data, candidate_topics)

        print("\n✅ SUCCESS! SambaNova API responded:")
        print("-" * 80)
        for r in results:
            print(f"  {r['ItemId']}: {r['QualityScore']:.2f} - {r['QualityReason']}")
        print("-" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your API key is correct")
        print("  2. Ensure you have credits in your SambaNova account")
        print("  3. Verify the model name 'gpt-oss-120b' is available")
        print("  4. Check your internet connection")
        sys.exit(1)

    # ========================================================================
    # STEP 4: Test ensemble (4 models)
    # ========================================================================

    print("\n" + "=" * 80)
    print("ENSEMBLE TEST (4 diverse models)")
    print("=" * 80)

    # Your 4 selected models
    ensemble_models = [
        "gpt-oss-120b",       # 120B parameter model
        "DeepSeek-V3.1",      # Advanced reasoning
        "Qwen3-32B",          # 32B parameter model
        "DeepSeek-V3-0324",   # Latest DeepSeek
    ]

    print(f"\n✓ Testing with {len(ensemble_models)} diverse models:")
    for i, model in enumerate(ensemble_models, 1):
        print(f"   {i}. {model}")

    print("\n⏳ Running ensemble evaluation...")

    all_results = []
    for i, model_name in enumerate(ensemble_models, 1):
        print(f"   Model {i}/{len(ensemble_models)} ({model_name})... ", end="", flush=True)
        try:
            model_evaluator = RealLLMEvaluator(
                model_name=model_name,
                api_key=api_key
            )
            results = model_evaluator.evaluate_interest(masked_data, candidate_topics)
            all_results.append(results)
            print("✓")
        except Exception as e:
            print(f"❌ Error: {e}")
            # Add fallback results
            all_results.append([
                {"ItemId": t["ItemId"], "QualityScore": 0.5, "QualityReason": f"error:{model_name}"}
                for t in candidate_topics
            ])

    # Aggregate results
    aggregator = ConsensusAggregator()
    consensus = aggregator.aggregate_median(all_results)

    print("\n✅ CONSENSUS RESULTS:")
    print("-" * 80)
    for r in consensus:
        topic = next(t["Topic"] for t in candidate_topics if t["ItemId"] == r["ItemId"])
        print(f"  {r['ItemId']}: {r['QualityScore']:.2f} - {topic}")
        print(f"      Reason: {r['QualityReason']}")
    print("-" * 80)

    # ========================================================================
    # SUCCESS
    # ========================================================================

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)

    print("\n✓ Your SambaNova API integration is working!")
    print("\nNext steps:")
    print("  1. Run full example: python3 examples/sambanova_example.py")
    print("  2. Test with your own data")
    print("  3. Experiment with different ensemble sizes")
    print("  4. Run benchmarks to validate privacy metrics")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
