#!/usr/bin/env python3
"""
Verify Ensemble Consensus Fix - Quick Demonstration

This script demonstrates the difference between the INVALID and VALID approaches
to ensemble consensus without requiring API calls or benchmark runs.
"""

import sys
from collections import Counter

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def simulate_ensemble_outputs():
    """Simulate 4 model outputs with some PII leakage"""
    return [
        ("I can help with that request.", 2),  # (output, pii_leaked_count)
        ("I can help with that request.", 0),
        ("I can assist you with that.", 1),
        ("I can help with that request.", 3),
    ]

def invalid_approach(outputs):
    """INVALID: Uses ground truth PII counts to select output"""
    print("\n" + "=" * 80)
    print("INVALID APPROACH (OLD CODE)")
    print("=" * 80)

    outputs_with_leakage = [(out, leaked) for out, leaked in outputs]

    print("\nOutputs from 4 models:")
    for i, (output, leaked) in enumerate(outputs_with_leakage, 1):
        print(f"  Model {i}: '{output}' (PII leaked: {leaked})")

    # This is WRONG - uses ground truth to select
    best_output = min(outputs_with_leakage, key=lambda x: x[1])

    print(f"\nSelection method: Pick output with minimum PII leakage")
    print(f"Selected: '{best_output[0]}' (leaked: {best_output[1]})")
    print("\nPROBLEM: This uses ground truth PII during selection!")
    print("         If you know ground truth, why do you need ensemble?")
    print("         This is circular logic and scientifically invalid.")

def valid_approach(outputs):
    """VALID: Uses majority voting WITHOUT ground truth"""
    print("\n" + "=" * 80)
    print("VALID APPROACH (NEW CODE)")
    print("=" * 80)

    # Extract just the text outputs (ignore leakage counts during selection)
    text_outputs = [out for out, _ in outputs]

    print("\nOutputs from 4 models:")
    for i, output in enumerate(text_outputs, 1):
        print(f"  Model {i}: '{output}'")

    # Count occurrences WITHOUT using ground truth
    output_counts = Counter(text_outputs)
    unique_outputs = len(output_counts)

    print(f"\nAggregation method: Majority voting (no ground truth used)")
    print(f"Unique outputs: {unique_outputs}")

    for output, count in output_counts.most_common():
        print(f"  '{output}': {count}/{len(text_outputs)} models")

    # Select based on majority (or fallback)
    if len(output_counts) == 1:
        final_output = text_outputs[0]
        consensus_type = "unanimous"
    elif output_counts.most_common(1)[0][1] >= len(text_outputs) / 2:
        final_output = output_counts.most_common(1)[0][0]
        consensus_type = "majority"
    else:
        final_output = min(text_outputs, key=len)
        consensus_type = "shortest_fallback"

    print(f"\nConsensus type: {consensus_type}")
    print(f"Final output: '{final_output}'")

    # NOW evaluate using ground truth (for measurement only)
    actual_leakage = next(leaked for out, leaked in outputs if out == final_output)
    print(f"\n--- EVALUATION PHASE (uses ground truth) ---")
    print(f"PII leaked in consensus output: {actual_leakage}")
    print("\nThis is VALID: Ground truth used for MEASUREMENT, not SELECTION")

def main():
    print("=" * 80)
    print("ENSEMBLE CONSENSUS: INVALID vs VALID APPROACH")
    print("=" * 80)
    print("\nThis demonstrates the fix for the ensemble consensus bug.")
    print("The key insight: Ground truth should be used for EVALUATION,")
    print("not for SELECTION of ensemble outputs.")

    outputs = simulate_ensemble_outputs()

    # Show both approaches
    invalid_approach(outputs)
    valid_approach(outputs)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nINVALID: Uses PII leakage counts to pick 'best' output")
    print("         → Circular logic, not applicable in production")
    print("\nVALID:   Uses majority voting to aggregate outputs")
    print("         → Then evaluates consensus output against ground truth")
    print("         → Simulates real production environment")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
