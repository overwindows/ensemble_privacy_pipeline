"""
Benchmark Comparison: With vs Without Privacy-Preserving Pipeline

This script runs comprehensive evaluation comparing:
1. Baseline (no privacy) - Direct LLM evaluation with raw queries
2. Our Pipeline (with privacy) - Masked + ensemble + consensus

Outputs detailed comparison metrics and generates report.
"""

import sys
import json
import time
from typing import List, Dict, Any
import numpy as np

# Add parent directory to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import pipeline components
from src.privacy_core import PrivacyRedactor, ConsensusAggregator, analyze_privacy_leakage
from examples.real_llm_example import RealLLMEvaluator


# ============================================================================
# BASELINE MODEL (NO PRIVACY)
# ============================================================================

class BaselineModel:
    """
    Baseline model WITHOUT privacy protection.

    Directly processes raw queries and outputs detailed evidence.
    This is what systems do today - and why they leak PII!
    """

    def __init__(self):
        self.evaluator = None  # Will use simple mock logic

    def evaluate(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate WITHOUT privacy protection.

        Returns outputs that contain actual queries (PII leak!).
        """
        # Extract queries
        queries = []
        if 'BingSearch' in user_data:
            queries.extend([q['query'] for q in user_data['BingSearch']])
        if 'MSNClicks' in user_data:
            queries.extend([c['title'] for c in user_data['MSNClicks']])

        # Simulate scoring (in real system, would call LLM with raw queries)
        if self.evaluator:
            result = self.evaluator.evaluate_interest(user_data, topic="health")
        else:
            # Mock result
            result = {
                'score': 0.85,
                'reasoning': f"High interest based on queries: {', '.join(queries[:2])}"
            }

        # Baseline output includes ACTUAL QUERIES (privacy violation!)
        return {
            'ItemId': 'health-topic',
            'QualityScore': result.get('score', 0.85),
            'Evidence': queries[:3],  # ❌ Leaks actual queries!
            'QualityReason': result.get('reasoning', 'Based on search history')
        }


# ============================================================================
# PRIVACY-PRESERVING MODEL (OUR PIPELINE)
# ============================================================================

class PrivacyPreservingModel:
    """
    Our privacy-preserving pipeline.

    Uses: Masking → Ensemble → Consensus
    """

    def __init__(self, model_names: List[str], api_key: str):
        self.model_names = model_names
        self.api_key = api_key
        self.redactor = PrivacyRedactor()
        self.aggregator = ConsensusAggregator()

        # Create evaluators for each model
        self.evaluators = [
            RealLLMEvaluator(model_name=model, api_key=api_key)
            for model in model_names
        ]

    def evaluate(self, user_data: Dict[str, Any], candidate_topics: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate WITH privacy protection.

        Returns outputs with only generic metadata (no PII).
        """
        # Step 1: Redact sensitive data
        masked_data = self.redactor.redact_user_data(user_data)

        # Step 2: Ensemble evaluation
        all_results = []
        for evaluator in self.evaluators:
            try:
                results = evaluator.evaluate_interest(masked_data, candidate_topics)
                all_results.append(results)
            except Exception as e:
                print(f"Warning: Model {evaluator.model_name} failed: {e}")
                # Add fallback
                all_results.append([
                    {"ItemId": t["ItemId"], "QualityScore": 0.5, "QualityReason": "error"}
                    for t in candidate_topics
                ])

        # Step 3: Consensus aggregation
        consensus = self.aggregator.aggregate_median(all_results)

        # Return first result (single topic evaluation)
        if consensus:
            return consensus[0]
        else:
            return {
                'ItemId': candidate_topics[0]['ItemId'],
                'QualityScore': 0.5,
                'QualityReason': 'no consensus'
            }


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkRunner:
    """
    Runs comprehensive benchmarks comparing baseline vs. privacy-preserving pipeline.
    """

    def __init__(self):
        self.evaluator = EvaluationPipeline()
        self.baseline_model = BaselineModel()
        self.privacy_model = PrivacyPreservingModel(num_models=5)

    def run_full_benchmark(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Run complete benchmark evaluation.

        Args:
            num_samples: Number of test samples to evaluate

        Returns:
            Comprehensive results dictionary
        """
        print(f"\n{'='*70}")
        print(f"  PRIVACY-PRESERVING LLM PIPELINE - BENCHMARK EVALUATION")
        print(f"{'='*70}\n")

        results = {
            'config': {
                'num_samples': num_samples,
                'ensemble_size': 5,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'datasets': {},
            'metrics': {},
            'comparison': {}
        }

        # ===== MEDICAL DATA BENCHMARK =====
        print("1. Medical Data Benchmark (Sensitive PHI)")
        print("-" * 70)

        medical_dataset = self.evaluator.dataset_loader.load_synthetic_pii_dataset(num_samples)
        medical_results = self._run_dataset_evaluation(
            dataset=medical_dataset,
            dataset_name="Medical"
        )
        results['datasets']['medical'] = medical_results

        # ===== FINANCIAL DATA BENCHMARK =====
        print("\n2. Financial Data Benchmark (Sensitive Financial Info)")
        print("-" * 70)

        financial_dataset = self.evaluator.dataset_loader.load_financial_pii_dataset(num_samples)
        financial_results = self._run_dataset_evaluation(
            dataset=financial_dataset,
            dataset_name="Financial"
        )
        results['datasets']['financial'] = financial_results

        # ===== AGGREGATE METRICS =====
        print("\n3. Computing Aggregate Metrics...")
        print("-" * 70)

        results['metrics'] = self._compute_aggregate_metrics(results['datasets'])

        # ===== GENERATE COMPARISON REPORT =====
        results['comparison'] = self._generate_comparison_report(results['metrics'])

        return results

    def _run_dataset_evaluation(self, dataset: List[Dict], dataset_name: str) -> Dict[str, Any]:
        """Run evaluation on a single dataset."""

        print(f"   Evaluating {len(dataset)} samples...")

        # Prepare data
        user_data_samples = []
        ground_truth_queries = []

        for sample in dataset:
            # Convert to user data format
            user_data = {
                'BingSearch': [{'query': q} for q in sample['queries']],
                'MSNClicks': []
            }
            user_data_samples.append(user_data)
            ground_truth_queries.append(sample['queries'])

        # Run baseline (NO privacy)
        print(f"   Running baseline (no privacy)...")
        baseline_outputs = []
        baseline_start = time.time()

        for user_data in user_data_samples:
            output = self.baseline_model.evaluate(user_data)
            baseline_outputs.append(json.dumps(output))

        baseline_time = time.time() - baseline_start

        # Run privacy-preserving pipeline
        print(f"   Running privacy-preserving pipeline...")
        privacy_outputs = []
        privacy_start = time.time()

        for user_data in user_data_samples:
            output = self.privacy_model.evaluate(user_data)
            privacy_outputs.append(json.dumps(output))

        privacy_time = time.time() - privacy_start

        # Evaluate privacy
        print(f"   Evaluating privacy metrics...")

        privacy_pii = self.evaluator.privacy_evaluator.detect_pii_leakage(privacy_outputs)
        baseline_pii = self.evaluator.privacy_evaluator.detect_pii_leakage(baseline_outputs)

        privacy_recon = self.evaluator.privacy_evaluator.evaluate_reconstruction_attack(
            privacy_outputs, ground_truth_queries
        )
        baseline_recon = self.evaluator.privacy_evaluator.evaluate_reconstruction_attack(
            baseline_outputs, ground_truth_queries
        )

        # Compile results
        results = {
            'privacy': {
                'pii_leakage': {
                    'baseline': baseline_pii['leakage_rate'],
                    'privacy_preserving': privacy_pii['leakage_rate'],
                    'improvement_pct': (baseline_pii['leakage_rate'] - privacy_pii['leakage_rate']) * 100
                },
                'reconstruction_attack': {
                    'baseline': baseline_recon['reconstruction_rate'],
                    'privacy_preserving': privacy_recon['reconstruction_rate'],
                    'improvement_pct': (baseline_recon['reconstruction_rate'] - privacy_recon['reconstruction_rate']) * 100
                },
                'detailed': {
                    'baseline_pii': baseline_pii,
                    'privacy_pii': privacy_pii,
                    'baseline_recon': baseline_recon,
                    'privacy_recon': privacy_recon
                }
            },
            'performance': {
                'baseline_time_sec': baseline_time,
                'privacy_time_sec': privacy_time,
                'overhead_multiplier': privacy_time / baseline_time if baseline_time > 0 else 0
            }
        }

        # Print summary
        print(f"\n   Results for {dataset_name} Dataset:")
        print(f"   ├─ PII Leakage:")
        print(f"   │  ├─ Baseline:          {baseline_pii['leakage_rate']*100:.1f}%")
        print(f"   │  ├─ With Privacy:      {privacy_pii['leakage_rate']*100:.1f}%")
        print(f"   │  └─ Improvement:       {results['privacy']['pii_leakage']['improvement_pct']:.1f}%")
        print(f"   ├─ Reconstruction Attack:")
        print(f"   │  ├─ Baseline Success:  {baseline_recon['reconstruction_rate']*100:.1f}%")
        print(f"   │  ├─ With Privacy:      {privacy_recon['reconstruction_rate']*100:.1f}%")
        print(f"   │  └─ Improvement:       {results['privacy']['reconstruction_attack']['improvement_pct']:.1f}%")
        print(f"   └─ Performance:")
        print(f"      ├─ Baseline Time:    {baseline_time:.2f}s")
        print(f"      ├─ Privacy Time:     {privacy_time:.2f}s")
        print(f"      └─ Overhead:         {results['performance']['overhead_multiplier']:.1f}x")

        return results

    def _compute_aggregate_metrics(self, dataset_results: Dict) -> Dict[str, Any]:
        """Compute aggregate metrics across all datasets."""

        all_pii_baseline = []
        all_pii_privacy = []
        all_recon_baseline = []
        all_recon_privacy = []

        for dataset_name, results in dataset_results.items():
            all_pii_baseline.append(results['privacy']['pii_leakage']['baseline'])
            all_pii_privacy.append(results['privacy']['pii_leakage']['privacy_preserving'])
            all_recon_baseline.append(results['privacy']['reconstruction_attack']['baseline'])
            all_recon_privacy.append(results['privacy']['reconstruction_attack']['privacy_preserving'])

        return {
            'avg_pii_leakage': {
                'baseline': np.mean(all_pii_baseline),
                'privacy_preserving': np.mean(all_pii_privacy),
                'improvement_pct': (np.mean(all_pii_baseline) - np.mean(all_pii_privacy)) * 100
            },
            'avg_reconstruction': {
                'baseline': np.mean(all_recon_baseline),
                'privacy_preserving': np.mean(all_recon_privacy),
                'improvement_pct': (np.mean(all_recon_baseline) - np.mean(all_recon_privacy)) * 100
            }
        }

    def _generate_comparison_report(self, metrics: Dict) -> Dict[str, str]:
        """Generate human-readable comparison report."""

        pii_improvement = metrics['avg_pii_leakage']['improvement_pct']
        recon_improvement = metrics['avg_reconstruction']['improvement_pct']

        report = {
            'privacy_improvement': f"{pii_improvement:.1f}% reduction in PII leakage",
            'reconstruction_prevention': f"{recon_improvement:.1f}% reduction in reconstruction success",
            'overall_verdict': ""
        }

        # Overall verdict
        if pii_improvement >= 90 and recon_improvement >= 90:
            report['overall_verdict'] = "✅ EXCELLENT: Privacy-preserving pipeline provides >90% improvement"
        elif pii_improvement >= 70 and recon_improvement >= 70:
            report['overall_verdict'] = "✅ GOOD: Privacy-preserving pipeline provides >70% improvement"
        elif pii_improvement >= 50 and recon_improvement >= 50:
            report['overall_verdict'] = "⚠️  MODERATE: Privacy-preserving pipeline provides >50% improvement"
        else:
            report['overall_verdict'] = "❌ WEAK: Privacy-preserving pipeline provides <50% improvement"

        return report

    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run benchmark comparison."""

    print("\n" + "="*70)
    print("  PRIVACY-PRESERVING LLM PIPELINE")
    print("  Benchmark Evaluation: With vs. Without Privacy")
    print("="*70 + "\n")

    # Initialize runner
    runner = BenchmarkRunner()

    # Run benchmarks
    results = runner.run_full_benchmark(num_samples=100)

    # Print final summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)

    print(f"\nAggregate Results (across all datasets):")
    print(f"├─ PII Leakage:")
    print(f"│  ├─ Baseline (no privacy):      {results['metrics']['avg_pii_leakage']['baseline']*100:.1f}%")
    print(f"│  ├─ With Privacy Pipeline:      {results['metrics']['avg_pii_leakage']['privacy_preserving']*100:.1f}%")
    print(f"│  └─ Improvement:                {results['metrics']['avg_pii_leakage']['improvement_pct']:.1f}%")
    print(f"│")
    print(f"├─ Reconstruction Attack:")
    print(f"│  ├─ Baseline Success Rate:      {results['metrics']['avg_reconstruction']['baseline']*100:.1f}%")
    print(f"│  ├─ With Privacy Pipeline:      {results['metrics']['avg_reconstruction']['privacy_preserving']*100:.1f}%")
    print(f"│  └─ Improvement:                {results['metrics']['avg_reconstruction']['improvement_pct']:.1f}%")
    print(f"│")
    print(f"└─ Overall Verdict:")
    print(f"   {results['comparison']['overall_verdict']}")

    print("\n" + "="*70)

    # Save results
    runner.save_results(results)

    print("\nBenchmark evaluation complete!")
    print("Review benchmark_results.json for detailed metrics.")

    return results


if __name__ == "__main__":
    results = main()
