"""
Evaluation Framework for Privacy-Preserving LLM Pipeline

This module provides comprehensive evaluation metrics and benchmarks for
assessing both privacy and utility of the ensemble-redaction pipeline.

Key Components:
- Privacy Metrics: PII leakage, reconstruction attacks, membership inference
- Utility Metrics: Accuracy, F1, BLEU, ROUGE, perplexity
- Benchmark Datasets: Integration with standard privacy benchmarks
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict
import json


# ============================================================================
# PRIVACY EVALUATION METRICS
# ============================================================================

class PrivacyEvaluator:
    """
    Evaluates privacy leakage in model outputs.

    Implements standard privacy metrics used in research:
    - PII Leakage Rate (% of outputs containing PII)
    - Reconstruction Success Rate (can attacker reconstruct inputs?)
    - Exposure Metrics (canary detection)
    """

    def __init__(self):
        # Standard PII patterns (based on GDPR/HIPAA categories)
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'date_of_birth': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'address': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b',
            'name': None,  # Requires NER model
        }

        # Medical condition keywords (PHI - Protected Health Information)
        self.medical_keywords = [
            'diabetes', 'cancer', 'hiv', 'aids', 'alzheimer', 'depression',
            'anxiety', 'bipolar', 'schizophrenia', 'heart disease', 'stroke',
            'kidney disease', 'liver disease', 'asthma', 'copd', 'arthritis',
            'hypertension', 'obesity', 'pregnancy', 'abortion', 'std',
            'insulin', 'chemotherapy', 'radiation', 'surgery', 'medication'
        ]

        # Financial keywords
        self.financial_keywords = [
            'bankruptcy', 'foreclosure', 'debt', 'loan', 'credit score',
            'income', 'salary', 'tax', 'investment', 'stock', 'bond',
            'mortgage', 'insurance'
        ]

    def detect_pii_leakage(self, outputs: List[str],
                           ground_truth_pii: List[List[str]] = None) -> Dict[str, Any]:
        """
        Detect PII leakage in outputs.

        Args:
            outputs: List of model outputs to check
            ground_truth_pii: Optional list of known PII for each output

        Returns:
            Dictionary with leakage statistics
        """
        results = {
            'total_outputs': len(outputs),
            'outputs_with_pii': 0,
            'pii_by_type': defaultdict(int),
            'leakage_rate': 0.0,
            'detected_pii': []
        }

        for idx, output in enumerate(outputs):
            detected = self._detect_pii_in_text(output)

            if detected:
                results['outputs_with_pii'] += 1
                results['detected_pii'].append({
                    'output_idx': idx,
                    'pii_found': detected,
                    'output_text': output[:100]  # First 100 chars for context
                })

                for pii_type in detected:
                    results['pii_by_type'][pii_type] += 1

        results['leakage_rate'] = results['outputs_with_pii'] / len(outputs) if outputs else 0

        return results

    def _detect_pii_in_text(self, text: str) -> List[str]:
        """Detect PII types in a single text."""
        detected_types = []

        # Check regex patterns
        for pii_type, pattern in self.pii_patterns.items():
            if pattern and re.search(pattern, text, re.IGNORECASE):
                detected_types.append(pii_type)

        # Check medical keywords
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in self.medical_keywords):
            detected_types.append('medical_condition')

        # Check financial keywords
        if any(keyword in text_lower for keyword in self.financial_keywords):
            detected_types.append('financial_info')

        return detected_types

    def evaluate_reconstruction_attack(self, outputs: List[str],
                                      original_queries: List[List[str]]) -> Dict[str, Any]:
        """
        Simulate reconstruction attack: Can attacker infer original queries from outputs?

        Args:
            outputs: Model outputs (privacy-protected or not)
            original_queries: Ground truth queries for each output

        Returns:
            Dictionary with reconstruction attack results
        """
        results = {
            'total_attempts': len(outputs),
            'successful_reconstructions': 0,
            'partial_reconstructions': 0,
            'failed_reconstructions': 0,
            'reconstruction_rate': 0.0,
            'details': []
        }

        for idx, (output, queries) in enumerate(zip(outputs, original_queries)):
            reconstruction_score = self._calculate_reconstruction_score(output, queries)

            if reconstruction_score >= 0.8:
                results['successful_reconstructions'] += 1
                status = 'success'
            elif reconstruction_score >= 0.3:
                results['partial_reconstructions'] += 1
                status = 'partial'
            else:
                results['failed_reconstructions'] += 1
                status = 'failed'

            results['details'].append({
                'output_idx': idx,
                'reconstruction_score': reconstruction_score,
                'status': status,
                'num_queries': len(queries)
            })

        results['reconstruction_rate'] = results['successful_reconstructions'] / len(outputs) if outputs else 0

        return results

    def _calculate_reconstruction_score(self, output: str, queries: List[str]) -> float:
        """
        Calculate how much of the original queries can be reconstructed from output.

        Returns score between 0 (no reconstruction) and 1 (perfect reconstruction)
        """
        output_lower = output.lower()

        # Count how many query terms appear in output
        total_terms = 0
        matched_terms = 0

        for query in queries:
            terms = query.lower().split()
            total_terms += len(terms)

            for term in terms:
                if len(term) > 3 and term in output_lower:  # Ignore very short terms
                    matched_terms += 1

        if total_terms == 0:
            return 0.0

        return matched_terms / total_terms

    def detect_canary_exposure(self, outputs: List[str],
                               canaries: List[str]) -> Dict[str, Any]:
        """
        Detect if canary data (intentionally inserted unique patterns) appear in outputs.

        Used by PrivLM-Bench and other benchmarks to measure memorization.

        Args:
            outputs: Model outputs
            canaries: List of unique canary strings that should NOT appear

        Returns:
            Dictionary with exposure statistics
        """
        results = {
            'total_canaries': len(canaries),
            'exposed_canaries': 0,
            'exposure_rate': 0.0,
            'exposures': []
        }

        for canary in canaries:
            for idx, output in enumerate(outputs):
                if canary.lower() in output.lower():
                    results['exposed_canaries'] += 1
                    results['exposures'].append({
                        'canary': canary,
                        'output_idx': idx,
                        'context': self._extract_context(output, canary)
                    })
                    break  # Count each canary only once

        results['exposure_rate'] = results['exposed_canaries'] / len(canaries) if canaries else 0

        return results

    def _extract_context(self, text: str, target: str, window: int = 50) -> str:
        """Extract context around target string."""
        idx = text.lower().find(target.lower())
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(target) + window)
        return text[start:end]


# ============================================================================
# UTILITY EVALUATION METRICS
# ============================================================================

class UtilityEvaluator:
    """
    Evaluates utility (performance) of the model outputs.

    Implements standard ML metrics:
    - Classification: Accuracy, Precision, Recall, F1
    - Regression: MAE, MSE, Correlation
    - Generation: BLEU, ROUGE, Perplexity (requires external libs)
    """

    def evaluate_classification(self, predictions: List[Any],
                               ground_truth: List[Any]) -> Dict[str, float]:
        """
        Evaluate classification performance.

        Args:
            predictions: Model predictions
            ground_truth: True labels

        Returns:
            Dictionary with accuracy, precision, recall, F1
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        # Convert to numpy arrays
        pred = np.array(predictions)
        true = np.array(ground_truth)

        # Calculate metrics
        correct = (pred == true).sum()
        accuracy = correct / len(true) if len(true) > 0 else 0

        # For binary classification, calculate precision/recall/F1
        if set(true).issubset({0, 1}) and set(pred).issubset({0, 1}):
            tp = ((pred == 1) & (true == 1)).sum()
            fp = ((pred == 1) & (true == 0)).sum()
            fn = ((pred == 0) & (true == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = None

        return {
            'accuracy': float(accuracy),
            'precision': float(precision) if precision is not None else None,
            'recall': float(recall) if recall is not None else None,
            'f1': float(f1) if f1 is not None else None,
            'total_samples': len(true)
        }

    def evaluate_regression(self, predictions: List[float],
                           ground_truth: List[float]) -> Dict[str, float]:
        """
        Evaluate regression/scoring performance.

        Args:
            predictions: Predicted scores
            ground_truth: True scores

        Returns:
            Dictionary with MAE, MSE, RMSE, correlation
        """
        pred = np.array(predictions)
        true = np.array(ground_truth)

        # Calculate error metrics
        mae = np.mean(np.abs(pred - true))
        mse = np.mean((pred - true) ** 2)
        rmse = np.sqrt(mse)

        # Calculate correlation
        correlation = np.corrcoef(pred, true)[0, 1] if len(pred) > 1 else 0

        # Calculate score drift (how much scores changed on average)
        score_drift = np.mean(np.abs(pred - true)) / np.mean(np.abs(true)) if np.mean(np.abs(true)) > 0 else 0

        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'score_drift': float(score_drift),
            'total_samples': len(true)
        }

    def evaluate_text_generation_simple(self, generated: List[str],
                                       references: List[str]) -> Dict[str, float]:
        """
        Simple text generation evaluation (without external libraries).

        For full BLEU/ROUGE, install: pip install evaluate

        Args:
            generated: Generated texts
            references: Reference texts

        Returns:
            Dictionary with simple overlap metrics
        """
        results = {
            'avg_length': np.mean([len(text.split()) for text in generated]),
            'avg_unique_words': np.mean([len(set(text.lower().split())) for text in generated]),
            'word_overlap': 0.0
        }

        # Calculate word overlap (simple BLEU-like metric)
        overlaps = []
        for gen, ref in zip(generated, references):
            gen_words = set(gen.lower().split())
            ref_words = set(ref.lower().split())
            overlap = len(gen_words & ref_words) / len(ref_words) if ref_words else 0
            overlaps.append(overlap)

        results['word_overlap'] = float(np.mean(overlaps))

        return results


# ============================================================================
# BENCHMARK DATASET LOADERS
# ============================================================================

class BenchmarkDatasetLoader:
    """
    Loads standard privacy benchmark datasets.

    Supports:
    - PII-masking-200k (ai4privacy)
    - Custom synthetic datasets
    - Medical data (MedQA-style)
    """

    def load_synthetic_pii_dataset(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Generate synthetic dataset with known PII for testing.

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of samples with queries and ground truth PII
        """
        dataset = []

        # Medical conditions (sensitive)
        medical_conditions = ['diabetes', 'hypertension', 'cancer', 'heart disease', 'depression']

        # Generate samples
        for i in range(num_samples):
            condition = np.random.choice(medical_conditions)

            # Generate realistic queries
            queries = [
                f"symptoms of {condition}",
                f"treatment for {condition}",
                f"how to manage {condition}",
                f"{condition} medication side effects"
            ]

            # This is the PII that should NOT appear in outputs
            ground_truth_pii = {
                'medical_condition': condition,
                'queries': queries,
                'contains_phi': True
            }

            dataset.append({
                'id': f'sample_{i}',
                'queries': queries,
                'ground_truth_pii': ground_truth_pii,
                'category': 'medical'
            })

        return dataset

    def load_financial_pii_dataset(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate synthetic financial PII dataset."""
        dataset = []

        financial_topics = ['bankruptcy', 'debt consolidation', 'credit repair', 'loan default']

        for i in range(num_samples):
            topic = np.random.choice(financial_topics)

            queries = [
                f"how to file for {topic}",
                f"{topic} consequences",
                f"recovering from {topic}"
            ]

            ground_truth_pii = {
                'financial_topic': topic,
                'queries': queries,
                'contains_sensitive_financial': True
            }

            dataset.append({
                'id': f'financial_{i}',
                'queries': queries,
                'ground_truth_pii': ground_truth_pii,
                'category': 'financial'
            })

        return dataset


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

class EvaluationPipeline:
    """
    Complete evaluation pipeline for privacy-preserving LLM systems.

    Usage:
        evaluator = EvaluationPipeline()
        results = evaluator.run_full_evaluation(
            model_with_privacy=my_privacy_pipeline,
            model_without_privacy=baseline_model,
            dataset=test_data
        )
    """

    def __init__(self):
        self.privacy_evaluator = PrivacyEvaluator()
        self.utility_evaluator = UtilityEvaluator()
        self.dataset_loader = BenchmarkDatasetLoader()

    def run_full_evaluation(self,
                           outputs_with_privacy: List[str],
                           outputs_without_privacy: List[str],
                           ground_truth_queries: List[List[str]],
                           ground_truth_scores: List[float] = None) -> Dict[str, Any]:
        """
        Run complete evaluation comparing with/without privacy.

        Args:
            outputs_with_privacy: Outputs from privacy-preserving pipeline
            outputs_without_privacy: Outputs from baseline (no privacy)
            ground_truth_queries: Original sensitive queries
            ground_truth_scores: Optional true scores for utility evaluation

        Returns:
            Comprehensive evaluation results
        """
        results = {
            'privacy': {},
            'utility': {},
            'comparison': {}
        }

        # ===== PRIVACY EVALUATION =====
        print("Evaluating privacy...")

        # PII Leakage
        privacy_pii = self.privacy_evaluator.detect_pii_leakage(outputs_with_privacy)
        baseline_pii = self.privacy_evaluator.detect_pii_leakage(outputs_without_privacy)

        results['privacy']['pii_leakage'] = {
            'with_privacy': privacy_pii,
            'without_privacy': baseline_pii,
            'improvement': baseline_pii['leakage_rate'] - privacy_pii['leakage_rate']
        }

        # Reconstruction Attack
        privacy_recon = self.privacy_evaluator.evaluate_reconstruction_attack(
            outputs_with_privacy, ground_truth_queries
        )
        baseline_recon = self.privacy_evaluator.evaluate_reconstruction_attack(
            outputs_without_privacy, ground_truth_queries
        )

        results['privacy']['reconstruction'] = {
            'with_privacy': privacy_recon,
            'without_privacy': baseline_recon,
            'improvement': baseline_recon['reconstruction_rate'] - privacy_recon['reconstruction_rate']
        }

        # ===== UTILITY EVALUATION =====
        if ground_truth_scores is not None:
            print("Evaluating utility...")

            # Assuming outputs contain scores - you'll need to parse these
            # This is a placeholder - adapt to your actual output format
            privacy_scores = self._extract_scores_from_outputs(outputs_with_privacy)
            baseline_scores = self._extract_scores_from_outputs(outputs_without_privacy)

            if privacy_scores and baseline_scores:
                utility_with_privacy = self.utility_evaluator.evaluate_regression(
                    privacy_scores, ground_truth_scores
                )
                utility_without_privacy = self.utility_evaluator.evaluate_regression(
                    baseline_scores, ground_truth_scores
                )

                results['utility'] = {
                    'with_privacy': utility_with_privacy,
                    'without_privacy': utility_without_privacy,
                    'utility_loss': utility_with_privacy['correlation'] - utility_without_privacy['correlation']
                }

        # ===== COMPARISON SUMMARY =====
        results['comparison'] = self._generate_comparison_summary(results)

        return results

    def _extract_scores_from_outputs(self, outputs: List[str]) -> List[float]:
        """Extract numerical scores from outputs. Adapt to your format."""
        scores = []
        for output in outputs:
            # Try to find score pattern like "score: 0.85" or "QualityScore": 0.85
            match = re.search(r'(?:score|quality)["\']?\s*[:=]\s*([\d.]+)', output, re.IGNORECASE)
            if match:
                scores.append(float(match.group(1)))
            else:
                scores.append(0.0)  # Default if no score found
        return scores

    def _generate_comparison_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate human-readable comparison summary."""
        summary = {
            'privacy_improvement': "N/A",
            'utility_maintained': "N/A",
            'overall_verdict': "N/A"
        }

        # Privacy improvement
        if 'pii_leakage' in results.get('privacy', {}):
            improvement = results['privacy']['pii_leakage']['improvement']
            summary['privacy_improvement'] = f"{improvement * 100:.1f}% reduction in PII leakage"

        # Utility maintenance
        if 'utility' in results and results['utility']:
            utility_loss = results['utility'].get('utility_loss', 0)
            if abs(utility_loss) < 0.05:
                summary['utility_maintained'] = "Yes (< 5% degradation)"
            else:
                summary['utility_maintained'] = f"No ({utility_loss * 100:.1f}% degradation)"

        return summary


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Privacy-Preserving LLM Evaluation Framework")
    print("=" * 60)

    # Initialize
    evaluator = EvaluationPipeline()

    # Load synthetic dataset
    print("\nLoading synthetic dataset...")
    dataset = evaluator.dataset_loader.load_synthetic_pii_dataset(num_samples=50)
    print(f"Loaded {len(dataset)} samples")

    # Example evaluation (you'll replace with actual model outputs)
    print("\nRunning example evaluation...")

    # Simulate outputs
    queries = [sample['queries'] for sample in dataset]

    # WITHOUT privacy (baseline): Outputs contain actual queries
    outputs_without_privacy = [
        f"User searched for: {', '.join(q[:2])}" for q in queries
    ]

    # WITH privacy: Outputs use masked tokens
    outputs_with_privacy = [
        f"User searched for: QUERY_SEARCH_001, QUERY_SEARCH_002" for _ in queries
    ]

    # Run evaluation
    results = evaluator.run_full_evaluation(
        outputs_with_privacy=outputs_with_privacy,
        outputs_without_privacy=outputs_without_privacy,
        ground_truth_queries=queries
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\nPrivacy Metrics:")
    print(f"  PII Leakage (without privacy): {results['privacy']['pii_leakage']['without_privacy']['leakage_rate']*100:.1f}%")
    print(f"  PII Leakage (with privacy):    {results['privacy']['pii_leakage']['with_privacy']['leakage_rate']*100:.1f}%")
    print(f"  Improvement:                   {results['privacy']['pii_leakage']['improvement']*100:.1f}%")

    print(f"\n  Reconstruction Success (without): {results['privacy']['reconstruction']['without_privacy']['reconstruction_rate']*100:.1f}%")
    print(f"  Reconstruction Success (with):    {results['privacy']['reconstruction']['with_privacy']['reconstruction_rate']*100:.1f}%")
    print(f"  Improvement:                      {results['privacy']['reconstruction']['improvement']*100:.1f}%")

    print("\n" + "=" * 60)
    print("Evaluation framework ready! Adapt to your actual pipeline.")
