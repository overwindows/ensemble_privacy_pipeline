#!/usr/bin/env python3
"""
Custom 4-Model Ensemble Privacy Pipeline

This script uses the following models:
1. gpt-oss-120b
2. DeepSeek-V3.1
3. Qwen3-32B
4. DeepSeek-V3-0324

Usage:
  python3 run_demo_pipeline.py
"""

import json
import os
import sys
import re
import requests
from datetime import datetime
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.privacy_core import PrivacyRedactor, ConsensusAggregator
from examples.real_llm_example import RealLLMEvaluator


# ============================================================================
# 4-MODEL ENSEMBLE CONFIGURATION
# ============================================================================

ENSEMBLE_MODELS = [
    "gpt-oss-120b",       # 120B parameter open-source model
    "DeepSeek-V3.1",      # Advanced reasoning model
    "Qwen3-32B",          # 32B parameter model
    "DeepSeek-V3-0324",   # Latest DeepSeek variant
]


# ============================================================================
# BASELINE EVALUATOR (NO PRIVACY PROTECTION) - UNSAFE PROMPT
# ============================================================================

UNSAFE_SYSTEM_PROMPT = """You are an Interest Evaluator. Analyze user data and evaluate topics.

Output JSON array with:
- ItemId
- QualityScore (0-1)
- QualityReason (explain why, you can reference specific searches, articles, or behaviors)

Be detailed in your reasoning - cite specific evidence from the user data."""

class BaselineUnsafeEvaluator:
    """
    Baseline that uses UNSAFE prompts - allows LLM to cite specific data.
    This demonstrates what happens WITHOUT privacy-preserving prompts.
    """
    
    def __init__(self, model_name: str, api_key: str, base_url: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url or "https://api.sambanova.ai/v1"
    
    def evaluate_interest(self, raw_user_data: Dict, candidate_topics: List[Dict]) -> List[Dict]:
        """Call LLM with UNSAFE prompt that allows citing specific data."""
        # Create prompt that encourages detailed responses
        topics_str = "\n".join([f"{t['ItemId']}. {t['Topic']}" for t in candidate_topics])
        
        user_prompt = f"""Analyze this user's interests and evaluate the following topics.
        
User Data:
{json.dumps(raw_user_data, indent=2)}

Topics to evaluate:
{topics_str}

For each topic, provide a QualityScore (0-1) and explain your reasoning with specific evidence from the user data.
Output as JSON array."""

        # Call SambaNova API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": UNSAFE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                results = json.loads(json_match.group(0))
                return results
            else:
                raise ValueError("No JSON array found in response")
                
        except Exception as e:
            print(f"Error: {e}")
            # Fallback to simulated
            return self._simulate_unsafe_output(raw_user_data, candidate_topics)
    
    def _simulate_unsafe_output(self, raw_data: Dict, topics: List[Dict]) -> List[Dict]:
        """Simulate what an unsafe LLM might output (includes specific data)."""
        results = []
        for topic in topics:
            score, reason = self._score_with_specific_data(topic["Topic"], raw_data)
            results.append({
                "ItemId": topic["ItemId"],
                "QualityScore": score,
                "QualityReason": reason
            })
        return results
    
    def _score_with_specific_data(self, topic: str, raw_data: Dict):
        """Score topic and include specific queries/titles in reason (UNSAFE!)."""
        topic_lower = topic.lower()
        evidence = []
        
        # Include specific search queries in reason
        if "BingSearch" in raw_data:
            for search in raw_data["BingSearch"]:
                query = search["query"]
                if any(word in query.lower() for word in topic_lower.split()):
                    evidence.append(f'searched for "{query}"')
        
        # Include specific article titles in reason
        if "MSNClicks" in raw_data:
            for click in raw_data["MSNClicks"]:
                title = click["title"]
                if any(word in title.lower() for word in topic_lower.split()):
                    evidence.append(f'read "{title[:60]}..."')
        
        if evidence:
            reason = f"User {' and '.join(evidence[:3])}"  # Include up to 3 examples
            score = min(0.9, 0.6 + len(evidence) * 0.1)
        else:
            reason = "No matching behavior found"
            score = 0.25
        
        return score, reason


class BaselineEvaluator:
    """
    Baseline approach without privacy protection.
    This directly uses raw user data - demonstrating what happens WITHOUT the pipeline.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def evaluate_interest(self, raw_user_data: Dict, candidate_topics: List[Dict]) -> List[Dict]:
        """
        Evaluate topics using RAW data (UNSAFE - for comparison only).
        This simulates a typical system without privacy protection.
        """
        results = []
        
        for topic in candidate_topics:
            score, reason, leaked_data = self._score_topic(
                topic["Topic"],
                raw_user_data
            )
            
            result = {
                "ItemId": topic["ItemId"],
                "QualityScore": score,
                "QualityReason": reason,
            }
            
            # Include leaked data to show what gets exposed
            if leaked_data:
                result["LEAKED_DATA"] = leaked_data
            
            results.append(result)
        
        return results
    
    def _score_topic(self, topic: str, raw_data: Dict):
        """Score topic and track what data gets leaked."""
        leaked_data = {}
        evidence_count = 0
        topic_lower = topic.lower()
        
        # Check search queries - exposes actual searches
        if "BingSearch" in raw_data:
            matching = []
            for search in raw_data["BingSearch"]:
                query = search["query"]
                if any(word in query.lower() for word in topic_lower.split()):
                    matching.append(query)
                    evidence_count += 1
            if matching:
                leaked_data["queries"] = matching
        
        # Check article clicks - exposes reading history
        if "MSNClicks" in raw_data:
            matching = []
            for click in raw_data["MSNClicks"]:
                title = click["title"]
                if any(word in title.lower() for word in topic_lower.split()):
                    matching.append(title)
                    evidence_count += 1
            if matching:
                leaked_data["titles"] = matching
        
        # Score based on evidence
        if evidence_count == 0:
            score = 0.25
            reason = "No matching behavior"
        elif evidence_count <= 2:
            score = 0.60
            reason = "Some matching behavior"
        elif evidence_count <= 4:
            score = 0.75
            reason = "Multiple matches"
        else:
            score = 0.85
            reason = "Strong match"
        
        return score, reason, leaked_data


# ============================================================================
# PRIVACY ANALYSIS UTILITIES
# ============================================================================

def analyze_leakage(results: List[Dict], pii_keywords: List[str], raw_data: Dict = None) -> Dict:
    """Analyze what private information is leaked in results."""
    leakage = {
        "queries_leaked": 0,
        "titles_leaked": 0,
        "pii_keywords_found": [],
        "has_leaked_data_field": False,
        "leaked_queries": [],
        "leaked_titles": [],
    }
    
    results_str = json.dumps(results).lower()
    
    # Check for explicit leaked data fields (from simulated baseline)
    for result in results:
        if "LEAKED_DATA" in result:
            leakage["has_leaked_data_field"] = True
            leaked = result["LEAKED_DATA"]
            if "queries" in leaked:
                leakage["queries_leaked"] += len(leaked["queries"])
                leakage["leaked_queries"].extend(leaked["queries"])
            if "titles" in leaked:
                leakage["titles_leaked"] += len(leaked["titles"])
                leakage["leaked_titles"].extend(leaked["titles"])
    
    # If raw data provided, check if actual queries/titles appear in output
    if raw_data and not leakage["has_leaked_data_field"]:
        if "BingSearch" in raw_data:
            for search in raw_data["BingSearch"]:
                query = search["query"]
                # Check if this specific query appears in the output
                if query.lower() in results_str:
                    leakage["queries_leaked"] += 1
                    leakage["leaked_queries"].append(query)
        
        if "MSNClicks" in raw_data:
            for click in raw_data["MSNClicks"]:
                title = click["title"]
                # Check if this specific title appears in the output
                if title.lower() in results_str:
                    leakage["titles_leaked"] += 1
                    leakage["leaked_titles"].append(title)
    
    # Check for PII keywords in output
    for keyword in pii_keywords:
        if keyword.lower() in results_str:
            leakage["pii_keywords_found"].append(keyword)
    
    return leakage


def run_pipeline(raw_user_data: dict, candidate_topics: list, api_key: str, pii_keywords: list = None):
    """
    Run the complete privacy-preserving pipeline with your 4-model ensemble.
    Also runs a baseline (no protection) for comparison.

    Args:
        raw_user_data: Raw sensitive user data (dict)
        candidate_topics: List of topics to evaluate
        api_key: API key
        pii_keywords: List of sensitive keywords to track (optional)

    Returns:
        dict: Privacy-safe results with consensus scores and comparison metrics
    """

    print("=" * 80)
    print("ENSEMBLE-REDACTION PRIVACY PIPELINE DEMO")
    print("Side-by-Side Comparison: With vs Without Protection")
    print("=" * 80)
    
    if pii_keywords is None:
        pii_keywords = []

    # ========================================================================
    # TASK DESCRIPTION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TASK DESCRIPTION")
    print("=" * 80)
    
    print("""
🎯 TASK: Evaluate topic relevance for ONE user
⚠️  CHALLENGE: Do this WITHOUT leaking the user's sensitive personal data
""")

    # ========================================================================
    # INPUT DATA OVERVIEW
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("INPUT DATA")
    print("=" * 80)
    
    print("\n📋 RAW USER DATA:")
    print(json.dumps(raw_user_data, indent=2))
    
    print("\n🎯 CANDIDATE TOPICS:")
    for topic in candidate_topics:
        print(f"   {topic['ItemId']}. {topic['Topic']}")

    # ========================================================================
    # BASELINE: WITHOUT PRIVACY PROTECTION (ACTUAL LLM)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("BASELINE: WITHOUT PRIVACY PROTECTION")
    print("=" * 80)
    
    print("\n⚙️  Running baseline: RAW data + UNSAFE prompt + Single model")
    baseline_model = ENSEMBLE_MODELS[0]
    print(f"   Model: {baseline_model}")
    
    try:
        baseline_evaluator = BaselineUnsafeEvaluator(
            model_name=baseline_model,
            api_key=api_key
        )
        baseline_results = baseline_evaluator.evaluate_interest(raw_user_data, candidate_topics)
        print(f"   ✓ Completed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        baseline_evaluator = BaselineUnsafeEvaluator(model_name="Simulated", api_key="dummy")
        baseline_results = baseline_evaluator._simulate_unsafe_output(raw_user_data, candidate_topics)
    
    print("\n📤 BASELINE OUTPUT:")
    print(json.dumps(baseline_results, indent=2))
    
    # Analyze baseline leakage
    baseline_leakage = analyze_leakage(baseline_results, pii_keywords, raw_data=raw_user_data)

    # ========================================================================
    # PROTECTED PIPELINE: WITH ENSEMBLE-REDACTION
    # ========================================================================

    print("\n" + "=" * 80)
    print("PROTECTED PIPELINE: ENSEMBLE-REDACTION")
    print("=" * 80)

    # ========================================================================
    # STEP 1: REDACTION & MASKING
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 1: REDACTION & MASKING")
    print("=" * 80)

    redactor = PrivacyRedactor()
    masked_data = redactor.redact_user_data(raw_user_data)

    print("\n📝 DATA TRANSFORMATION:")
    
    # Show before/after for searches
    if "BingSearch" in raw_user_data and raw_user_data["BingSearch"]:
        print("\n🔍 Queries:")
        for i, search in enumerate(raw_user_data['BingSearch'][:3], 1):
            print(f"   Before: \"{search['query']}\"")
            print(f"   After:  QUERY_SEARCH_{i:03d}")
    
    # Show before/after for articles
    if "MSNClicks" in raw_user_data and raw_user_data["MSNClicks"]:
        print("\n📰 Articles:")
        for i, click in enumerate(raw_user_data['MSNClicks'][:3], 1):
            print(f"   Before: \"{click['title'][:50]}...\"")
            print(f"   After:  QUERY_MSN_{i:03d}")
    
    # Show demographics masking
    if "demographics" in raw_user_data and "age" in raw_user_data["demographics"]:
        age = raw_user_data["demographics"]["age"]
        age_range = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
        print(f"\n👤 Demographics:")
        print(f"   Before: Age {age}")
        print(f"   After:  Age {age_range}")
    
    print("\n📤 MASKED DATA:")
    print(json.dumps(masked_data, indent=2))

    # ========================================================================
    # STEP 2: ENSEMBLE EVALUATION (4 Models)
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 2: ENSEMBLE EVALUATION")
    print("=" * 80)

    print(f"\n⚙️  Calling {len(ENSEMBLE_MODELS)} models with MASKED data + SAFE prompts...")

    all_results = []
    model_timings = []

    for i, model_name in enumerate(ENSEMBLE_MODELS, 1):
        print(f"\n   Model {i}: {model_name}")

        try:
            start_time = datetime.now()
            evaluator = RealLLMEvaluator(model_name=model_name, api_key=api_key)
            results = evaluator.evaluate_interest(masked_data, candidate_topics)
            elapsed = (datetime.now() - start_time).total_seconds()
            model_timings.append(elapsed)

            print(f"   Output:")
            print(json.dumps(results, indent=6))

            all_results.append(results)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_results.append([
                {"ItemId": t["ItemId"], "QualityScore": 0.5, "QualityReason": f"error:{model_name}"}
                for t in candidate_topics
            ])

    # ========================================================================
    # STEP 3: CONSENSUS AGGREGATION
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 3: CONSENSUS AGGREGATION")
    print("=" * 80)

    aggregator = ConsensusAggregator()
    consensus_median = aggregator.aggregate_median(all_results)
    consensus_intersection = aggregator.aggregate_intersection(all_results)

    print("\n📤 CONSENSUS OUTPUT (Median):")
    print(json.dumps(consensus_median, indent=2))

    # Analyze protected pipeline leakage
    protected_leakage = analyze_leakage(consensus_median, pii_keywords, raw_data=raw_user_data)

    # ========================================================================
    # COMPARISON
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    baseline_status = "LEAKED" if (baseline_leakage['queries_leaked'] > 0 or baseline_leakage['titles_leaked'] > 0) else "SAFE"
    protected_status = "SAFE" if (protected_leakage['queries_leaked'] == 0 and protected_leakage['titles_leaked'] == 0) else "LEAKED"
    
    print(f"""
╔════════════════════════════════╦═══════════════╦═══════════════╗
║ Metric                         ║ Baseline      ║ Protected     ║
╠════════════════════════════════╬═══════════════╬═══════════════╣
║ Queries Leaked                 ║ {baseline_leakage['queries_leaked']:^13} ║ {protected_leakage['queries_leaked']:^13} ║
║ Titles Leaked                  ║ {baseline_leakage['titles_leaked']:^13} ║ {protected_leakage['titles_leaked']:^13} ║
║ Status                         ║ {baseline_status:^13} ║ {protected_status:^13} ║
╚════════════════════════════════╩═══════════════╩═══════════════╝
""")

    return {
        "baseline_results": baseline_results,
        "baseline_leakage": baseline_leakage,
        "consensus_median": consensus_median,
        "consensus_intersection": consensus_intersection,
        "protected_leakage": protected_leakage,
        "model_timings": model_timings,
        "total_time": sum(model_timings),
    }


def main():
    """Main execution."""

    # Check for API key
    api_key = os.getenv("SAMBANOVA_API_KEY")
    if not api_key:
        print("❌ Error: SAMBANOVA_API_KEY not set!")
        print("\nSet your API key:")
        print("  export SAMBANOVA_API_KEY='your-key-here'")
        sys.exit(1)

    print(f"✓ API Key found: {api_key[:20]}...\n")

    # ========================================================================
    # EXAMPLE DATA
    # ========================================================================

    # Example: User with diabetes (sensitive medical data)
    raw_user_data = {
        "MSNClicks": [
            {
                "title": "New diabetes treatment shows promise in clinical trials",
                "timestamp": "2024-01-15T10:30:00"
            },
            {
                "title": "Understanding type 2 diabetes: symptoms and prevention",
                "timestamp": "2024-01-14T15:20:00"
            },
            {
                "title": "Best fitness trackers for monitoring blood sugar levels",
                "timestamp": "2024-01-13T09:15:00"
            },
            {
                "title": "Depression and chronic illness: Finding support",
                "timestamp": "2024-01-12T14:20:00"
            }
        ],
        "BingSearch": [
            {
                "query": "diabetes diet plan",
                "timestamp": "2024-01-15T11:00:00"
            },
            {
                "query": "how to lower blood sugar naturally",
                "timestamp": "2024-01-14T16:30:00"
            },
            {
                "query": "diabetes medication side effects",
                "timestamp": "2024-01-13T10:30:00"
            }
        ],
        "BingClickedQueries": [
            {
                "query": "continuous glucose monitoring devices",
                "url": "https://www.healthline.com/diabetes-cgm"
            }
        ],
        "MSNUpvotes": [
            {
                "title": "Living well with diabetes: expert advice and tips"
            }
        ],
        "MAI": ["Health"] * 10 + ["Fitness"] * 3 + ["Technology"] * 1,
        "demographics": {
            "age": 42,
            "gender": "F",
            "location": "Seattle, WA"
        }
    }

    candidate_topics = [
        {
            "ItemId": "A",
            "Topic": "Managing diabetes with healthy eating and exercise"
        },
        {
            "ItemId": "B",
            "Topic": "Latest advancements in artificial intelligence"
        },
        {
            "ItemId": "C",
            "Topic": "Women's health: wellness tips for busy professionals",
            "Demographics": {"gender": "F"}
        },
        {
            "ItemId": "D",
            "Topic": "Men's grooming and style trends"
        },
        {
            "ItemId": "E",
            "Topic": "Fitness tracking apps and wearable technology"
        }
    ]

    # Track sensitive keywords that should NOT appear in output
    pii_keywords = [
        "diabetes",
        "depression",
        "blood sugar",
        "medication",
        "Seattle",
        "42",  # Age
    ]

    # Run pipeline with baseline comparison
    results = run_pipeline(raw_user_data, candidate_topics, api_key, pii_keywords)

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
