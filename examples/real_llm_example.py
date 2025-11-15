"""
Production-Ready Ensemble Pipeline with Real LLM APIs

This shows how to adapt the example to use real LLMs:
- OpenAI (GPT-4)
- Anthropic (Claude)
- Google (Gemini)
- Open-source models via vLLM

Replace MockLLMEvaluator with RealLLMEvaluator.
"""

import json
import os
from typing import Dict, List
import asyncio

# For real LLM calls
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from sambanova import SambaNova
    HAS_SAMBANOVA = True
except ImportError:
    HAS_SAMBANOVA = False


# ============================================================================
# SYSTEM PROMPT (From Your Appendix A)
# ============================================================================

SYSTEM_PROMPT = """You are an Interest Evaluator. Output ONLY one valid JSON array.
Each element must have exactly:
  - ItemId
  - QualityScore (0–1 float)
  - QualityReason (<=150 chars)

Forbidden keys: error, invalid, status.

Rules:
1. Data Sources: browsing_history, raw_queries, demographics.
   Distinct sources = different source types.

2. Noise Filtering:
   Ignore all navigation / utility queries (youtube, login, homepage, bare domains, translators, URL-only).

3. Persona (INTERNAL ONLY — DO NOT OUTPUT):
   Construct internal persona summary:
     - CoreThemes
     - SecondaryThemes
     - BehavioralPatterns
     - RecencyHints
     - DemographicFactors (penalize mismatches)
     - Uncertainties
   Use persona ONLY for consistent scoring.

4. Scoring Tiers (not to output):
   <0.35  = no evidence
   0.35–0.55 = weak hint
   0.55–0.70 = strong one-source evidence
   0.70–0.82 = strong two-source evidence
   >=0.82   = very strong multi-source evidence
   Demographic mismatch caps score ≤0.20.

5. Reason Format:
   Must follow:
     "Strong:browsing_history+raw_queries"
     "Weak:raw_queries"
     "None:no supporting evidence"
   NEVER invent titles or queries.

6. Hallucination Guard:
   Use ONLY provided masked data. If uncertain → lower score.

7. Output format:
   [
     {"ItemId":"A","QualityScore":0.61,"QualityReason":"browsing_history+raw_queries"},
     {"ItemId":"B","QualityScore":0.20,"QualityReason":"gender mismatch"}
   ]

8. After closing bracket STOP.
"""


def create_user_prompt(masked_user_data: Dict, candidate_topics: List[Dict]) -> str:
    """Create user prompt from masked data and candidates."""

    # Format candidate topics as JSONLINES
    candidate_lines = "\n".join([
        json.dumps({"ItemId": t["ItemId"], "Topic": t["Topic"],
                   "Demographics": t.get("Demographics", {})})
        for t in candidate_topics
    ])

    user_prompt = f"""Evaluate interest for each candidate's topic.

User Data (masked):
{json.dumps(masked_user_data, indent=2)}

Candidate Topics:
{candidate_lines}

Instructions:
- Aggregate evidence by distinct source types.
- Apply scoring tiers and demographic penalties.
- If no evidence: score <0.35, reason="no supporting evidence".
- Output ONLY a JSON array.
"""

    return user_prompt


# ============================================================================
# REAL LLM EVALUATOR
# ============================================================================

class RealLLMEvaluator:
    """
    Production evaluator using real LLM APIs.
    """

    def __init__(self, model_name: str, api_key: str = None, base_url: str = None):
        """
        Args:
            model_name: "gpt-4", "claude-3-5-sonnet-20241022", "gemini-pro", "Meta-Llama-3.1-8B-Instruct", etc.
            api_key: API key (or set via environment variable)
            base_url: Base URL for API (optional, for custom API endpoints)
        """
        import sys
        print(f"[RealLLMEvaluator] Initializing for model: {model_name}", flush=True)

        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url

        print(f"[RealLLMEvaluator] API key length: {len(self.api_key) if self.api_key else 0}", flush=True)

        # Determine provider
        # Check for custom LLM API models first
        if any(keyword in model_name.lower() for keyword in ["gpt-oss", "deepseek", "qwen", "llama", "mistral"]):
            # Custom LLM API (OpenAI-compatible endpoint)
            print(f"[RealLLMEvaluator] Provider: custom_llm_api", flush=True)

            self.provider = "sambanova"
            if not HAS_SAMBANOVA:
                raise ImportError("Install sambanova: pip install sambanova")

            # Use OpenAI-compatible API client
            api_base = base_url or os.getenv("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1")
            print(f"[RealLLMEvaluator] Creating API client with base_url: {api_base}", flush=True)

            self.client = SambaNova(
                api_key=self.api_key,
                base_url=api_base
            )
            print(f"[RealLLMEvaluator] ✓ API client created successfully", flush=True)

        elif "gpt" in model_name.lower():
            self.provider = "openai"
            if not HAS_OPENAI:
                raise ImportError("Install openai: pip install openai")
            self.client = openai.OpenAI(api_key=self.api_key)

        elif "claude" in model_name.lower():
            self.provider = "anthropic"
            if not HAS_ANTHROPIC:
                raise ImportError("Install anthropic: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=self.api_key)

        elif "gemini" in model_name.lower():
            self.provider = "google"
            # Would need google-generativeai package
            raise NotImplementedError("Gemini support not yet implemented")

        else:
            # Assume vLLM or other OpenAI-compatible endpoint
            self.provider = "openai_compatible"
            self.client = openai.OpenAI(
                api_key=self.api_key or "EMPTY",
                base_url=base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            )

    def _get_api_key(self) -> str:
        """Get API key from environment."""
        if "gpt" in self.model_name.lower():
            return os.getenv("OPENAI_API_KEY")
        elif "claude" in self.model_name.lower():
            return os.getenv("ANTHROPIC_API_KEY")
        elif "gemini" in self.model_name.lower():
            return os.getenv("GOOGLE_API_KEY")
        elif any(keyword in self.model_name.lower() for keyword in ["gpt-oss", "deepseek", "qwen", "llama", "mistral"]):
            return os.getenv("LLM_API_KEY")
        return None

    def evaluate_interest(self, masked_user_data: Dict,
                         candidate_topics: List[Dict]) -> List[Dict]:
        """
        Call real LLM API to evaluate interest.

        Returns: List[{"ItemId": str, "QualityScore": float, "QualityReason": str}]
        """
        user_prompt = create_user_prompt(masked_user_data, candidate_topics)

        if self.provider == "openai" or self.provider == "openai_compatible":
            return self._call_openai(user_prompt)
        elif self.provider == "sambanova":
            return self._call_sambanova(user_prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(user_prompt)
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")

    def _call_openai(self, user_prompt: str) -> List[Dict]:
        """Call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Low temperature for consistent scoring
                max_tokens=1000,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON array
            results = self._parse_json_response(content)
            return results

        except Exception as e:
            print(f"Error calling {self.model_name}: {e}")
            return []

    def _call_anthropic(self, user_prompt: str) -> List[Dict]:
        """Call Anthropic API."""
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0.3,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            content = message.content[0].text.strip()

            # Parse JSON array
            results = self._parse_json_response(content)
            return results

        except Exception as e:
            print(f"Error calling {self.model_name}: {e}")
            return []

    def _call_sambanova(self, user_prompt: str) -> List[Dict]:
        """Call Custom LLM API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent scoring
                top_p=0.1,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON array
            results = self._parse_json_response(content)
            return results

        except Exception as e:
            print(f"Error calling {self.model_name}: {e}")
            return []

    def _parse_json_response(self, content: str) -> List[Dict]:
        """Parse JSON array from LLM response."""
        # Find JSON array in response
        start = content.find('[')
        end = content.rfind(']') + 1

        if start == -1 or end == 0:
            print(f"No JSON array found in response: {content[:100]}")
            return []

        json_str = content[start:end]

        try:
            results = json.loads(json_str)

            # Validate format
            for r in results:
                assert "ItemId" in r, f"Missing ItemId in {r}"
                assert "QualityScore" in r, f"Missing QualityScore in {r}"
                assert "QualityReason" in r, f"Missing QualityReason in {r}"

                # Ensure score is float in [0, 1]
                r["QualityScore"] = float(r["QualityScore"])
                assert 0 <= r["QualityScore"] <= 1, f"Invalid score {r['QualityScore']}"

            return results

        except (json.JSONDecodeError, AssertionError) as e:
            print(f"Error parsing JSON: {e}")
            print(f"Content: {json_str[:200]}")
            return []


# ============================================================================
# ASYNC BATCH EVALUATION (For Speed)
# ============================================================================

class AsyncEnsembleEvaluator:
    """
    Run multiple LLMs in parallel for speed.
    """

    def __init__(self, model_configs: List[Dict]):
        """
        Args:
            model_configs: [
                {"model": "gpt-4", "api_key": "..."},
                {"model": "claude-3-5-sonnet-20241022", "api_key": "..."},
                ...
            ]
        """
        self.evaluators = [
            RealLLMEvaluator(cfg["model"], cfg.get("api_key"))
            for cfg in model_configs
        ]

    async def evaluate_all(self, masked_user_data: Dict,
                          candidate_topics: List[Dict]) -> List[List[Dict]]:
        """
        Evaluate with all models in parallel.

        Returns: List of results (one per model)
        """
        tasks = []

        for evaluator in self.evaluators:
            # Wrap sync call in async
            task = asyncio.to_thread(
                evaluator.evaluate_interest,
                masked_user_data,
                candidate_topics
            )
            tasks.append(task)

        # Run all in parallel
        all_results = await asyncio.gather(*tasks)
        return all_results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_with_real_llms():
    """
    Example showing how to use real LLMs in the pipeline.
    """

    # NOTE: This is a DEMO - you need to set API keys!
    print("="*80)
    print("EXAMPLE: Using Real LLMs in Ensemble Pipeline")
    print("="*80)

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Set OPENAI_API_KEY environment variable to run with GPT-4")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Set ANTHROPIC_API_KEY environment variable to run with Claude")

    print("\n" + "="*80)
    print("Configuration")
    print("="*80)

    # Configure ensemble models
    model_configs = [
        {"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")},
        {"model": "gpt-4-turbo", "api_key": os.getenv("OPENAI_API_KEY")},
        {"model": "claude-3-5-sonnet-20241022", "api_key": os.getenv("ANTHROPIC_API_KEY")},
        # Add more models as needed
    ]

    # Filter out models without API keys
    valid_configs = [cfg for cfg in model_configs if cfg["api_key"]]

    if not valid_configs:
        print("\n❌ No valid API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("\nTo run this example:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        print("  python ensemble_with_real_llms.py")
        return

    print(f"\n✓ Found {len(valid_configs)} valid model configurations:")
    for cfg in valid_configs:
        print(f"  - {cfg['model']}")

    # Example masked data (from previous example)
    masked_user_data = {
        "MSNClicks": [
            {"token": "QUERY_MSN_001", "timestamp": "recent"},
            {"token": "QUERY_MSN_002", "timestamp": "recent"},
            {"token": "QUERY_MSN_003", "timestamp": "recent"}
        ],
        "BingSearch": [
            {"token": "QUERY_SEARCH_001", "timestamp": "recent"},
            {"token": "QUERY_SEARCH_002", "timestamp": "recent"}
        ],
        "MAI": {"Health": 10, "Fitness": 3, "Technology": 1},
        "demographics": {"age_range": "35-44", "gender": "F"}
    }

    candidate_topics = [
        {"ItemId": "A", "Topic": "Managing diabetes with healthy eating and exercise"},
        {"ItemId": "B", "Topic": "Latest advancements in artificial intelligence"},
        {"ItemId": "C", "Topic": "Women's health: wellness tips", "Demographics": {"gender": "F"}},
    ]

    print("\n" + "="*80)
    print("Evaluating with Real LLMs...")
    print("="*80)

    # Option 1: Sequential evaluation
    print("\nSequential evaluation (slower but simpler):")
    for cfg in valid_configs[:1]:  # Just first model for demo
        evaluator = RealLLMEvaluator(cfg["model"], cfg["api_key"])
        results = evaluator.evaluate_interest(masked_user_data, candidate_topics)

        print(f"\n{cfg['model']} Results:")
        print(json.dumps(results, indent=2))

    # Option 2: Parallel evaluation (faster)
    print("\n" + "="*80)
    print("Parallel evaluation (faster):")
    print("="*80)

    async def run_async():
        ensemble = AsyncEnsembleEvaluator(valid_configs)
        all_results = await ensemble.evaluate_all(masked_user_data, candidate_topics)

        for i, (cfg, results) in enumerate(zip(valid_configs, all_results)):
            print(f"\nModel {i+1} ({cfg['model']}):")
            print(json.dumps(results, indent=2))

        return all_results

    # Run async
    # all_results = asyncio.run(run_async())

    # Then aggregate with consensus (from previous example)
    # consensus = ConsensusAggregator().aggregate_median(all_results)

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. Set API keys for your LLM providers")
    print("2. Run full pipeline with real models")
    print("3. Compare consensus methods (median, intersection, trimmed mean)")
    print("4. Measure privacy metrics (PII leakage, variance reduction)")
    print("5. Deploy inside privacy boundary")


# ============================================================================
# COST ESTIMATION
# ============================================================================

def estimate_cost(num_users: int, candidates_per_user: int, num_models: int):
    """
    Estimate API costs for ensemble evaluation.
    """
    print("="*80)
    print("COST ESTIMATION")
    print("="*80)

    # Rough token estimates
    tokens_per_evaluation = 1500  # System prompt + user data + candidates + response

    # Pricing (as of 2024)
    pricing = {
        "gpt-4": 0.03 / 1000,  # $0.03 per 1K input tokens
        "gpt-4-turbo": 0.01 / 1000,
        "claude-3-5-sonnet-20241022": 0.003 / 1000,
        "gemini-pro": 0.00025 / 1000,
    }

    total_evaluations = num_users * num_models
    total_tokens = total_evaluations * tokens_per_evaluation

    print(f"\nScenario:")
    print(f"  - Users: {num_users:,}")
    print(f"  - Candidates per user: {candidates_per_user}")
    print(f"  - Models in ensemble: {num_models}")
    print(f"  - Total evaluations: {total_evaluations:,}")
    print(f"  - Total tokens: {total_tokens:,}")

    print(f"\nEstimated costs per model:")
    for model, price_per_token in pricing.items():
        cost = total_tokens * price_per_token
        print(f"  {model:30s}: ${cost:,.2f}")

    # Example ensemble
    ensemble_cost = (
        total_tokens * pricing["gpt-4"] +
        total_tokens * pricing["claude-3-5-sonnet-20241022"]
    ) / num_models

    print(f"\nExample ensemble (GPT-4 + Claude):")
    print(f"  Cost per user: ${ensemble_cost / num_users:.4f}")
    print(f"  Total cost: ${ensemble_cost * 2:.2f}")  # 2 models


if __name__ == "__main__":
    # Show example
    example_with_real_llms()

    print("\n")

    # Cost estimation
    estimate_cost(
        num_users=10000,
        candidates_per_user=5,
        num_models=5
    )
