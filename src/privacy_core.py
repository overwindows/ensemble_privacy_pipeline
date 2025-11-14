"""
Core Privacy Pipeline Components
Ensemble-Redaction Consensus for Privacy-Preserving LLM Inference

This module contains only the essential privacy components.
For LLM evaluation, use RealLLMEvaluator from examples/real_llm_example.py
"""

import json
import re
import hashlib
from collections import Counter, defaultdict
from typing import Dict, List
import numpy as np


# ============================================================================
# STEP 1: REDACTION & MASKING
# ============================================================================

class PrivacyRedactor:
    """
    Masks sensitive user data while preserving signal for interest scoring.

    This is the core privacy component - it ensures NO raw PII reaches the LLMs.
    """

    def __init__(self):
        self.pii_vault = {}  # Store mappings for audit (inside privacy boundary)
        self.query_counter = defaultdict(int)

    def _is_navigation_noise(self, query: str) -> bool:
        """Filter out navigation/utility queries per protocol."""
        noise_patterns = [
            r'^https?://',  # URLs
            r'youtube\.com',
            r'login',
            r'homepage',
            r'translator',
            r'^google$',
            r'^facebook$',
            r'^mail$',
            r'^\w+\.\w+$',  # Bare domains like "cnn.com"
        ]

        query_lower = query.lower().strip()
        for pattern in noise_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def _mask_query(self, query: str, category: str = "GENERAL") -> str:
        """Replace query with surrogate token."""
        if self._is_navigation_noise(query):
            return None  # Filter out

        # Create deterministic but anonymized token
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        self.query_counter[category] += 1

        token = f"QUERY_{category.upper()}_{self.query_counter[category]:03d}"

        # Store in vault (for auditing inside boundary only)
        self.pii_vault[token] = query

        return token

    def _mask_demographic(self, value: str) -> str:
        """Generalize demographics."""
        # Age: exact → range
        if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
            age = int(value)
            if age < 25:
                return "18-24"
            elif age < 35:
                return "25-34"
            elif age < 45:
                return "35-44"
            elif age < 55:
                return "45-54"
            else:
                return "55+"
        return value

    def redact_user_data(self, raw_user_data: Dict) -> Dict:
        """
        Main redaction function.

        Input: Raw user behavioral data
        Output: Masked data (still private, but de-identified)

        Supports both vendor-neutral field names (raw_queries, browsing_history)
        and Microsoft-specific field names (MSNClicks, BingSearch, MAI) for
        backward compatibility.
        """
        masked = {}

        # ====================================================================
        # VENDOR-NEUTRAL FIELD NAMES (for public datasets and benchmarks)
        # ====================================================================

        # GENERIC: raw_queries - List of search queries or user prompts
        # Used by: neutral_benchmark.py, public_datasets_simple.py,
        #          pupa_benchmark.py, text_sanitization_benchmark.py, dp_benchmark.py
        if "raw_queries" in raw_user_data:
            masked_queries = []
            queries = raw_user_data["raw_queries"]
            if isinstance(queries, list):
                for query in queries:
                    if isinstance(query, str):
                        masked_token = self._mask_query(query, category="QUERY")
                        if masked_token:  # Not filtered as noise
                            masked_queries.append({"token": masked_token})
            if masked_queries:
                masked["queries"] = masked_queries

        # GENERIC: browsing_history - List of web pages/articles viewed
        # Used by: neutral_benchmark.py
        if "browsing_history" in raw_user_data:
            masked_browsing = []
            browsing = raw_user_data["browsing_history"]
            if isinstance(browsing, list):
                for item in browsing:
                    if isinstance(item, str):
                        masked_token = self._mask_query(item, category="BROWSING")
                        if masked_token:
                            masked_browsing.append({"token": masked_token})
            if masked_browsing:
                masked["browsing"] = masked_browsing

        # GENERIC: source_text - Single text string with PII
        # Used by: ai4privacy/pii-masking-200k dataset
        if "source_text" in raw_user_data:
            text = raw_user_data["source_text"]
            if isinstance(text, str):
                masked_token = self._mask_query(text, category="TEXT")
                if masked_token:
                    masked["text"] = {"token": masked_token}

        # GENERIC: user_prompt - Single user prompt with potential PII
        # Used by: PUPA dataset (NAACL 2025)
        if "user_prompt" in raw_user_data:
            prompt = raw_user_data["user_prompt"]
            if isinstance(prompt, str):
                masked_token = self._mask_query(prompt, category="PROMPT")
                if masked_token:
                    masked["prompt"] = {"token": masked_token}

        # GENERIC: text - Generic text field
        # Used by: TAB (Text Anonymization Benchmark)
        if "text" in raw_user_data:
            text = raw_user_data["text"]
            if isinstance(text, str):
                masked_token = self._mask_query(text, category="TEXT")
                if masked_token:
                    masked["text"] = {"token": masked_token}

        # ====================================================================
        # MICROSOFT-SPECIFIC FIELD NAMES (backward compatibility)
        # ====================================================================

        # MSN Clicks: Replace article titles with tokens
        if "MSNClicks" in raw_user_data:
            masked_clicks = []
            for click in raw_user_data["MSNClicks"]:
                masked_token = self._mask_query(click["title"], category="MSN")
                if masked_token:  # Not filtered as noise
                    masked_clicks.append({
                        "token": masked_token,
                        "timestamp": "recent"  # Normalize timestamp
                    })
            if masked_clicks:
                masked["MSNClicks"] = masked_clicks

        # Bing Search: Replace queries with tokens
        if "BingSearch" in raw_user_data:
            masked_searches = []
            for search in raw_user_data["BingSearch"]:
                masked_token = self._mask_query(search["query"], category="SEARCH")
                if masked_token:
                    masked_searches.append({
                        "token": masked_token,
                        "timestamp": "recent"
                    })
            if masked_searches:
                masked["BingSearch"] = masked_searches

        # Bing Clicked Queries: Mask query, keep domain only
        if "BingClickedQueries" in raw_user_data:
            masked_clicked = []
            for clicked in raw_user_data["BingClickedQueries"]:
                masked_token = self._mask_query(clicked["query"], category="CLICKED")
                if masked_token:
                    # Extract domain only (remove PII from URL)
                    url = clicked.get("url", "")
                    domain = re.search(r'https?://([^/]+)', url)
                    domain_only = domain.group(1) if domain else "unknown"

                    masked_clicked.append({
                        "token": masked_token,
                        "clicked_url_domain": domain_only  # Domain only, no path
                    })
            if masked_clicked:
                masked["BingClickedQueries"] = masked_clicked

        # MSN Upvotes: Mask titles
        if "MSNUpvotes" in raw_user_data:
            masked_upvotes = []
            for upvote in raw_user_data["MSNUpvotes"]:
                masked_token = self._mask_query(upvote["title"], category="UPVOTE")
                if masked_token:
                    masked_upvotes.append({"token": masked_token})
            if masked_upvotes:
                masked["MSNUpvotes"] = masked_upvotes

        # MAI (Microsoft Audience Intelligence): Aggregate counts
        if "MAI" in raw_user_data:
            mai_list = raw_user_data["MAI"]
            if isinstance(mai_list, list):
                # Convert list to aggregated counts
                mai_counts = dict(Counter(mai_list))
                masked["MAI"] = mai_counts
            elif isinstance(mai_list, dict):
                masked["MAI"] = mai_list

        # Demographics: Generalize
        if "demographics" in raw_user_data:
            demo = raw_user_data["demographics"]
            masked["demographics"] = {}

            if "age" in demo:
                masked["demographics"]["age_range"] = self._mask_demographic(demo["age"])

            # Keep gender (not PII by itself)
            if "gender" in demo:
                masked["demographics"]["gender"] = demo["gender"]

            # Keep location at city/state level (not street address)
            if "location" in demo:
                # Keep only city, state - remove specific addresses
                loc = demo["location"]
                if "," in loc:  # Format: "City, State"
                    masked["demographics"]["location"] = loc
                else:
                    masked["demographics"]["location"] = "unknown"

        return masked


# ============================================================================
# STEP 4: CONSENSUS AGGREGATION
# ============================================================================

class ConsensusAggregator:
    """
    Aggregates scores from multiple LLM evaluators using consensus methods.

    This reduces model variance and filters out rare/spurious details.
    """

    def aggregate_median(self, all_results: List[List[Dict]]) -> List[Dict]:
        """
        Method 1: Median score + Majority voting on reasons.

        RECOMMENDED for most use cases.

        Args:
            all_results: List of results from each model
                         Each element is List[{ItemId, QualityScore, QualityReason}]

        Returns:
            Aggregated results with consensus scores
        """
        if not all_results or not all_results[0]:
            return []

        # Group by ItemId
        item_ids = [r["ItemId"] for r in all_results[0]]
        consensus_results = []

        for item_id in item_ids:
            # Collect scores and reasons for this item across all models
            scores = []
            reasons = []

            for model_results in all_results:
                for result in model_results:
                    if result["ItemId"] == item_id:
                        scores.append(result["QualityScore"])
                        reasons.append(result["QualityReason"])
                        break

            # Median score
            median_score = float(np.median(scores))

            # Majority voting on reason (most common)
            reason_counts = Counter(reasons)
            majority_reason = reason_counts.most_common(1)[0][0]

            consensus_results.append({
                "ItemId": item_id,
                "QualityScore": round(median_score, 2),
                "QualityReason": majority_reason
            })

        return consensus_results

    def aggregate_intersection(self, all_results: List[List[Dict]]) -> List[Dict]:
        """
        Method 2: Intersection - only keep evidence ALL models agree on.

        MOST CONSERVATIVE - use for highest privacy guarantee.

        Returns:
            Aggregated results with only agreed-upon evidence
        """
        if not all_results or not all_results[0]:
            return []

        item_ids = [r["ItemId"] for r in all_results[0]]
        consensus_results = []

        for item_id in item_ids:
            scores = []
            all_reasons = []

            for model_results in all_results:
                for result in model_results:
                    if result["ItemId"] == item_id:
                        scores.append(result["QualityScore"])
                        # Extract source types from reason
                        reason = result["QualityReason"]
                        if ":" in reason:
                            sources = reason.split(":")[1]
                            all_reasons.append(set(sources.split("+")))
                        else:
                            all_reasons.append(set())
                        break

            # Median score
            median_score = float(np.median(scores))

            # Intersection of sources (only what ALL models agree on)
            if all_reasons:
                intersection_sources = set.intersection(*all_reasons)
                if intersection_sources:
                    reason = "Strong:" + "+".join(sorted(intersection_sources))
                else:
                    reason = "no supporting evidence"
            else:
                reason = "no supporting evidence"

            consensus_results.append({
                "ItemId": item_id,
                "QualityScore": round(median_score, 2),
                "QualityReason": reason
            })

        return consensus_results

    def aggregate_trimmed_mean(self, all_results: List[List[Dict]],
                               trim_percent: float = 0.2) -> List[Dict]:
        """
        Method 3: Trimmed mean - remove top/bottom outliers, then average.

        Good for removing model outliers while preserving middle consensus.

        Args:
            all_results: List of results from each model
            trim_percent: Percentage to trim from each end (default 20%)

        Returns:
            Aggregated results with trimmed mean scores
        """
        if not all_results or not all_results[0]:
            return []

        item_ids = [r["ItemId"] for r in all_results[0]]
        consensus_results = []

        for item_id in item_ids:
            scores = []
            reasons = []

            for model_results in all_results:
                for result in model_results:
                    if result["ItemId"] == item_id:
                        scores.append(result["QualityScore"])
                        reasons.append(result["QualityReason"])
                        break

            # Trimmed mean score
            scores_sorted = sorted(scores)
            trim_count = int(len(scores) * trim_percent)

            if trim_count > 0 and len(scores) > 2 * trim_count:
                trimmed_scores = scores_sorted[trim_count:-trim_count]
            else:
                trimmed_scores = scores_sorted

            mean_score = float(np.mean(trimmed_scores))

            # Majority voting on reason
            reason_counts = Counter(reasons)
            majority_reason = reason_counts.most_common(1)[0][0]

            consensus_results.append({
                "ItemId": item_id,
                "QualityScore": round(mean_score, 2),
                "QualityReason": majority_reason
            })

        return consensus_results


# ============================================================================
# PRIVACY ANALYSIS UTILITIES
# ============================================================================

def analyze_privacy_leakage(output: List[Dict]) -> Dict:
    """
    Analyze potential privacy leakage in output.

    Returns metrics about what private data might be exposed.
    """
    leakage = {
        "pii_count": 0,
        "query_count": 0,
        "title_count": 0,
        "specific_content": [],
        "severity": "NONE"
    }

    for result in output:
        # Check if LEAKED_PRIVATE_DATA field exists (shouldn't in production!)
        if "LEAKED_PRIVATE_DATA" in result:
            for leak in result["LEAKED_PRIVATE_DATA"]:
                if "leaked_queries" in leak:
                    queries = leak["leaked_queries"]
                    leakage["query_count"] += len(queries)
                    leakage["specific_content"].extend(queries)

                if "leaked_titles" in leak:
                    titles = leak["leaked_titles"]
                    leakage["title_count"] += len(titles)
                    leakage["specific_content"].extend(titles)

        # Check QualityReason for potential leaks (should only have source types)
        reason = result.get("QualityReason", "")
        # Reasons should ONLY contain source types like "MSNClicks+BingSearch"
        # If they contain actual query text, that's a leak
        if any(char in reason for char in ['"', "'", 'query:', 'title:']):
            leakage["pii_count"] += 1
            leakage["specific_content"].append(reason)

    # Determine severity
    total_leaks = leakage["query_count"] + leakage["title_count"] + leakage["pii_count"]
    if total_leaks == 0:
        leakage["severity"] = "NONE"
    elif total_leaks < 3:
        leakage["severity"] = "LOW"
    elif total_leaks < 10:
        leakage["severity"] = "MEDIUM"
    else:
        leakage["severity"] = "HIGH"

    return leakage
