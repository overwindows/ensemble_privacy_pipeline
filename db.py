"""Seed the showcase DB with initial content if empty."""
from sqlalchemy.orm import Session
from models import Page, Section, get_engine, init_db

RESULTS_TABLE = """| Benchmark | Samples | Privacy Protection | Task Performance |
|-----------|---------|--------------------|------------------|
| Vendor-Neutral Synthetic | 300 | 100% (300/300) | 85.0% accuracy (255/300) |
| Text Masking (ai4privacy) | 1,000 | 28.8% full protection (288/1000) | 4,012 PII entities tested |
| Question Answering (PUPA) | 901 | 81.2% protected (3904/4806) | 100% response success (vs PAPILLON 85.5%) |
| Document Sanitization (TAB) | 1,268 | 99.9% direct IDs (1267/1268) | PERSON/ORG/LOC 99.9%, CODE 18.9% |
| Differential Privacy Comparison | 100 | 0% canary exposure, 98.5% MIA resistance | Comparable to DP (epsilon=1.0) |"""

DP_TABLE = """| Aspect | This Approach | Differential Privacy |
|--------|--------------|----------------------|
| **Privacy Guarantee** | Empirical (benchmark-measured) | Formal (mathematical proof) |
| **Training Required** | No | Yes (DP-SGD) |
| **Utility** | Measured on benchmarks | Typically utility degradation |
| **Applicability** | Inference-time only | Training and/or inference |"""


def seed(engine, force=False):
    init_db(engine)
    with Session(engine) as session:
        if session.query(Page).count() > 0 and not force:
            return
        if session.query(Page).count() > 0 and force:
            for page in session.query(Page).all():
                session.delete(page)
            session.commit()

        overview = Page(
            slug="", title="Ensemble-Redaction Privacy Pipeline",
            nav_label="Overview",
            hero=(
                "A **training-free** privacy-preserving approach for LLM inference on sensitive "
                "user data. It combines an input-masking layer that redacts PII with ensemble "
                "consensus voting to reduce model variance **without any raw PII reaching the "
                "LLMs.**"
            ),
        )
        overview.sections = [
            Section(position=0, heading="How it works (3 steps)",
                    body="""1. **Mask** — `PrivacyRedactor` replaces sensitive queries with surrogate tokens (`QUERY_*`) before any model sees them. Age is generalized to ranges, demographics are blurred.
2. **Evaluate** — an ensemble of LLMs scores the masked data independently.
3. **Aggregate** — `ConsensusAggregator` merges results (median / trimmed mean / intersection) to cut individual model variance."""),
            Section(position=1, heading="Why it matters",
                    body="""Most privacy tooling is **training-time** and hurts utility. This approach is **inference-time and training-free** — compatible with any LLM API — and preserves task performance while measurably reducing PII exposure."""),
        ]

        results = Page(
            slug="results", title="Benchmarks & Results",
            nav_label="Results",
            hero="Measured against public privacy benchmarks. **Empirical** privacy protection with strong task performance.",
        )
        results.sections = [
            Section(position=0, heading="Benchmark summary", body=RESULTS_TABLE),
            Section(position=1, heading="Comparison vs Differential Privacy", body=DP_TABLE),
        ]

        roadmap = Page(
            slug="roadmap", title="Roadmap", nav_label="Roadmap",
            hero="Planned next steps for the pipeline.",
        )
        roadmap.sections = [
            Section(position=0, heading="Near term",
                    body="""- Expand benchmark coverage (more domains & sample sizes).
- Publish a walkthrough doc for the PUPA ensemble flow.
- Improve low-performing areas (e.g. CODE entity masking at 18.9%)."""),
            Section(position=1, heading="Later",
                    body="""- Compare against more baselines.
- Explore per-entity privacy thresholds.""", ),
        ]

        demo = Page(
            slug="demo", title="Live Demo", nav_label="Live Demo",
            hero="Feed in some example behavioral data and see the redaction layer mask it **before** inference.",
        )
        demo.sections = []

        brainstorm = Page(
            slug="brainstorm", title="Brainstorm", nav_label="Brainstorm",
            hero="Open thoughts, research directions, and open questions.",
        )
        brainstorm.sections = [
            Section(position=0, heading="Open questions", body="""- How do per-entity privacy thresholds interact with utility?
- Can lightweight private signals (e.g. category-level features) improve consensus quality?
- Where does empirical protection fall short of formal DP guarantees?"""),
        ]

        session.add_all([overview, results, roadmap, demo, brainstorm])
        session.commit()
