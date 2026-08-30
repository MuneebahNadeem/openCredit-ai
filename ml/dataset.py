"""
Dataset — converts InvestigationResult objects into a training-ready DataFrame.

There are two ways to get data into the training pipeline:

1. **Real data** (preferred, when Person 1's agent is running):
   Pass a list of ``(InvestigationResult, label)`` pairs to ``build_dataset()``.
   Labels are integers: 1 = trustworthy / approvable, 0 = not trustworthy.

2. **Synthetic data** (for development and testing):
   Call ``generate_synthetic_dataset(n_samples)`` which returns the same
   DataFrame structure using programmatically generated Pakistani business
   profiles.  When real data arrives, replace this call with ``build_dataset()``
   — no other code changes needed.

The output DataFrame always has:
- 60 numeric feature columns (from ``extract_features()``)
- 1 ``trust_label`` column  (0 or 1)
- 1 ``potential_label`` column (0 or 1)
- 1 ``business_name`` column  (string, for traceability)

Usage::

    # With real data
    from ml.dataset import build_dataset
    df = build_dataset(list_of_result_label_pairs)

    # With synthetic data (development)
    from ml.dataset import generate_synthetic_dataset
    df = generate_synthetic_dataset(n_samples=300)
    X = df.drop(columns=["trust_label", "potential_label", "business_name"])
    y_trust = df["trust_label"]
    y_potential = df["potential_label"]
"""

from __future__ import annotations

import random
from typing import List, Tuple

import pandas as pd

from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.feature import DiscoveredFeature, FeatureCategory
from agent.schemas.input import BusinessInput
from agent.schemas.result import (
    InvestigationResult,
    InvestigationStatus,
    Signal,
)
from ml.feature_extractor import extract_features


# ── Label types ───────────────────────────────────────────────────────────────

# A labeled sample: (result, trust_label, potential_label)
# trust_label:     1 = trustworthy,       0 = not trustworthy
# potential_label: 1 = good potential,    0 = low potential
LabeledSample = Tuple[InvestigationResult, int, int]


# ── Real-data entry point ─────────────────────────────────────────────────────

def build_dataset(samples: List[LabeledSample]) -> pd.DataFrame:
    """
    Convert a list of labeled InvestigationResults into a training DataFrame.

    Parameters
    ----------
    samples:
        List of ``(InvestigationResult, trust_label, potential_label)`` tuples.
        trust_label and potential_label must be 0 or 1.

    Returns
    -------
    pd.DataFrame
        60 feature columns + ``trust_label`` + ``potential_label`` +
        ``business_name``.  One row per sample.
    """
    rows = []
    for result, trust_label, potential_label in samples:
        features = extract_features(result)
        features["trust_label"] = float(trust_label)
        features["potential_label"] = float(potential_label)
        features["business_name"] = result.business_input.name
        rows.append(features)

    df = pd.DataFrame(rows)
    # Ensure label columns are integer type
    df["trust_label"] = df["trust_label"].astype(int)
    df["potential_label"] = df["potential_label"].astype(int)
    return df


# ── Synthetic data generator ──────────────────────────────────────────────────

# Pakistani business name fragments for realistic-looking synthetic names
_PK_PREFIXES = [
    "Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta", "Multan",
    "Faisalabad", "Rawalpindi", "Sialkot", "Gujranwala",
]
_PK_TYPES = [
    "Traders", "Enterprises", "Solutions", "Industries", "Store",
    "Services", "Works", "Brothers", "Co.", "Group",
]
_PK_SECTORS = [
    "Textile", "Food", "Tech", "Auto", "Pharma", "Retail",
    "Electronics", "Logistics", "Construction", "Agriculture",
]


def _random_name(rng: random.Random, idx: int) -> str:
    city = rng.choice(_PK_PREFIXES)
    sector = rng.choice(_PK_SECTORS)
    btype = rng.choice(_PK_TYPES)
    return f"{city} {sector} {btype} #{idx}"


def _make_evidence_item(
    rng: random.Random,
    field: str,
    value: str,
    etype: EvidenceType,
    reliability: SourceReliability,
    confidence: float,
    source_name: str,
    snippet: str,
) -> EvidenceItem:
    return EvidenceItem(
        field_name=field,
        value=value,
        evidence_type=etype,
        source_name=source_name,
        source_reliability=reliability,
        confidence=round(confidence, 2),
        raw_snippet=snippet,
    )


def _build_trustworthy_result(rng: random.Random, name: str) -> InvestigationResult:
    """
    Generate a synthetic high-trust, high-potential Pakistani business.

    Characteristics: mostly HIGH/MEDIUM sources, corroborated or observed
    evidence, positive sentiment snippets, many features found, few risk signals.
    """
    n_evidence = rng.randint(8, 15)
    evidence = []

    positive_snippets = [
        "excellent quality products consistently delivered on time",
        "very reliable supplier highly recommended by customers",
        "outstanding service and professional staff",
        "great reputation in local market transparent operations",
        "verified business registration legitimate operations",
        "authentic products good customer feedback",
        "reputable seller consistent quality praised by buyers",
        "trustworthy vendor established five years ago",
    ]
    fields = [
        "registration_status", "years_in_business", "customer_reviews",
        "social_presence", "revenue_estimate", "employee_count",
        "website_age", "complaints_record", "tax_compliance", "product_quality",
        "delivery_record", "market_reputation", "online_presence", "certifications",
        "return_policy",
    ]

    for i in range(n_evidence):
        reliability = rng.choice([
            SourceReliability.HIGH, SourceReliability.HIGH,
            SourceReliability.MEDIUM, SourceReliability.MEDIUM,
            SourceReliability.LOW,
        ])
        etype = rng.choice([
            EvidenceType.OBSERVED, EvidenceType.OBSERVED,
            EvidenceType.CORROBORATED, EvidenceType.CORROBORATED,
            EvidenceType.INFERENCE,
        ])
        confidence = rng.uniform(0.70, 0.97)
        snippet = rng.choice(positive_snippets)
        field = fields[i % len(fields)]
        evidence.append(_make_evidence_item(
            rng, field, f"value_{i}", etype, reliability,
            confidence, f"source_{rng.randint(1, 6)}", snippet,
        ))

    features = [
        DiscoveredFeature(name="registration", category=FeatureCategory.IDENTITY,
                          value="SECP registered", reason="Verified registration",
                          searched=True, confidence=0.9),
        DiscoveredFeature(name="reviews", category=FeatureCategory.REPUTATION,
                          value="4.5 stars", reason="Customer feedback signal",
                          searched=True, confidence=0.85),
        DiscoveredFeature(name="social_followers", category=FeatureCategory.AUDIENCE,
                          value=str(rng.randint(2000, 50000)),
                          reason="Social media reach", searched=True, confidence=0.8),
        DiscoveredFeature(name="monthly_demand", category=FeatureCategory.DEMAND,
                          value=str(rng.randint(50, 500)) + " orders/month",
                          reason="Demand indicator", searched=True, confidence=0.75),
        DiscoveredFeature(name="growth_rate", category=FeatureCategory.GROWTH,
                          value=f"{rng.randint(10, 40)}% YoY",
                          reason="Growth trajectory", searched=True, confidence=0.7),
    ]

    n_positive = rng.randint(3, 6)
    n_risk = rng.randint(0, 1)
    positive_signals = [
        Signal(label=f"Positive signal {i}", detail="Positive indicator found",
               evidence_refs=["registration_status"])
        for i in range(n_positive)
    ]
    risk_signals = [
        Signal(label=f"Minor risk {i}", detail="Minor risk noted",
               evidence_refs=["complaints_record"])
        for i in range(n_risk)
    ]

    return InvestigationResult(
        business_input=BusinessInput(name=name, location="Pakistan"),
        status=InvestigationStatus.COMPLETE,
        searches_performed=rng.randint(6, 12),
        sources_examined=rng.randint(4, 8),
        evidence=evidence,
        features=features,
        positive_signals=positive_signals,
        risk_signals=risk_signals,
        missing_information=[],
        sources=[f"https://source{i}.pk" for i in range(rng.randint(3, 6))],
        justification="",
    )


def _build_risky_result(rng: random.Random, name: str) -> InvestigationResult:
    """
    Generate a synthetic low-trust, low-potential Pakistani business.

    Characteristics: mostly LOW/UNKNOWN sources, inference-heavy evidence,
    negative sentiment snippets, few features found, many risk signals.
    """
    n_evidence = rng.randint(2, 7)
    evidence = []

    negative_snippets = [
        "scam complaints reported by multiple customers",
        "fraudulent seller fake products delivered",
        "terrible service no response to complaints",
        "suspicious operations no registration found",
        "dishonest misleading product descriptions",
        "unreliable vendor consistently late deliveries",
        "negative reviews across multiple platforms",
        "fraud allegations reported to consumer court",
    ]
    fields = [
        "registration_status", "complaints_record", "refund_policy",
        "contact_info", "social_presence", "reviews",
    ]

    for i in range(n_evidence):
        reliability = rng.choice([
            SourceReliability.LOW, SourceReliability.LOW,
            SourceReliability.UNKNOWN, SourceReliability.UNKNOWN,
            SourceReliability.MEDIUM,
        ])
        etype = rng.choice([
            EvidenceType.INFERENCE, EvidenceType.INFERENCE,
            EvidenceType.UNKNOWN, EvidenceType.OBSERVED,
        ])
        confidence = rng.uniform(0.20, 0.55)
        snippet = rng.choice(negative_snippets)
        field = fields[i % len(fields)]
        evidence.append(_make_evidence_item(
            rng, field, f"value_{i}", etype, reliability,
            confidence, f"source_{rng.randint(1, 3)}", snippet,
        ))

    features = [
        DiscoveredFeature(name="registration", category=FeatureCategory.IDENTITY,
                          value=None, reason="No registration found",
                          searched=True, confidence=0.3),
        DiscoveredFeature(name="risk_flag", category=FeatureCategory.RISK,
                          value="fraud allegations",
                          reason="Consumer complaints detected",
                          searched=True, confidence=0.6),
    ]

    n_positive = rng.randint(0, 1)
    n_risk = rng.randint(3, 6)
    positive_signals = [
        Signal(label="Minor positive", detail="Some online presence",
               evidence_refs=["social_presence"])
        for _ in range(n_positive)
    ]
    risk_signals = [
        Signal(label=f"Risk signal {i}", detail="Fraud or complaint indicator",
               evidence_refs=["complaints_record"])
        for i in range(n_risk)
    ]

    missing = ["SECP registration", "Tax records", "Business address",
               "Physical verification", "Bank statements"]

    return InvestigationResult(
        business_input=BusinessInput(name=name, location="Pakistan"),
        status=rng.choice([
            InvestigationStatus.PARTIAL,
            InvestigationStatus.PARTIAL,
            InvestigationStatus.COMPLETE,
        ]),
        searches_performed=rng.randint(2, 5),
        sources_examined=rng.randint(1, 3),
        evidence=evidence,
        features=features,
        positive_signals=positive_signals,
        risk_signals=risk_signals,
        missing_information=rng.sample(missing, rng.randint(2, 4)),
        sources=[f"https://source{i}.pk" for i in range(rng.randint(1, 3))],
        justification="",
    )


def _build_medium_result(rng: random.Random, name: str) -> InvestigationResult:
    """
    Generate a synthetic medium-trust, medium-potential Pakistani business.

    Mixed signals — some positive evidence, some gaps, moderate confidence.
    """
    n_evidence = rng.randint(4, 9)
    evidence = []

    mixed_snippets = [
        "decent quality but delivery sometimes delayed",
        "okay service some customers satisfied others not",
        "average reviews mixed feedback from buyers",
        "fair pricing but limited product range",
        "good local presence but no online verification",
        "some complaints but also positive testimonials",
        "moderate reliability inconsistent quality",
    ]
    fields = [
        "registration_status", "customer_reviews", "social_presence",
        "revenue_estimate", "complaints_record", "website_age", "product_quality",
    ]

    for i in range(n_evidence):
        reliability = rng.choice([
            SourceReliability.HIGH, SourceReliability.MEDIUM,
            SourceReliability.MEDIUM, SourceReliability.LOW,
        ])
        etype = rng.choice([
            EvidenceType.OBSERVED, EvidenceType.INFERENCE,
            EvidenceType.CORROBORATED, EvidenceType.UNKNOWN,
        ])
        confidence = rng.uniform(0.45, 0.75)
        snippet = rng.choice(mixed_snippets)
        field = fields[i % len(fields)]
        evidence.append(_make_evidence_item(
            rng, field, f"value_{i}", etype, reliability,
            confidence, f"source_{rng.randint(1, 4)}", snippet,
        ))

    features = [
        DiscoveredFeature(name="registration", category=FeatureCategory.IDENTITY,
                          value="local registration", reason="Partial verification",
                          searched=True, confidence=0.6),
        DiscoveredFeature(name="reviews", category=FeatureCategory.REPUTATION,
                          value="3.2 stars", reason="Mixed reviews",
                          searched=True, confidence=0.65),
        DiscoveredFeature(name="audience", category=FeatureCategory.AUDIENCE,
                          value=str(rng.randint(200, 2000)),
                          reason="Moderate social following", searched=True,
                          confidence=0.55),
    ]

    n_positive = rng.randint(1, 3)
    n_risk = rng.randint(1, 3)
    positive_signals = [
        Signal(label=f"Positive {i}", detail="Moderate positive signal",
               evidence_refs=["registration_status"])
        for i in range(n_positive)
    ]
    risk_signals = [
        Signal(label=f"Risk {i}", detail="Moderate risk signal",
               evidence_refs=["complaints_record"])
        for i in range(n_risk)
    ]
    missing = ["Tax records", "SECP filing", "Physical address"]

    return InvestigationResult(
        business_input=BusinessInput(name=name, location="Pakistan"),
        status=InvestigationStatus.COMPLETE,
        searches_performed=rng.randint(4, 8),
        sources_examined=rng.randint(2, 5),
        evidence=evidence,
        features=features,
        positive_signals=positive_signals,
        risk_signals=risk_signals,
        missing_information=rng.sample(missing, rng.randint(1, 2)),
        sources=[f"https://source{i}.pk" for i in range(rng.randint(2, 4))],
        justification="",
    )


def generate_synthetic_dataset(
    n_samples: int = 300,
    seed: int = 42,
    trust_ratio: float = 0.40,
    risky_ratio: float = 0.35,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset of Pakistani business investigations.

    Each sample is a structurally complete ``InvestigationResult`` built to
    mimic what Person 1's agent will produce for real businesses.  When real
    labeled data is available, replace this function with ``build_dataset()``
    — the DataFrame schema is identical.

    Parameters
    ----------
    n_samples:
        Total number of synthetic business profiles to generate.
    seed:
        Random seed for reproducibility.
    trust_ratio:
        Fraction of samples that are high-trust / approvable (label=1).
        Remaining (1 - trust_ratio - risky_ratio) are medium-trust.
    risky_ratio:
        Fraction of samples that are risky / not approvable (label=0).

    Returns
    -------
    pd.DataFrame
        60 feature columns + ``trust_label`` + ``potential_label`` +
        ``business_name``.
    """
    rng = random.Random(seed)

    n_trust = int(n_samples * trust_ratio)
    n_risky = int(n_samples * risky_ratio)
    n_medium = n_samples - n_trust - n_risky

    samples: List[LabeledSample] = []

    for i in range(n_trust):
        name = _random_name(rng, i)
        result = _build_trustworthy_result(rng, name)
        # trustworthy → trust=1, potential=1
        samples.append((result, 1, 1))

    for i in range(n_risky):
        name = _random_name(rng, n_trust + i)
        result = _build_risky_result(rng, name)
        # risky → trust=0, potential=0
        samples.append((result, 0, 0))

    for i in range(n_medium):
        name = _random_name(rng, n_trust + n_risky + i)
        result = _build_medium_result(rng, name)
        # medium trust → trust=1, potential=0  (adequate but limited upside)
        samples.append((result, 1, 0))

    # Shuffle so classes are not grouped
    rng.shuffle(samples)

    return build_dataset(samples)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return only the feature column names (excludes label and name columns)."""
    return [c for c in df.columns
            if c not in ("trust_label", "potential_label", "business_name")]


def split_features_labels(
    df: pd.DataFrame,
    target: str = "trust_label",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataset DataFrame into features X and label series y.

    Parameters
    ----------
    df:
        DataFrame from ``build_dataset()`` or ``generate_synthetic_dataset()``.
    target:
        Either ``"trust_label"`` or ``"potential_label"``.

    Returns
    -------
    X : pd.DataFrame  — 60 feature columns
    y : pd.Series     — integer labels (0 or 1)
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y
