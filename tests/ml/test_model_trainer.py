"""
Tests for ml/dataset.py and ml/model_trainer.py.

Run with:  python -m pytest tests/ml/test_model_trainer.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import pandas as pd
import numpy as np

from agent.schemas.input import BusinessInput
from agent.schemas.evidence import EvidenceItem, EvidenceType, SourceReliability
from agent.schemas.result import InvestigationResult, InvestigationStatus, Signal
from ml.dataset import (
    build_dataset,
    generate_synthetic_dataset,
    get_feature_columns,
    split_features_labels,
)
from ml.model_trainer import (
    TrainingReport,
    ModelResult,
    train_all_models,
    load_model,
    predict,
    predict_proba,
    _make_models,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_minimal_result(name: str = "Test Co") -> InvestigationResult:
    evidence = [
        EvidenceItem(
            field_name="registration",
            value="registered",
            evidence_type=EvidenceType.OBSERVED,
            source_name="SECP",
            source_reliability=SourceReliability.HIGH,
            confidence=0.9,
            raw_snippet="excellent verified registration",
        ),
        EvidenceItem(
            field_name="reviews",
            value="4.5 stars",
            evidence_type=EvidenceType.CORROBORATED,
            source_name="Daraz",
            source_reliability=SourceReliability.MEDIUM,
            confidence=0.8,
            raw_snippet="great product quality",
        ),
    ]
    return InvestigationResult(
        business_input=BusinessInput(name=name),
        status=InvestigationStatus.COMPLETE,
        searches_performed=5,
        sources_examined=3,
        evidence=evidence,
        features=[],
        positive_signals=[Signal(label="Good reviews", detail="High rating",
                                 evidence_refs=["reviews"])],
        risk_signals=[],
        missing_information=[],
        sources=["https://secp.gov.pk"],
        justification="",
    )


# ── Dataset tests ─────────────────────────────────────────────────────────────

class TestBuildDataset:

    def test_returns_dataframe(self):
        samples = [(make_minimal_result("A"), 1, 1)]
        df = build_dataset(samples)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_sample(self):
        samples = [
            (make_minimal_result("A"), 1, 1),
            (make_minimal_result("B"), 0, 0),
            (make_minimal_result("C"), 1, 0),
        ]
        df = build_dataset(samples)
        assert len(df) == 3

    def test_has_trust_label_column(self):
        df = build_dataset([(make_minimal_result(), 1, 1)])
        assert "trust_label" in df.columns

    def test_has_potential_label_column(self):
        df = build_dataset([(make_minimal_result(), 1, 0)])
        assert "potential_label" in df.columns

    def test_has_business_name_column(self):
        df = build_dataset([(make_minimal_result("Lahore Traders"), 1, 1)])
        assert "business_name" in df.columns
        assert df["business_name"].iloc[0] == "Lahore Traders"

    def test_label_values_correct(self):
        samples = [
            (make_minimal_result("A"), 1, 0),
            (make_minimal_result("B"), 0, 1),
        ]
        df = build_dataset(samples)
        assert df["trust_label"].tolist() == [1, 0]
        assert df["potential_label"].tolist() == [0, 1]

    def test_label_columns_are_int(self):
        df = build_dataset([(make_minimal_result(), 1, 1)])
        assert df["trust_label"].dtype == int
        assert df["potential_label"].dtype == int

    def test_feature_columns_are_float(self):
        df = build_dataset([(make_minimal_result(), 1, 1)])
        feat_cols = get_feature_columns(df)
        for col in feat_cols:
            assert df[col].dtype in (np.float64, float)

    def test_correct_number_of_features(self):
        df = build_dataset([(make_minimal_result(), 1, 1)])
        assert len(get_feature_columns(df)) == 60


class TestGenerateSyntheticDataset:

    def test_returns_dataframe(self):
        df = generate_synthetic_dataset(n_samples=20)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = generate_synthetic_dataset(n_samples=30)
        assert len(df) == 30

    def test_has_required_columns(self):
        df = generate_synthetic_dataset(n_samples=10)
        assert "trust_label" in df.columns
        assert "potential_label" in df.columns
        assert "business_name" in df.columns

    def test_has_60_feature_columns(self):
        df = generate_synthetic_dataset(n_samples=10)
        assert len(get_feature_columns(df)) == 60

    def test_labels_are_binary(self):
        df = generate_synthetic_dataset(n_samples=50)
        assert set(df["trust_label"].unique()).issubset({0, 1})
        assert set(df["potential_label"].unique()).issubset({0, 1})

    def test_both_classes_present(self):
        df = generate_synthetic_dataset(n_samples=100)
        assert 0 in df["trust_label"].values
        assert 1 in df["trust_label"].values

    def test_reproducible_with_same_seed(self):
        df1 = generate_synthetic_dataset(n_samples=20, seed=7)
        df2 = generate_synthetic_dataset(n_samples=20, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_give_different_data(self):
        df1 = generate_synthetic_dataset(n_samples=20, seed=1)
        df2 = generate_synthetic_dataset(n_samples=20, seed=2)
        assert not df1["business_name"].equals(df2["business_name"])

    def test_all_features_finite(self):
        df = generate_synthetic_dataset(n_samples=30)
        feat_cols = get_feature_columns(df)
        assert df[feat_cols].notna().all().all()
        assert np.isfinite(df[feat_cols].values).all()

    def test_business_names_are_strings(self):
        df = generate_synthetic_dataset(n_samples=10)
        # pandas 3.x uses StringDtype; pandas 2.x uses object — both are string types
        assert pd.api.types.is_string_dtype(df["business_name"])


class TestSplitFeaturesLabels:

    def test_returns_tuple_of_two(self):
        df = generate_synthetic_dataset(n_samples=20)
        result = split_features_labels(df)
        assert len(result) == 2

    def test_X_has_60_columns(self):
        df = generate_synthetic_dataset(n_samples=20)
        X, y = split_features_labels(df)
        assert X.shape[1] == 60

    def test_y_has_correct_length(self):
        df = generate_synthetic_dataset(n_samples=20)
        X, y = split_features_labels(df)
        assert len(y) == 20

    def test_trust_label_target(self):
        df = generate_synthetic_dataset(n_samples=20)
        _, y = split_features_labels(df, target="trust_label")
        assert y.name == "trust_label"

    def test_potential_label_target(self):
        df = generate_synthetic_dataset(n_samples=20)
        _, y = split_features_labels(df, target="potential_label")
        assert y.name == "potential_label"

    def test_no_label_columns_in_X(self):
        df = generate_synthetic_dataset(n_samples=20)
        X, _ = split_features_labels(df)
        assert "trust_label" not in X.columns
        assert "potential_label" not in X.columns
        assert "business_name" not in X.columns


# ── Model trainer tests ───────────────────────────────────────────────────────

class TestMakeModels:

    def test_returns_three_models(self):
        models = _make_models()
        assert len(models) == 3

    def test_expected_model_names(self):
        models = _make_models()
        assert "random_forest" in models
        assert "gradient_boosting" in models
        assert "xgboost" in models

    def test_each_model_has_fit_predict(self):
        for name, clf in _make_models().items():
            assert hasattr(clf, "fit"), f"{name} missing fit()"
            assert hasattr(clf, "predict"), f"{name} missing predict()"
            assert hasattr(clf, "predict_proba"), f"{name} missing predict_proba()"


class TestTrainAllModels:

    @pytest.fixture(scope="class")
    def report(self):
        # Use small synthetic dataset so tests run fast
        return train_all_models(n_synthetic=120, seed=0)

    def test_returns_training_report(self, report):
        assert isinstance(report, TrainingReport)

    def test_six_results(self, report):
        # 3 models × 2 targets = 6
        assert len(report.results) == 6

    def test_all_results_are_model_result(self, report):
        for r in report.results:
            assert isinstance(r, ModelResult)

    def test_all_targets_covered(self, report):
        targets = {r.target for r in report.results}
        assert "trust_label" in targets
        assert "potential_label" in targets

    def test_all_model_names_covered(self, report):
        names = {r.model_name for r in report.results}
        assert "random_forest" in names
        assert "gradient_boosting" in names
        assert "xgboost" in names

    def test_accuracy_in_range(self, report):
        for r in report.results:
            assert 0.0 <= r.accuracy <= 1.0, f"{r.model_name}/{r.target}: {r.accuracy}"

    def test_f1_in_range(self, report):
        for r in report.results:
            assert 0.0 <= r.f1 <= 1.0

    def test_roc_auc_in_range(self, report):
        for r in report.results:
            assert 0.0 <= r.roc_auc <= 1.0

    def test_cv_accuracy_in_range(self, report):
        for r in report.results:
            assert 0.0 <= r.cv_accuracy_mean <= 1.0

    def test_training_time_positive(self, report):
        for r in report.results:
            assert r.training_time_s > 0.0

    def test_models_saved_to_disk(self, report):
        for r in report.results:
            assert os.path.exists(r.model_path), f"Model file missing: {r.model_path}"

    def test_dataset_size_stored(self, report):
        assert report.dataset_size == 120

    def test_feature_count_stored(self, report):
        assert report.feature_count == 60

    def test_data_source_synthetic(self, report):
        assert report.data_source == "synthetic"

    def test_summary_is_string(self, report):
        assert isinstance(report.summary(), str)

    def test_summary_contains_model_names(self, report):
        s = report.summary()
        assert "random_forest" in s
        assert "xgboost" in s

    def test_best_model_returns_model_result(self, report):
        best = report.best_model("trust_label")
        assert isinstance(best, ModelResult)

    def test_best_model_has_highest_auc(self, report):
        best = report.best_model("trust_label")
        trust_results = [r for r in report.results if r.target == "trust_label"]
        assert best.roc_auc == max(r.roc_auc for r in trust_results)


class TestLoadAndPredict:

    @pytest.fixture(scope="class", autouse=True)
    def train_first(self):
        train_all_models(n_synthetic=120, seed=0)

    def test_load_model_returns_classifier(self):
        clf = load_model("random_forest", "trust_label")
        assert hasattr(clf, "predict")

    def test_predict_returns_array(self):
        df = generate_synthetic_dataset(n_samples=5, seed=99)
        X, _ = split_features_labels(df)
        preds = predict("random_forest", "trust_label", X)
        assert len(preds) == 5

    def test_predict_labels_are_binary(self):
        df = generate_synthetic_dataset(n_samples=10, seed=99)
        X, _ = split_features_labels(df)
        preds = predict("xgboost", "trust_label", X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_returns_array(self):
        df = generate_synthetic_dataset(n_samples=5, seed=99)
        X, _ = split_features_labels(df)
        probs = predict_proba("gradient_boosting", "trust_label", X)
        assert len(probs) == 5

    def test_predict_proba_in_range(self):
        df = generate_synthetic_dataset(n_samples=10, seed=99)
        X, _ = split_features_labels(df)
        probs = predict_proba("random_forest", "trust_label", X)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_load_nonexistent_model_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("random_forest", "nonexistent_target")
