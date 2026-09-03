"""
Tests for ml/model_evaluator.py — model comparison and evaluation.

Run with:  python -m pytest tests/ml/test_model_evaluator.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math

import pytest

from ml.dataset import generate_synthetic_dataset, get_feature_columns
from ml.model_trainer import train_all_models
from ml.model_evaluator import (
    EvaluationReport,
    ModelEvaluation,
    FeatureImportanceEntry,
    evaluate_saved_models,
    feature_importance,
    write_markdown_report,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_SYNTHETIC = 120
SEED = 0


@pytest.fixture(scope="module")
def trained():
    """Train small models once so the saved artifacts exist on disk."""
    return train_all_models(n_synthetic=N_SYNTHETIC, seed=SEED)


@pytest.fixture(scope="module")
def report(trained):
    """Evaluate the saved models once for the whole module."""
    return evaluate_saved_models(n_synthetic=N_SYNTHETIC, seed=SEED)


def make_evaluation(**overrides) -> ModelEvaluation:
    """A manually-built ModelEvaluation for structure tests."""
    defaults = dict(
        model_name="random_forest",
        target="trust_label",
        accuracy=0.9,
        precision=0.85,
        recall=0.88,
        f1=0.86,
        roc_auc=0.92,
        log_loss=0.3,
        brier=0.1,
        cv_accuracy_mean=0.88,
        cv_accuracy_std=0.03,
        tp=18, tn=22, fp=3, fn=3,
        train_samples=96,
        test_samples=24,
        n_features=60,
        model_path="data/models/random_forest__trust_label.pkl",
        loaded_from_disk=True,
    )
    defaults.update(overrides)
    return ModelEvaluation(**defaults)


# ── ModelEvaluation dataclass ────────────────────────────────────────────────

class TestModelEvaluation:

    def test_creation(self):
        e = make_evaluation()
        assert e.model_name == "random_forest"
        assert e.target == "trust_label"

    def test_confusion_total(self):
        e = make_evaluation()
        assert e.confusion_total == 18 + 22 + 3 + 3

    def test_default_loaded_from_disk(self):
        e = make_evaluation()
        assert e.loaded_from_disk is True


# ── EvaluationReport accessors ───────────────────────────────────────────────

class TestEvaluationReport:

    def test_for_target_filters(self):
        r = EvaluationReport(evaluations=[
            make_evaluation(target="trust_label"),
            make_evaluation(target="potential_label"),
        ])
        assert len(r.for_target("trust_label")) == 1
        assert len(r.for_target("potential_label")) == 1

    def test_for_target_empty(self):
        r = EvaluationReport(evaluations=[])
        assert r.for_target("trust_label") == []

    def test_ranking_sorted_descending(self):
        r = EvaluationReport(evaluations=[
            make_evaluation(model_name="a", roc_auc=0.70),
            make_evaluation(model_name="b", roc_auc=0.95),
            make_evaluation(model_name="c", roc_auc=0.80),
        ])
        ranked = r.ranking("trust_label")
        assert [e.model_name for e in ranked] == ["b", "c", "a"]

    def test_ranking_by_other_metric(self):
        r = EvaluationReport(evaluations=[
            make_evaluation(model_name="a", f1=0.99, roc_auc=0.10),
            make_evaluation(model_name="b", f1=0.50, roc_auc=0.99),
        ])
        ranked = r.ranking("trust_label", metric="f1")
        assert ranked[0].model_name == "a"

    def test_ranking_unknown_metric_raises(self):
        r = EvaluationReport(evaluations=[make_evaluation()])
        with pytest.raises(ValueError):
            r.ranking("trust_label", metric="not_a_metric")

    def test_best_returns_top(self):
        r = EvaluationReport(evaluations=[
            make_evaluation(model_name="a", roc_auc=0.70),
            make_evaluation(model_name="b", roc_auc=0.95),
        ])
        assert r.best("trust_label").model_name == "b"

    def test_best_empty_returns_none(self):
        r = EvaluationReport(evaluations=[])
        assert r.best("trust_label") is None

    def test_summary_is_string(self):
        r = EvaluationReport(evaluations=[make_evaluation()])
        assert isinstance(r.summary(), str)

    def test_summary_contains_model_names(self):
        r = EvaluationReport(evaluations=[
            make_evaluation(model_name="xgboost"),
            make_evaluation(model_name="random_forest"),
        ])
        s = r.summary()
        assert "xgboost" in s
        assert "random_forest" in s

    def test_summary_contains_best(self):
        r = EvaluationReport(evaluations=[
            make_evaluation(model_name="a", roc_auc=0.70),
            make_evaluation(model_name="b", roc_auc=0.95),
        ])
        s = r.summary()
        assert "Best for trust_label: b" in s

    def test_markdown_is_string(self):
        r = EvaluationReport(evaluations=[make_evaluation()])
        assert isinstance(r.markdown(), str)

    def test_markdown_contains_header(self):
        r = EvaluationReport(evaluations=[make_evaluation()])
        assert "Model Evaluation Report" in r.markdown()

    def test_markdown_contains_dataset_info(self):
        r = EvaluationReport(
            evaluations=[make_evaluation()],
            dataset_size=300, feature_count=60, data_source="synthetic",
        )
        md = r.markdown()
        assert "300" in md
        assert "synthetic" in md

    def test_markdown_contains_ranking_table(self):
        r = EvaluationReport(evaluations=[
            make_evaluation(model_name="xgboost", roc_auc=0.95),
            make_evaluation(model_name="random_forest", roc_auc=0.70),
        ])
        md = r.markdown()
        assert "Winner" in md
        assert "xgboost" in md

    def test_markdown_synthetic_includes_caveat(self):
        r = EvaluationReport(
            evaluations=[make_evaluation()], data_source="synthetic",
        )
        assert "Caveat" in r.markdown()

    def test_markdown_real_data_has_no_caveat(self):
        r = EvaluationReport(
            evaluations=[make_evaluation()], data_source="real",
        )
        assert "Caveat" not in r.markdown()


# ── evaluate_saved_models (integration) ──────────────────────────────────────

class TestEvaluateSavedModels:

    def test_returns_evaluation_report(self, report):
        assert isinstance(report, EvaluationReport)

    def test_six_evaluations(self, report):
        # 3 models × 2 targets
        assert len(report.evaluations) == 6

    def test_all_evaluations_are_model_evaluation(self, report):
        for e in report.evaluations:
            assert isinstance(e, ModelEvaluation)

    def test_all_models_covered(self, report):
        names = {e.model_name for e in report.evaluations}
        assert names == {"random_forest", "gradient_boosting", "xgboost"}

    def test_all_targets_covered(self, report):
        targets = {e.target for e in report.evaluations}
        assert targets == {"trust_label", "potential_label"}

    def test_report_metadata(self, report):
        assert report.dataset_size == N_SYNTHETIC
        assert report.feature_count == 60
        assert report.data_source == "synthetic"
        assert report.seed == SEED
        assert report.test_size == 0.20

    def test_generated_at_populated(self, report):
        assert len(report.generated_at) > 0

    @pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "f1", "roc_auc"])
    def test_classification_metrics_in_range(self, report, metric):
        for e in report.evaluations:
            value = getattr(e, metric)
            assert 0.0 <= value <= 1.0, (
                f"{e.model_name}/{e.target} {metric}={value}"
            )

    def test_brier_in_range(self, report):
        for e in report.evaluations:
            assert 0.0 <= e.brier <= 1.0

    def test_log_loss_non_negative(self, report):
        for e in report.evaluations:
            assert e.log_loss >= 0.0

    def test_cv_accuracy_in_range(self, report):
        for e in report.evaluations:
            assert 0.0 <= e.cv_accuracy_mean <= 1.0
            assert e.cv_accuracy_std >= 0.0

    def test_confusion_matches_test_size(self, report):
        for e in report.evaluations:
            assert e.confusion_total == e.test_samples

    def test_train_test_split_sizes(self, report):
        for e in report.evaluations:
            assert e.train_samples + e.test_samples == N_SYNTHETIC

    def test_n_features_is_60(self, report):
        for e in report.evaluations:
            assert e.n_features == 60

    def test_loaded_from_disk_true(self, report):
        for e in report.evaluations:
            assert e.loaded_from_disk is True

    def test_best_is_max_roc_auc(self, report):
        for target in ("trust_label", "potential_label"):
            best = report.best(target)
            all_aucs = [e.roc_auc for e in report.for_target(target)]
            assert best.roc_auc == max(all_aucs)

    def test_metrics_reasonable_on_synthetic(self, report):
        """Models trained on synthetic data should clearly beat chance."""
        for e in report.evaluations:
            assert e.accuracy > 0.60, (
                f"{e.model_name}/{e.target} accuracy={e.accuracy}"
            )

    def test_real_dataframe_evaluation(self, trained):
        """Evaluation accepts a pre-built df (real-data path)."""
        df = generate_synthetic_dataset(n_samples=N_SYNTHETIC, seed=SEED)
        r = evaluate_saved_models(df=df, seed=SEED)
        assert isinstance(r, EvaluationReport)
        assert r.data_source == "real"
        assert r.dataset_size == N_SYNTHETIC
        assert len(r.evaluations) == 6

    def test_subset_of_models(self, trained):
        r = evaluate_saved_models(
            n_synthetic=N_SYNTHETIC, seed=SEED,
            model_names=["random_forest"],
        )
        assert len(r.evaluations) == 2  # 1 model × 2 targets
        assert all(e.model_name == "random_forest" for e in r.evaluations)

    def test_subset_of_targets(self, trained):
        r = evaluate_saved_models(
            n_synthetic=N_SYNTHETIC, seed=SEED,
            targets=["trust_label"],
        )
        assert len(r.evaluations) == 3  # 3 models × 1 target
        assert all(e.target == "trust_label" for e in r.evaluations)

    def test_missing_model_raises(self, trained):
        with pytest.raises(FileNotFoundError):
            evaluate_saved_models(
                n_synthetic=N_SYNTHETIC, seed=SEED,
                model_names=["no_such_model"],
            )


# ── feature_importance ───────────────────────────────────────────────────────

class TestFeatureImportance:

    def test_returns_list(self, trained):
        entries = feature_importance("random_forest", "trust_label")
        assert isinstance(entries, list)

    def test_entries_are_feature_importance_entry(self, trained):
        for e in feature_importance("xgboost", "potential_label"):
            assert isinstance(e, FeatureImportanceEntry)

    def test_top_n_respected(self, trained):
        entries = feature_importance("random_forest", "trust_label", top_n=5)
        assert len(entries) == 5

    def test_sorted_descending(self, trained):
        entries = feature_importance("gradient_boosting", "trust_label", top_n=10)
        importances = [e.importance for e in entries]
        assert importances == sorted(importances, reverse=True)

    def test_ranks_are_sequential(self, trained):
        entries = feature_importance("random_forest", "trust_label", top_n=8)
        assert [e.rank for e in entries] == list(range(1, 9))

    def test_features_are_valid_names(self, trained):
        valid = set(get_feature_columns(generate_synthetic_dataset(n_samples=2)))
        for e in feature_importance("xgboost", "trust_label"):
            assert e.feature in valid

    def test_importances_in_range(self, trained):
        for e in feature_importance("random_forest", "potential_label"):
            assert 0.0 <= e.importance <= 1.0

    def test_importances_sum_leq_one(self, trained):
        """Full feature importance vector sums to ≤ 1 for tree models."""
        entries = feature_importance(
            "gradient_boosting", "trust_label", top_n=60
        )
        assert sum(e.importance for e in entries) <= 1.0 + 1e-6


# ── write_markdown_report ────────────────────────────────────────────────────

class TestWriteMarkdownReport:

    def test_writes_file_and_returns_path(self, report, tmp_path):
        path = write_markdown_report(report, path=str(tmp_path / "eval.md"))
        assert os.path.exists(path)
        assert path == str(tmp_path / "eval.md")

    def test_file_contains_header(self, report, tmp_path):
        path = write_markdown_report(report, path=str(tmp_path / "eval.md"))
        content = open(path, encoding="utf-8").read()
        assert "Model Evaluation Report" in content

    def test_file_contains_model_names(self, report, tmp_path):
        path = write_markdown_report(report, path=str(tmp_path / "eval.md"))
        content = open(path, encoding="utf-8").read()
        assert "xgboost" in content
        assert "random_forest" in content
        assert "gradient_boosting" in content

    def test_file_contains_winners(self, report, tmp_path):
        path = write_markdown_report(report, path=str(tmp_path / "eval.md"))
        content = open(path, encoding="utf-8").read()
        assert "Winner" in content

    def test_file_contains_importance_section(self, report, tmp_path):
        path = write_markdown_report(report, path=str(tmp_path / "eval.md"))
        content = open(path, encoding="utf-8").read()
        assert "Top features" in content

    def test_importance_can_be_disabled(self, report, tmp_path):
        path = write_markdown_report(
            report, path=str(tmp_path / "eval.md"), include_importance=False
        )
        content = open(path, encoding="utf-8").read()
        assert "Top features" not in content

    def test_default_path_constant_exists(self):
        from ml.model_evaluator import _DEFAULT_REPORT_PATH
        assert _DEFAULT_REPORT_PATH.endswith("MODEL_EVALUATION.md")
