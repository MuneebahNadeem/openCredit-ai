"""
Model Evaluator — compares the saved models head-to-head and writes a report.

Loads the six trained artifacts from ``data/models/`` (3 algorithms × 2
targets, produced by ``ml/model_trainer.py``), evaluates each on a held-out
test set, and ranks them per target.  The evaluation protocol mirrors the
trainer (same split logic, same seed defaults) so the numbers are directly
comparable with the training report.

Metrics per model:
  - accuracy, precision, recall, F1, ROC-AUC   (classification)
  - log loss, Brier score                      (probabilistic calibration)
  - 5-fold cross-validated accuracy (mean ± std) — algorithm stability

Also provides tree feature-importance rankings and a markdown report writer.

Usage::

    from ml.model_evaluator import evaluate_saved_models, write_markdown_report

    report = evaluate_saved_models()          # synthetic data, seed 42
    print(report.summary())

    best = report.best("trust_label")         # winner by ROC-AUC
    print(best.model_name, best.roc_auc)

    write_markdown_report(report)             # documents/MODEL_EVALUATION.md

To evaluate on real labeled data::

    from ml.dataset import build_dataset
    df = build_dataset(labeled_samples)
    report = evaluate_saved_models(df=df)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from ml.dataset import (
    generate_synthetic_dataset,
    get_feature_columns,
    split_features_labels,
)
from ml.model_trainer import load_model

# ── Defaults (mirror the trainer so metrics are comparable) ──────────────────

_MODEL_NAMES = ["random_forest", "gradient_boosting", "xgboost"]
_TARGETS = ["trust_label", "potential_label"]

_DEFAULT_REPORT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "documents", "MODEL_EVALUATION.md"
)

# Metrics that can be used for ranking (higher is better).
_RANKABLE_METRICS = (
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "cv_accuracy_mean",
)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ModelEvaluation:
    """Metrics for one saved model on one target."""

    model_name: str
    target: str
    # Classification metrics (0.0–1.0)
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    # Probabilistic metrics (lower is better; NaN when undefined)
    log_loss: float
    brier: float
    # Cross-validated accuracy on the full dataset
    cv_accuracy_mean: float
    cv_accuracy_std: float
    # Confusion counts on the held-out test set
    tp: int
    tn: int
    fp: int
    fn: int
    # Dataset context
    train_samples: int
    test_samples: int
    n_features: int
    model_path: str
    loaded_from_disk: bool = True

    @property
    def confusion_total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn


@dataclass
class FeatureImportanceEntry:
    """One feature's importance in one model."""

    rank: int
    feature: str
    importance: float


@dataclass
class EvaluationReport:
    """Head-to-head comparison of the saved models on both targets."""

    evaluations: List[ModelEvaluation] = field(default_factory=list)
    dataset_size: int = 0
    feature_count: int = 0
    data_source: str = "synthetic"
    seed: int = 42
    test_size: float = 0.20
    generated_at: str = ""

    # ── Accessors ────────────────────────────────────────────────────────

    def for_target(self, target: str) -> List[ModelEvaluation]:
        """All evaluations for one target."""
        return [e for e in self.evaluations if e.target == target]

    def ranking(
        self, target: str, metric: str = "roc_auc",
    ) -> List[ModelEvaluation]:
        """Evaluations for a target, sorted best-first by ``metric``."""
        if metric not in _RANKABLE_METRICS:
            raise ValueError(
                f"Unknown ranking metric {metric!r}. "
                f"Choose from {sorted(_RANKABLE_METRICS)}."
            )
        evals = self.for_target(target)
        return sorted(evals, key=lambda e: getattr(e, metric), reverse=True)

    def best(
        self, target: str, metric: str = "roc_auc",
    ) -> Optional[ModelEvaluation]:
        """The winning model for a target by ``metric``."""
        ranked = self.ranking(target, metric=metric)
        return ranked[0] if ranked else None

    # ── Formatting ───────────────────────────────────────────────────────

    def summary(self) -> str:
        """Compact console table of all evaluations."""
        lines = [
            f"Model Evaluation Report — {self.data_source} data",
            f"Dataset: {self.dataset_size} samples, "
            f"{self.feature_count} features (seed {self.seed}, "
            f"test_size {self.test_size})",
            "",
            f"{'Model':<22} {'Target':<18} {'Acc':>6} {'Prec':>6} "
            f"{'Rec':>6} {'F1':>6} {'AUC':>6} {'CV Acc':>8}",
            "-" * 88,
        ]
        for e in sorted(self.evaluations, key=lambda x: (x.target, x.model_name)):
            lines.append(
                f"{e.model_name:<22} {e.target:<18} "
                f"{e.accuracy:>6.3f} {e.precision:>6.3f} {e.recall:>6.3f} "
                f"{e.f1:>6.3f} {e.roc_auc:>6.3f} "
                f"{e.cv_accuracy_mean:>6.3f}±{e.cv_accuracy_std:.3f}"
            )
        for target in _TARGETS:
            winner = self.best(target)
            if winner is not None:
                lines.append(
                    f"\nBest for {target}: {winner.model_name} "
                    f"(ROC-AUC {winner.roc_auc:.3f})"
                )
        return "\n".join(lines)

    def markdown(self) -> str:
        """Full markdown document of the evaluation."""
        now = self.generated_at or datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        lines = [
            "# OpenCredit AI — Model Evaluation Report",
            "",
            f"**Generated:** {now}  ",
            f"**Data source:** {self.data_source}  ",
            f"**Dataset:** {self.dataset_size} samples, "
            f"{self.feature_count} features  ",
            f"**Split:** seed {self.seed}, test_size {self.test_size}",
            "",
            "Evaluates the saved production artifacts in `data/models/` "
            "(3 algorithms × 2 targets) on a held-out test set.",
            "",
        ]

        for target in _TARGETS:
            label = target.replace("_label", "").replace("_", " ").title()
            ranked = self.ranking(target)
            if not ranked:
                continue
            lines += [
                f"## {label} — `{target}`",
                "",
                f"Ranking by ROC-AUC (best first):",
                "",
                "| # | Model | Acc | Prec | Rec | F1 | AUC | LogLoss | Brier | CV Acc |",
                "|---|-------|-----|------|-----|----|----|---------|-------|--------|",
            ]
            for i, e in enumerate(ranked, start=1):
                lines.append(
                    f"| {i} | `{e.model_name}` | {e.accuracy:.3f} "
                    f"| {e.precision:.3f} | {e.recall:.3f} | {e.f1:.3f} "
                    f"| {e.roc_auc:.3f} | {e.log_loss:.3f} | {e.brier:.3f} "
                    f"| {e.cv_accuracy_mean:.3f}±{e.cv_accuracy_std:.3f} |"
                )
            winner = ranked[0]
            lines += [
                "",
                f"**Winner:** `{winner.model_name}` "
                f"(ROC-AUC {winner.roc_auc:.3f})",
                "",
                "Confusion matrices (held-out test set):",
                "",
                "| Model | TP | TN | FP | FN |",
                "|-------|----|----|----|----|",
            ]
            for e in ranked:
                lines.append(
                    f"| `{e.model_name}` | {e.tp} | {e.tn} "
                    f"| {e.fp} | {e.fn} |"
                )
            lines.append("")

        lines += [
            "---",
            "",
            "*Metrics: accuracy / precision / recall / F1 / ROC-AUC on the "
            "held-out test set; LogLoss and Brier measure probabilistic "
            "calibration (lower is better); CV Acc is 5-fold cross-validated "
            "accuracy on the full dataset (algorithm stability).*",
            "",
        ]

        if self.data_source == "synthetic":
            lines += [
                "> **Caveat — synthetic data.** The synthetic generator "
                "derives both labels as deterministic functions of the 60 "
                "features, so perfect metrics here only confirm that each "
                "algorithm can learn the generator's rules.  They are **not** "
                "evidence of real-world performance.  Re-run this evaluation "
                "on real labeled investigations before trusting any model "
                "in production decisions.",
                "",
            ]

        return "\n".join(lines)


# ── Core evaluation ──────────────────────────────────────────────────────────

def _evaluate_single(
    clf,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Compute all test-set metrics for one fitted classifier."""
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    both_classes = len(np.unique(y_test)) == 2

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(
            float(precision_score(y_test, y_pred, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(
            float(roc_auc_score(y_test, y_prob)) if both_classes
            else float("nan"), 4
        ),
        "log_loss": round(
            float(log_loss(y_test, y_prob)) if both_classes
            else float("nan"), 4
        ),
        "brier": round(float(brier_score_loss(y_test, y_prob)), 4),
    }


def _confusion_counts(y_true, y_pred) -> tuple:
    """Return (tp, tn, fp, fn) from binary predictions."""
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    return int(tp), int(tn), int(fp), int(fn)


def evaluate_saved_models(
    df: Optional[pd.DataFrame] = None,
    n_synthetic: int = 300,
    test_size: float = 0.20,
    seed: int = 42,
    model_names: Optional[List[str]] = None,
    targets: Optional[List[str]] = None,
) -> EvaluationReport:
    """
    Evaluate all saved models on a held-out test set.

    Loads each saved artifact via ``ml.model_trainer.load_model`` — the
    production ``.pkl`` files are what gets scored, not freshly fitted
    copies.

    Parameters
    ----------
    df:
        Pre-built dataset (from ``build_dataset``) for real-data evaluation.
        When ``None``, a synthetic dataset is generated with ``n_synthetic``
        samples.
    n_synthetic:
        Synthetic sample count (ignored when ``df`` is given).
    test_size, seed:
        Split protocol — defaults mirror ``train_all_models`` so metrics are
        comparable with the training report.
    model_names, targets:
        Restrict evaluation to subsets.  Defaults: all 3 algorithms, both
        targets.

    Returns
    -------
    EvaluationReport with one ModelEvaluation per (model, target).

    Raises
    ------
    FileNotFoundError
        When a requested model has no saved artifact — run
        ``train_all_models()`` first.
    """
    if df is None:
        df = generate_synthetic_dataset(n_samples=n_synthetic, seed=seed)
        data_source = "synthetic"
    else:
        data_source = "real"

    names = model_names or _MODEL_NAMES
    target_list = targets or _TARGETS

    report = EvaluationReport(
        dataset_size=len(df),
        feature_count=len(get_feature_columns(df)),
        data_source=data_source,
        seed=seed,
        test_size=test_size,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    for target in target_list:
        X, y = split_features_labels(df, target=target)

        # Same split protocol as the trainer (stratified, seeded).
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        for model_name in names:
            # Score the saved production artifact, not a fresh fit.
            clf = load_model(model_name, target)

            metrics = _evaluate_single(clf, X_test, y_test)
            y_pred = clf.predict(X_test)
            tp, tn, fp, fn = _confusion_counts(y_test, y_pred)

            # Algorithm stability: 5-fold CV of a clone on the full dataset.
            cv_scores = cross_val_score(
                clone(clf), X, y, cv=5, scoring="accuracy", n_jobs=-1
            )

            report.evaluations.append(ModelEvaluation(
                model_name=model_name,
                target=target,
                train_samples=len(X_train),
                test_samples=len(X_test),
                n_features=X.shape[1],
                model_path=getattr(clf, "_opencredit_model_path", ""),
                tp=tp, tn=tn, fp=fp, fn=fn,
                cv_accuracy_mean=round(float(cv_scores.mean()), 4),
                cv_accuracy_std=round(float(cv_scores.std()), 4),
                loaded_from_disk=True,
                **metrics,
            ))

    return report


# ── Feature importance ───────────────────────────────────────────────────────

def feature_importance(
    model_name: str,
    target: str,
    top_n: int = 10,
) -> List[FeatureImportanceEntry]:
    """
    Top-N most important features for one saved model.

    All three algorithms are tree-based, so every saved model exposes
    ``feature_importances_``.  Feature names come from the standard
    60-column dataset schema.
    """
    clf = load_model(model_name, target)

    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        return []

    # Column names from the canonical dataset schema.
    sample_df = generate_synthetic_dataset(n_samples=2, seed=0)
    feature_names = get_feature_columns(sample_df)

    pairs = sorted(
        zip(feature_names, importances),
        key=lambda p: p[1],
        reverse=True,
    )[:top_n]

    return [
        FeatureImportanceEntry(rank=i, feature=name, importance=round(float(imp), 4))
        for i, (name, imp) in enumerate(pairs, start=1)
    ]


# ── Report writer ────────────────────────────────────────────────────────────

def write_markdown_report(
    report: EvaluationReport,
    path: Optional[str] = None,
    include_importance: bool = True,
    top_n: int = 10,
) -> str:
    """
    Write the evaluation report to a markdown file.

    When ``include_importance`` is set, the top features of each target's
    winning model are appended to the document.

    Returns the path written.
    """
    path = path or _DEFAULT_REPORT_PATH
    path = os.path.normpath(path)

    body = report.markdown()

    if include_importance:
        extra: List[str] = ["## Top features — winning models", ""]
        for target in _TARGETS:
            winner = report.best(target)
            if winner is None:
                continue
            entries = feature_importance(winner.model_name, target, top_n=top_n)
            if not entries:
                continue
            label = target.replace("_label", "").replace("_", " ").title()
            extra += [
                f"### {label} — `{winner.model_name}`",
                "",
                "| # | Feature | Importance |",
                "|---|---------|------------|",
            ]
            for e in entries:
                extra.append(
                    f"| {e.rank} | `{e.feature}` | {e.importance:.4f} |"
                )
            extra.append("")
        body = body + "\n".join(extra)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)

    return path
