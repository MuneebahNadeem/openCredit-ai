"""
Model Trainer — trains and persists ML models on investigation feature data.

Trains three classifiers on the 60 features extracted from InvestigationResult
objects:

- Random Forest
- Gradient Boosting  
- XGBoost

Each model is trained for both assessment targets (trustworthiness and business
potential) giving six trained models in total.  Models are saved to
``data/models/`` and can be loaded back for inference or evaluation.

Usage::

    # Train on synthetic data (development)
    from ml.model_trainer import train_all_models
    report = train_all_models()
    print(report.summary())

    # Train on real data (when Person 1's agent has run)
    from ml.dataset import build_dataset
    from ml.model_trainer import train_all_models
    df = build_dataset(your_labeled_samples)
    report = train_all_models(df=df)
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

from ml.dataset import generate_synthetic_dataset, split_features_labels

# ── Paths ─────────────────────────────────────────────────────────────────────

_MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "models"
)


def _model_path(model_name: str, target: str) -> str:
    os.makedirs(_MODELS_DIR, exist_ok=True)
    return os.path.join(_MODELS_DIR, f"{model_name}__{target}.pkl")


# ── Model definitions ─────────────────────────────────────────────────────────

def _make_models() -> Dict[str, object]:
    """Return fresh (untrained) instances of each classifier."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


# ── Training result ───────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Metrics for one trained model on one target."""

    model_name: str
    target: str
    accuracy: float
    f1: float
    roc_auc: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    train_samples: int
    test_samples: int
    n_features: int
    training_time_s: float
    model_path: str


@dataclass
class TrainingReport:
    """Results of training all models on both targets."""

    results: List[ModelResult] = field(default_factory=list)
    dataset_size: int = 0
    feature_count: int = 0
    data_source: str = "synthetic"

    def summary(self) -> str:
        lines = [
            f"Training Report — {self.data_source} data",
            f"Dataset: {self.dataset_size} samples, {self.feature_count} features",
            "",
            f"{'Model':<22} {'Target':<18} {'Acc':>6} {'F1':>6} {'AUC':>6} "
            f"{'CV Acc':>8} {'Time':>7}",
            "-" * 78,
        ]
        for r in sorted(self.results, key=lambda x: (x.target, x.model_name)):
            lines.append(
                f"{r.model_name:<22} {r.target:<18} "
                f"{r.accuracy:>6.3f} {r.f1:>6.3f} {r.roc_auc:>6.3f} "
                f"{r.cv_accuracy_mean:>6.3f}±{r.cv_accuracy_std:.3f} "
                f"{r.training_time_s:>5.1f}s"
            )
        return "\n".join(lines)

    def best_model(self, target: str = "trust_label") -> Optional[ModelResult]:
        """Return the best model for a given target by ROC-AUC."""
        target_results = [r for r in self.results if r.target == target]
        if not target_results:
            return None
        return max(target_results, key=lambda r: r.roc_auc)


# ── Core training ─────────────────────────────────────────────────────────────

def _train_single(
    model_name: str,
    clf,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target: str,
) -> ModelResult:
    """Train one model on one target and return its metrics."""
    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    # 5-fold cross-validation on the full training set
    cv_scores = cross_val_score(
        clf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )

    path = _model_path(model_name, target)
    with open(path, "wb") as f:
        pickle.dump(clf, f)

    return ModelResult(
        model_name=model_name,
        target=target,
        accuracy=round(accuracy, 4),
        f1=round(f1, 4),
        roc_auc=round(roc_auc, 4),
        cv_accuracy_mean=round(float(cv_scores.mean()), 4),
        cv_accuracy_std=round(float(cv_scores.std()), 4),
        train_samples=len(X_train),
        test_samples=len(X_test),
        n_features=X_train.shape[1],
        training_time_s=round(elapsed, 2),
        model_path=path,
    )


def train_all_models(
    df: Optional[pd.DataFrame] = None,
    n_synthetic: int = 300,
    test_size: float = 0.20,
    seed: int = 42,
) -> TrainingReport:
    """
    Train all three classifiers on both targets and save them.

    Parameters
    ----------
    df:
        Pre-built dataset DataFrame.  When ``None``, the synthetic generator
        is used.  Pass a real DataFrame from ``build_dataset()`` when real
        labeled data is available.
    n_synthetic:
        Number of synthetic samples to generate (ignored when ``df`` is given).
    test_size:
        Fraction of data to hold out for evaluation.
    seed:
        Random seed for train/test split.

    Returns
    -------
    TrainingReport
        Contains a ModelResult for each (model, target) combination (6 total)
        and a summary table.
    """
    if df is None:
        df = generate_synthetic_dataset(n_samples=n_synthetic, seed=seed)
        data_source = "synthetic"
    else:
        data_source = "real"

    report = TrainingReport(
        dataset_size=len(df),
        feature_count=df.shape[1] - 3,  # minus trust_label, potential_label, name
        data_source=data_source,
    )

    targets = ["trust_label", "potential_label"]

    for target in targets:
        X, y = split_features_labels(df, target=target)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        for model_name, clf in _make_models().items():
            result = _train_single(
                model_name, clf,
                X_train, X_test,
                y_train, y_test,
                target,
            )
            report.results.append(result)

    return report


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str, target: str):
    """
    Load a previously trained model from disk.

    Parameters
    ----------
    model_name:
        One of ``"random_forest"``, ``"gradient_boosting"``, ``"xgboost"``.
    target:
        Either ``"trust_label"`` or ``"potential_label"``.

    Returns
    -------
    Trained sklearn-compatible classifier.
    """
    path = _model_path(model_name, target)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at {path}. "
            f"Run train_all_models() first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model_name: str, target: str, X: pd.DataFrame) -> np.ndarray:
    """
    Load a model and run inference on a feature DataFrame.

    Returns predicted class labels (0 or 1).
    """
    clf = load_model(model_name, target)
    return clf.predict(X)


def predict_proba(model_name: str, target: str, X: pd.DataFrame) -> np.ndarray:
    """
    Load a model and return class probabilities.

    Returns an array of shape (n_samples,) with the probability of class 1.
    """
    clf = load_model(model_name, target)
    return clf.predict_proba(X)[:, 1]
