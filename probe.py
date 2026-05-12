"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary classifier that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py`` via
``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module`` for compatibility with the provided evaluation
    code, but the actual probe is a lightweight scikit-learn classifier.
    """

    def __init__(self) -> None:
        super().__init__()
        self._classifier = None
        self._selected_models: list[tuple[object, np.ndarray]] = []
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Kept for API compatibility with the original neural probe.

        The Layer-Stability Probe uses scikit-learn in ``fit`` instead.
        """
        return None

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch forward pass is not used by the sklearn probe.

        The method remains defined so ``HallucinationProbe`` still satisfies
        the original ``nn.Module``-based interface.
        """
        raise NotImplementedError("HallucinationProbe uses predict/predict_proba.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Stores the training data and fits a balanced logistic-regression
        default.  ``fit_hyperparameters`` may replace this with a selected
        single-layer, ensemble, or hybrid model using validation AUROC.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        self._X_train = np.asarray(X, dtype=np.float64)
        self._y_train = np.asarray(y, dtype=int)

        layout = self._infer_layer_groups(self._X_train.shape[1])
        if layout is not None:
            groups, _ = layout
            model_scores = [
                self._fit_best_logistic(
                    self._X_train[:, columns],
                    self._y_train,
                )
                for columns in groups
            ]
            # Pick a CV-selected single layer when no validation split is
            # available, as in final competition prediction.
            best_model, best_auc = model_scores[0]
            best_columns = groups[0]
            for (model, auc), columns in zip(model_scores, groups):
                if auc > best_auc:
                    best_auc = auc
                    best_model = model
                    best_columns = columns
            self._classifier = best_model
            self._selected_models = [(best_model, best_columns)]
        else:
            self._classifier = self._make_logistic(C=0.1)
            self._classifier.fit(self._X_train, self._y_train)
            self._selected_models = [
                (self._classifier, np.arange(self._X_train.shape[1]))
            ]
        self._threshold = 0.5
        return self

    def _make_logistic(self, C: float) -> Pipeline:
        """Create a scaled balanced logistic-regression pipeline."""
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=C,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=42,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def _infer_layer_groups(
        self,
        n_features: int,
    ) -> tuple[list[np.ndarray], np.ndarray] | None:
        """Infer layer-vector groups and scalar tail from aggregation output."""
        for num_layers in [5, 4, 3, 2, 1]:
            tail_len = 3 * num_layers - 1
            vector_dim_total = n_features - tail_len
            if vector_dim_total <= 0 or vector_dim_total % num_layers != 0:
                continue

            hidden_dim = vector_dim_total // num_layers
            if hidden_dim < 8:
                continue

            groups = [
                np.arange(i * hidden_dim, (i + 1) * hidden_dim)
                for i in range(num_layers)
            ]
            tail = np.arange(vector_dim_total, n_features)
            return groups, tail
        return None

    def _positive_proba(self, model, X: np.ndarray) -> np.ndarray:
        """Return class-1 probability from any fitted sklearn classifier."""
        probabilities = model.predict_proba(X)
        classes = list(model.classes_)
        return probabilities[:, classes.index(1)]

    def _validation_auc(self, y_true: np.ndarray, prob_pos: np.ndarray) -> float:
        """AUROC helper that stays finite if a split is degenerate."""
        try:
            return float(roc_auc_score(y_true, prob_pos))
        except ValueError:
            return -1.0

    def _fit_best_logistic(self, X: np.ndarray, y: np.ndarray) -> tuple[Pipeline, float]:
        """Choose regularization by stratified CV, then fit on all data."""
        c_values = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
        class_counts = np.bincount(y.astype(int), minlength=2)
        n_splits = int(min(3, class_counts.min()))

        best_c = 0.1
        best_score = -1.0
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for c_value in c_values:
                model = self._make_logistic(c_value)
                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring="roc_auc",
                    error_score=np.nan,
                )
                score = float(np.nanmean(scores))
                if score > best_score:
                    best_score = score
                    best_c = c_value

        model = self._make_logistic(best_c)
        model.fit(X, y)
        return model, best_score

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Probe has not been fitted. Call fit() first.")

        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=int)

        layout = self._infer_layer_groups(self._X_train.shape[1])
        candidate_models: list[dict] = []

        if layout is not None:
            groups, tail = layout
            for group_index, columns in enumerate(groups):
                model, _ = self._fit_best_logistic(
                    self._X_train[:, columns],
                    self._y_train,
                )
                probs_candidate = self._positive_proba(model, X_val[:, columns])
                candidate_models.append(
                    {
                        "auc": self._validation_auc(y_val, probs_candidate),
                        "models": [(model, columns)],
                        "probs": probs_candidate,
                        "name": f"layer_{group_index}",
                    }
                )

            candidate_models.sort(key=lambda item: item["auc"], reverse=True)

            top3 = candidate_models[:3]
            if len(top3) >= 2:
                ensemble_probs = np.mean([item["probs"] for item in top3], axis=0)
                candidate_models.append(
                    {
                        "auc": self._validation_auc(y_val, ensemble_probs),
                        "models": [
                            model_spec
                            for item in top3
                            for model_spec in item["models"]
                        ],
                        "probs": ensemble_probs,
                        "name": "top3_ensemble",
                    }
                )

            best_layer_columns = candidate_models[0]["models"][0][1]
            hybrid_columns = np.concatenate([best_layer_columns, tail])
            hybrid_model, _ = self._fit_best_logistic(
                self._X_train[:, hybrid_columns],
                self._y_train,
            )
            hybrid_probs = self._positive_proba(hybrid_model, X_val[:, hybrid_columns])
            candidate_models.append(
                {
                    "auc": self._validation_auc(y_val, hybrid_probs),
                    "models": [(hybrid_model, hybrid_columns)],
                    "probs": hybrid_probs,
                    "name": "best_layer_plus_tail",
                }
            )
        else:
            all_columns = np.arange(self._X_train.shape[1])
            model, _ = self._fit_best_logistic(self._X_train, self._y_train)
            probs_candidate = self._positive_proba(model, X_val)
            candidate_models.append(
                {
                    "auc": self._validation_auc(y_val, probs_candidate),
                    "models": [(model, all_columns)],
                    "probs": probs_candidate,
                    "name": "all_features",
                }
            )

        best_candidate = max(candidate_models, key=lambda item: item["auc"])
        self._selected_models = best_candidate["models"]
        self._classifier = self._selected_models[0][0]
        probs = best_candidate["probs"]

        best_threshold = 0.5
        best_f1 = -1.0
        for t in np.linspace(0.05, 0.95, 91):
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        if not self._selected_models:
            raise RuntimeError("Probe has not been fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        prob_pos = np.mean(
            [
                self._positive_proba(model, X[:, columns])
                for model, columns in self._selected_models
            ],
            axis=0,
        )
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
