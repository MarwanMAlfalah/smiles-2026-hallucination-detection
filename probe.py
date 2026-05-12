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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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
        self._classifier: Pipeline | None = None
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

        Scales features, optionally reduces dimensionality with PCA, and fits
        a balanced logistic-regression classifier.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        n_samples, n_features = X.shape
        n_components = min(128, n_samples - 1, n_features)

        steps = [("scaler", StandardScaler())]
        if n_components >= 2 and n_features > n_components:
            steps.append(
                (
                    "pca",
                    PCA(
                        n_components=n_components,
                        random_state=42,
                        svd_solver="randomized",
                    ),
                )
            )
        steps.append(
            (
                "logreg",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                    solver="lbfgs",
                ),
            )
        )

        self._classifier = Pipeline(steps)
        self._classifier.fit(X, y)
        self._threshold = 0.5
        return self

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
        probs = self.predict_proba(X_val)[:, 1]

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
        if self._classifier is None:
            raise RuntimeError("Probe has not been fitted. Call fit() first.")

        probabilities = self._classifier.predict_proba(X)
        if probabilities.shape[1] == 2:
            classes = list(self._classifier.classes_)
            neg_idx = classes.index(0)
            pos_idx = classes.index(1)
            return probabilities[:, [neg_idx, pos_idx]]

        prob_pos = probabilities[:, 0]
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
