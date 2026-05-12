"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


_LAYER_FRACTIONS = (0.35, 0.55, 0.75, 0.90, 1.00)


def _selected_layer_indices(n_layers: int) -> list[int]:
    """Pick unique middle-to-late layer indices for any hidden-state depth."""
    if n_layers <= 1:
        return [0]

    last_idx = n_layers - 1
    indices: list[int] = []
    for fraction in _LAYER_FRACTIONS:
        idx = int(round(fraction * last_idx))
        idx = min(max(idx, 1), last_idx)
        if idx not in indices:
            indices.append(idx)
    return indices


def _real_token_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return token positions where the attention mask marks real tokens."""
    positions = attention_mask.nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return torch.arange(attention_mask.shape[0], device=attention_mask.device)
    return positions


def _selected_layer_means(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[list[int], torch.Tensor]:
    """Mean-pool all real tokens for the selected layers."""
    layer_indices = _selected_layer_indices(hidden_states.shape[0])
    positions = _real_token_positions(attention_mask).to(hidden_states.device)
    means = [
        hidden_states[layer_idx, positions].mean(dim=0)
        for layer_idx in layer_indices
    ]
    return layer_indices, torch.stack(means, dim=0)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    layer_indices, mean_vectors = _selected_layer_means(hidden_states, attention_mask)
    real_positions = _real_token_positions(attention_mask).to(hidden_states.device)
    last_pos = int(real_positions[-1].item())

    pooled_features = []
    for layer_idx, mean_vector in zip(layer_indices, mean_vectors):
        last_token_vector = hidden_states[layer_idx, last_pos]
        pooled_features.extend([mean_vector, last_token_vector])

    if mean_vectors.shape[0] > 1:
        previous = mean_vectors[:-1]
        current = mean_vectors[1:]
        layer_distances = torch.linalg.vector_norm(current - previous, dim=1)
        layer_cosines = F.cosine_similarity(previous, current, dim=1)
    else:
        layer_distances = mean_vectors.new_zeros(0)
        layer_cosines = mean_vectors.new_zeros(0)

    scalar_features = torch.cat(
        [
            torch.linalg.vector_norm(mean_vectors, dim=1),
            layer_distances,
            layer_cosines,
            mean_vectors.std(dim=1),
        ],
        dim=0,
    )

    return torch.cat([*pooled_features, scalar_features], dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    _, mean_vectors = _selected_layer_means(hidden_states, attention_mask)
    real_positions = _real_token_positions(attention_mask).to(hidden_states.device)
    sequence_length = mean_vectors.new_tensor([float(real_positions.numel())])

    final_layer_tokens = hidden_states[-1, real_positions]
    token_norms = torch.linalg.vector_norm(final_layer_tokens, dim=1)
    token_norm_summary = torch.stack(
        [
            token_norms.mean(),
            token_norms.std(unbiased=False),
            token_norms.min(),
            token_norms.max(),
        ]
    )

    layer_norms = torch.linalg.vector_norm(mean_vectors, dim=1)
    layer_norm_summary = torch.stack(
        [
            layer_norms.mean(),
            layer_norms.std(unbiased=False),
        ]
    )

    return torch.cat([sequence_length, token_norm_summary, layer_norm_summary], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
