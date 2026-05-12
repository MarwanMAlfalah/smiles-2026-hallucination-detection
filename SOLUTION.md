# Solution: Single-Best-Layer Hidden-State Probe for Hallucination Detection

## 1. Reproducibility

To reproduce the solution from a fresh checkout, run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solution.py
```

Running `python solution.py` extracts hidden-state features from the provided Qwen2.5-0.5B model, trains and evaluates the probe, and generates:

- `results.json`
- `predictions.csv`

`predictions.csv` contains the final predictions for the provided unlabeled `test.csv` file.

## 2. Main idea

The first simple idea was to use hidden representations from several transformer layers. That gave useful signal, but concatenating many raw vectors creates a high-dimensional feature space, which is risky for a dataset with only 689 labelled samples.

The final approach tries to keep the useful hidden-state signal while reducing overfitting. It is a SAPLMA-inspired single-best-layer hidden-state probe. Instead of using all selected layers at once, it extracts last-real-token vectors from several candidate middle-to-late transformer layers and lets the probe choose the most informative layer using validation AUROC.

This is not an implementation of SAPLMA itself. It is only inspired by the idea that a single well-chosen layer can be more reliable than a large concatenation of many layer representations.

## 3. Feature extraction

Feature extraction is implemented in `aggregation.py`.

The official pipeline encodes the concatenation of `prompt + response`, so the probe does not do response-only pooling. I use the last real non-padding token from the full prompt-response sequence.

The aggregation step:

- selects candidate layers dynamically using fractions `0.30`, `0.50`, `0.70`, `0.90`, and `1.00`
- uses `attention_mask` to find the last real non-padding token
- extracts the last-real-token hidden vector from each selected layer
- concatenates the candidate layer vectors in a fixed order
- adds a small scalar tail with:
  - per-layer L2 norms
  - consecutive L2 distances
  - consecutive cosine similarities
  - end-to-end distance

For Qwen2.5-0.5B, this gives a final feature dimension of `4494`.

## 4. Probe/classifier

The classifier is implemented in `probe.py`.

The probe:

- infers the five layer groups plus the scalar tail from the feature dimension
- trains one regularized logistic-regression probe per layer group
- selects the logistic-regression `C` value using 3-fold stratified cross-validation on the training split
- selects the best layer using validation AUROC
- also checks a simple top-3 layer probability ensemble
- also checks a best-layer-plus-tail hybrid
- uses the validation set to tune the decision threshold for F1

The classifier uses `StandardScaler` and `LogisticRegression` with balanced class weights. `predict_proba` returns probabilities in the expected format:

```text
[p_truthful, p_hallucinated]
```

## 5. Experiments considered

The Rich Layer-Stability Probe used mean and last-token vectors from several layers. It gave useful signal, but it also created a large feature vector and showed more overfitting. Its internal held-out test split AUROC was `0.625`.

The Compact Spectral Trajectory Probe used only scalar statistics from hidden states. It reduced the feature dimension to 86 and improved accuracy/F1 in some places, but it did not improve the primary AUROC metric. Its internal held-out test split AUROC was `0.600`.

The final Single-Best-Layer Probe gave the best validation and internal held-out test split AUROC among my attempts. Its internal held-out test split AUROC was `0.664`.

## 6. Results

Final internal validation/evaluation results:

- validation AUROC: `0.7269`
- internal held-out test split AUROC: `0.6642`
- internal held-out test split accuracy: `0.7115`
- internal held-out test split F1: `0.8125`

The dataset is small, so I treated the validation and internal held-out test split numbers as guidance rather than absolute proof.

## 7. Files modified

The final solution modifies:

- `aggregation.py`
- `probe.py`

`splitting.py` was left unchanged because the existing stratified split was already appropriate.

This solution is intentionally lightweight. I chose not to fine-tune the language model or add external models, because I wanted to keep the pipeline reproducible and focused on hidden-state probing within the provided project structure.