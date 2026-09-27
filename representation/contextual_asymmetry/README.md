# Contextual representation and prediction-cost asymmetry

This package contains the sentence-level representation experiments used to
test whether matched American-English (AmE) and reference British-English
(Ref_BrE) variants remain semantically close while differing in model
prediction cost.

The package complements `representation/tokenization/`:

```text
tokenization asymmetry -> semantic equivalence -> prediction-cost asymmetry
```

The saved paper outputs are included so reviewers can inspect the results
without downloading gated models.

## Contents

- `scripts/compute_semantic_equivalence.py`: computes contextual cosine
  similarities for AmE/Ref_BrE counterfactuals and two controls.
- `scripts/compute_prediction_cost.py`: computes paired Ref_BrE-minus-AmE
  negative-log-likelihood gaps.
- `configs/`: model lists used for the 10-checkpoint runs.
- `results/sentence_pairs/`: saved controlled sentence-pair input used by the
  experiments.
- `results/semantic_equivalence/`: saved semantic-equivalence records,
  summaries, skipped-model logs, and run metadata.
- `results/prediction_cost/`: saved prediction-cost records, summaries,
  skipped-model logs, and run metadata.
- `plots/paper/`: paper-ready PDF and PNG figures. Figure/table rendering code
  is intentionally not duplicated in this release subpackage.

## Authentication and gated models

The full reruns load Hugging Face checkpoints. Some of these repositories may
require accepted access on Hugging Face before they can be downloaded or loaded
(for example, Llama and some Gemma/Mistral-family checkpoints). Do not place
tokens in this repository.

Preferred environment-variable route:

```bash
export HF_TOKEN="hf_..."
```

Alternative local CLI route:

```bash
hf auth login
hf auth whoami
```

The scripts also support `--local-files-only` when the checkpoints are already
available in the local Hugging Face cache. If access is missing, the scripts
record skipped models in `*_skips.jsonl` rather than embedding credentials.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r representation/contextual_asymmetry/requirements.txt
```

GPU runs should use a PyTorch build appropriate for the local CUDA setup.

## Sentence-pair input

The contextual experiments use the saved controlled sentence pairs in:

```text
representation/contextual_asymmetry/results/sentence_pairs/ame_bre_sentence_pairs.csv
```

These pairs were built from the shared resource
`resources/AmE_BrE_variations.json`. The release keeps the derived input used
for the paper-scale runs, but does not duplicate separate resource-construction
code in this subpackage. The confirmatory prediction-cost figure uses the
strict spelling/minimal-mention slice. The broader sentence-pair input also
contains orthographic-context rows for semantic-equivalence checks.

## Semantic-equivalence run

Example check:

```bash
python representation/contextual_asymmetry/scripts/compute_semantic_equivalence.py \
  --model google/gemma-3-1b-it \
  --max-rows 20 \
  --batch-size 2 \
  --bootstrap-samples 100 \
  --local-files-only
```

Full saved run:

```bash
python representation/contextual_asymmetry/scripts/compute_semantic_equivalence.py \
  --models representation/contextual_asymmetry/configs/semantic_equivalence_10model.yaml \
  --batch-size 2 \
  --dtype bfloat16 \
  --bootstrap-samples 1000
```

Saved outputs are in:

```text
representation/contextual_asymmetry/results/semantic_equivalence/
```

## Prediction-cost run

Example check:

```bash
python representation/contextual_asymmetry/scripts/compute_prediction_cost.py \
  --model google/gemma-3-1b-pt \
  --max-rows 50 \
  --batch-size 2 \
  --dtype bfloat16 \
  --bootstrap-samples 100 \
  --local-files-only
```

Full saved run:

```bash
python representation/contextual_asymmetry/scripts/compute_prediction_cost.py \
  --models representation/contextual_asymmetry/configs/prediction_cost_models.yaml \
  --batch-size 2 \
  --dtype bfloat16 \
  --bootstrap-samples 1000
```

Saved outputs are in:

```text
representation/contextual_asymmetry/results/prediction_cost/
```

The main paper figure uses:

```text
variant_type == spelling
template_type == minimal_mention
slice == same_sentence_token_count
```

Positive `mean_nll_per_token_gap_bre_minus_ame` means the Ref_BrE sentence is
more costly for the model to predict.

## Paper-ready figures

The release includes paper-ready figure files in:

```text
representation/contextual_asymmetry/plots/paper/
```

The figure/table creation code is not duplicated here; the plotted CSV summaries
and full experiment records are included under `results/`.

## Saved paper-scale results

- Semantic-equivalence run: 10/10 checkpoints complete; 108,000 similarity
  records; no skipped models.
- Prediction-cost run: 10/10 checkpoints complete; 85,150 loss records; no
  skipped models.

The results files and run metadata are anonymized and use repository-relative
paths only.
