# Tokenization analysis

This package evaluates the tokenizers listed in `configs/tokenizers.json` on the shared AmE--BrE variant pairs.
Download the full variant list from the Hugging Face dataset before paper-scale runs; see `resources/README.md`.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r representation/tokenization/requirements.txt
```

Some Hugging Face repositories require accepted access:

```bash
export HF_TOKEN="your_token"
```

The token can also be passed with `--hf-token-file PATH`. Token files should remain outside the repository.

## Run the paper tokenizers

```bash
python representation/tokenization/scripts/run_tokenizer_analysis.py run \
  --config representation/tokenization/configs/tokenizers.json \
  --output-dir representation/tokenization/results/paper
```

Use `--only MODEL_ID` to select one or more tokenizers. By default, an unavailable gated tokenizer is reported and the remaining models continue. Use `--fail-on-unavailable` when a missing tokenizer should stop the run.

## Local check

```bash
bash representation/tokenization/scripts/run_example_checks.sh
```

The local check uses a character tokenizer and does not contact Hugging Face.

## Outputs

- `tokenizer_summary.csv`: backend, repository, vocabulary sizes, and run information.
- `tokenizer_pair_lengths.csv`: token counts and fertility for every variant pair.
- `tokenizer_category_fertility.csv`: category means and paired Wilcoxon tests.
- `tokenizer_length_bins.csv`: token-length bin counts used for downstream reporting.
- `tokenizer_failures.json`: tokenizers that could not be loaded.

`configs/additional_tokenizers.json` contains optional models that are not part of the main table.
