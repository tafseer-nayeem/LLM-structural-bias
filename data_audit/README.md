# Data audit

This package counts paired American English (AmE) and reference British English (Ref-BrE) variants in pretraining and post-training data. It uses the same normalization, grouping rules, and summary statistics for every corpus.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r data_audit/requirements.txt
```

The default variant list is `resources/AmE_BrE_variations.json`.
Download it from the Hugging Face dataset before paper-scale runs; see `resources/README.md`.

## Check the installation

```bash
bash data_audit/scripts/run_offline_checks.sh
```

This runs a small local example and writes its output to `data_audit/results/example_check/`.

## Pretraining corpora

Download the word-frequency files from the link documented in `external/pretraining_word_frequencies/README.md` and place them in the directory structure shown there. Then run:

```bash
python data_audit/scripts/run_data_audit.py \
  --config data_audit/configs/pretraining_corpora.json \
  --output-dir data_audit/results/pretraining
```

### Document-level DiAlign

The frequency audit above is lexical. A separate command scores individual pretraining documents with DiAlign while retaining the same six corpus labels:

```bash
python -m nltk.downloader stopwords averaged_perceptron_tagger_eng maxent_ne_chunker_tab words
python data_audit/scripts/run_pretraining_dialign.py \
  --documents-per-corpus 1000
```

`--documents-per-corpus` is required because the source corpora are very large. Hub datasets are streamed after deterministic buffered shuffling. BookCorpus and the sampled Dolma release use the local paths described in `external/raw_pretraining/README.md`. Use `--only wikipedia common_crawl` to select corpora, or `--max-words N` to apply and record an explicit per-document length cap.

For an archival run, a Hugging Face commit hash can be placed in an entry's optional `revision` field. The selected revision is copied into the corresponding result summary.

The command writes one JSONL record per document and JSON/CSV corpus summaries to `results/pretraining_dialign/`. The document table contains source and scored-text hashes, a source identifier, document lengths, both alignment scores, its alignment label, and candidate and evidence n-gram counts rather than the document text. Mean alignment and dialect shares exclude documents for which Google Books provides no usable directional evidence; evidence coverage and the no-evidence count are reported separately. Top-contributor evidence is omitted by default and can be requested with `--include-contributors`.

DiAlign uses a local SQLite frequency store at `dialign/.cache/ngram_frequencies.sqlite3`. This is especially useful for document-level runs because n-grams shared across documents are queried only once.

## Post-training corpora

`configs/posttraining_corpora.json` lists the 21 instruction-tuning and preference datasets, the split and text fields used for each dataset, and the associated Hugging Face page.

```bash
python data_audit/scripts/run_data_audit.py \
  --config data_audit/configs/posttraining_corpora.json \
  --output-dir data_audit/results/posttraining
```

Use `--only DATASET_ID` to run selected datasets and `--limit N` for a short execution check. Existing per-dataset output is reused; pass `--force` to recompute it.

Hugging Face credentials may be supplied through `HF_TOKEN`. The token is read at runtime and is never written to an output file.

## Outputs

The lexical-audit command writes pair-level counts and probabilities, corpus summaries, category summaries, type summaries, and Wilcoxon signed-rank statistics in JSON and CSV formats. The document-level command writes per-document DiAlign scores and corpus summaries. The OpenAI Summarize Comparisons configuration excludes the ambiguous `mail`/`post` pair, whose non-dialectal uses otherwise distort the aggregate.
