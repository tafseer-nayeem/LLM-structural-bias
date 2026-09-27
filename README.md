# LLM Structural Bias

This repository contains the data, representation, and dialectal-alignment analyses accompanying the paper.

Paper:

```text
Which English Do LLMs Prefer? Triangulating Structural Bias Towards American English in Foundation Models
```

arXiv: `2604.04204`

- `data_audit/` contains lexical corpus audits and document-level DiAlign analysis for pretraining data, together with the post-training audit.
- `representation/tokenization/` measures tokenizer fertility and produces the token-length figures.
- `representation/contextual_asymmetry/` contains the sentence-level semantic-equivalence and prediction-cost experiments.
- `dialign/` provides the shared DiAlign implementation used for pretraining documents and generated text.
- `resources/` contains a tiny local example resource and a helper for downloading the full AmE--BrE variant resource from Hugging Face.

Each package has its own requirements and instructions. Commands in the documentation are written relative to the repository root.

## Full AmE--BrE resource

The full 1,813-pair AmE--BrE variant resource is released separately as a Hugging Face dataset:

```text
ame-bre-structural-bias
```

After creating or cloning the dataset repo under your Hugging Face namespace, download it into the code repository with:

```bash
pip install huggingface_hub
python resources/fetch_ame_bre_resource.py \
  --repo-id YOUR_USERNAME/ame-bre-structural-bias
```

This writes:

```text
resources/AmE_BrE_variations.json
```

Paper-scale scripts use that path by default. The local checks below use the small example resource and do not require network access.

## Quick checks

```bash
bash data_audit/scripts/run_offline_checks.sh
bash representation/tokenization/scripts/run_example_checks.sh
PYTHONPATH=dialign/src python -m unittest discover dialign/tests
```

The checks use local example data and do not download a corpus or model.

## Authentication

Some Hugging Face datasets and tokenizers require accepted access. Provide credentials through the environment rather than placing a token in the repository:

```bash
export HF_TOKEN="your_token"
```

For full representation reruns, gated Hugging Face model access may also be
configured with the CLI:

```bash
hf auth login
hf auth whoami
```

## Citation

```bibtex
@misc{nayeem2026whichenglish,
  title        = {Which English Do LLMs Prefer? Triangulating Structural Bias Towards American English in Foundation Models},
  author       = {Nayeem, Mir Tafseer and Rafiei, Davood},
  year         = {2026},
  eprint       = {2604.04204},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL}
}
```
