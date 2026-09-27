# DiAlign

This shared package implements the DiAlign score used for document-level pretraining analysis and generated-text analysis. It scores unique 2--5-grams using their relative frequencies in the Google Books American and British English corpora.

For an n-gram \(g\), the signed log ratio and divergence are

```text
r(g) = log2(f_AmE(g) / f_Ref-BrE(g))
d(g) = |f_AmE(g) - f_Ref-BrE(g)| / (f_AmE(g) + f_Ref-BrE(g)).
```

The weight is `d(g)` multiplied by `1.5` when the n-gram contains a curated variant marker. Its contribution is `abs(r(g)) * weight`. Positive and negative contributions are accumulated separately and normalized to sum to one.

The Google Books request is case-insensitive. When the API returns separate capitalization variants together with an aggregate `(All)` series, the aggregate series is used once; capitalization variants are not averaged again.

## Setup

From the repository root:

```bash
pip install -r dialign/requirements.txt
python -m nltk.downloader stopwords averaged_perceptron_tagger_eng maxent_ne_chunker_tab words
```

NLTK releases before 3.9 may use `averaged_perceptron_tagger` and `maxent_ne_chunker` instead.

## Score text

```bash
python dialign/scripts/run_dialign.py \
  --text "The programme was held in the city centre."
```

Frequencies are stored in `dialign/.cache/ngram_frequencies.sqlite3`. The local store is shared by every analysis that uses DiAlign, avoids repeated requests, and is not committed to the repository. Transport failures raise an error rather than being recorded as zero-frequency observations.

The reported method treats repeated occurrences of the same n-gram within one text as one piece of evidence. Results include the number of candidate n-grams and the number with nonzero directional contribution. The returned top contributors are diagnostic details and are not reused as global weights.
