# Pretraining word frequencies

Download the frequency archive from [Google Drive](https://drive.google.com/file/d/1ZvPFcnTTXbQHDNMmkS86AiGD7GuYdN9E/view?usp=sharing), extract it, and arrange the files as follows:

```text
data_audit/external/pretraining_word_frequencies/
  BookCorpus/word_frequency.json
  Common Crawl/word_frequency.json
  Dolma/word_frequency.json
  RedPajama/word_frequency.json
  RefinedWeb/word_frequency.json
  Wikipedia/word_frequency.json
```

Run the audit from the repository root:

```bash
python data_audit/scripts/run_data_audit.py \
  --config data_audit/configs/pretraining_corpora.json \
  --output-dir data_audit/results/pretraining
```

The large frequency files are not included in the repository.
