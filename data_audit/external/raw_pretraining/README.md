# Raw pretraining documents

The document-level DiAlign audit streams Wikipedia, C4, RefinedWeb, and RedPajama from Hugging Face. The original BookCorpus and Dolma analyses used local documents. Place those files at:

```text
data_audit/external/raw_pretraining/
|-- BookCorpus/
|   `-- downloaded Hugging Face dataset files
`-- Dolma/
    `-- v1_5r2_sample-*.json.gz
```

Place the downloaded `bookcorpus/bookcorpus` repository contents in the BookCorpus directory; it is loaded in the same way as the original analysis and must expose a `text` field. Each Dolma JSON record must also have a `text` field and may include its usual metadata. These files are inputs and are not included in the repository.
