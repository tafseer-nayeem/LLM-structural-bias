# Resources

The AmE--BrE variant resource is released as a Hugging Face dataset rather than duplicated in the code archive.

Recommended dataset name:

```text
ame-bre-structural-bias
```

To download the full resource into this directory:

```bash
pip install datasets
python resources/fetch_ame_bre_resource.py \
  --repo-id YOUR_USERNAME/ame-bre-structural-bias
```

This creates:

```text
resources/AmE_BrE_variations.json
```

The runnable paper-scale scripts use this path by default. You can also pass a custom path with the relevant command-line option, for example `--variant-file`.

`example_AmE_BrE_variations.json` is only for local execution checks.
