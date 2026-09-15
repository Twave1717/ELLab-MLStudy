# Official 2SFS dataset snapshot

The files in `upstream/` are copied without local modifications from:

- Repository: <https://github.com/FarinaMatteo/rethinking_fewshot_vlms>
- Commit: `64ac143c7d22803bfeeaadfe42f570220ab29b06`
- Source directory: `datasets/`

The upstream project is MIT licensed. See `LICENSE` in this directory.

Project-specific integration belongs in `adapter.py`; do not edit the upstream
snapshot. The snapshot expects datasets in the CoOp directory layout, including
the `split_zhou_*.json` files, and the public 2SFS JSONL few-shot splits under
each dataset's `split_fewshot/` directory.

`split_catalog.json` locks the SHA-256 digests of the public 1/2/4/8/16-shot,
seed 1/2/3 manifests, their five source archives, the CoOp split JSON files,
and FGVC's native annotations. The files come from the official links in the
2SFS and CoOp READMEs. Use `python -m datasets.official_2sfs.prepare` to
download, verify, and install the public files.
