# Versioned model assets

`manifest.json` records workspace-relative paths, exact SHA-256 hashes and sizes.
The `pairs` section binds the FP32, original BF16-heavy, small mixed and QAT
camera/BEV pairs used by the preserved comparison. Never mix camera and BEV
archives from different quantization experiments without validating the pair.

The `qat` pair is **PTQ on QAT-trained weights**, using the separate BEV
shape-metadata compatibility copy. The QDQ-hint compilation is a distinct
experiment. Its simulator results are kept separately under `reports/qat`.
The original BEV archive and unchanged-ELF compatibility provenance are retained.

The selected runnable QAT camera/compatible BEV pair is now also committed under
[`demo/models`](../demo/README.md), with clips, geometry and a prebuilt executable.
`demo/manifest.json` pins these payloads. Other large files are stored under
ignored `artifacts/`, with an independently restorable copy outside the repository
in the checksummed comparison bundle.
The six generated QAT binaries previously tracked under `qat/runs` are retained
in `artifacts/qat-runs`. Compiler IR and checkpoints are generated outputs.
Small training, export and validation reports remain under `reports/qat`.

```bash
python3 scripts/comparison_assets.py archive --archive /new/storage/comparison-assets.tar.gz
python3 scripts/comparison_assets.py restore \
  --archive /new/storage/comparison-assets.tar.gz --asset-root /empty/restore-directory
python3 scripts/comparison_assets.py verify --asset-root /empty/restore-directory
```

Archive and JSON sidecar are an immutable pair. The cleanup archive currently
lives at `/workspace/benchmark-archives/flashocc_20261027-assets.tar.gz`, outside
the Git repository. It is local model storage; no public release URL is implied.
Keep the original asset hierarchy when restoring. The comparison runner accepts
`--asset-root`, independently of the source checkout and output directory.

Deleting files in a new commit does not remove old binaries from Git history.
This cleanup does not rewrite published history.
