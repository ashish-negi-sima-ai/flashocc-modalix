# Cleanup audit, 2026-10-27

The frozen [reference comparison](../../benchmarks/flashocc_20261027/README.md)
was preserved before cleanup. This audit describes fresh validation of the
cleaned application using assets restored into an independent directory.
No commit, push, public release or Git history rewrite was performed.

## Source cleanup

1. Dataset preparation now supports mini and trainval, official scene splits,
   streamed metadata, exact token selections and Occ3D semantic/visibility arrays.
   All 40 original mini tokens were preserved. One real labeled sample was
   rebuilt with every image, geometry, pooling-rank, label and mask array equal.
2. The root README is a current quickstart. Historical investigations are under
   `docs/experiments`, with compatibility links/stubs from the old locations.
3. All 24 generated QAT outputs were removed from the source tree after verifying
   their retained copies. The six model binaries total 111.13 MiB. Models live
   under ignored `artifacts/`, indexed by `models/manifest.json`; small reports
   are under `reports/qat`. Full compiler IR remains generated local data.
4. The complete comparison now has a portable accuracy/performance/summarize
   runner. It accepts an independent asset root, separate result/build paths,
   explicit runtime settings and an optional Playwright interpreter. It refuses
   to mix a new workload with an existing component's results.
5. Device and Insight endpoints come from local configuration, environment or
   CLI. Model presets are workspace-relative. The normal build path is
   `build/modalix`, separate from the protected reference binaries. Host 3D
   links and printed addresses are configurable.
6. Camera batch/packing investigations and their runtime patch are grouped under
   `experiments/`; old Python entry points retain CLI/import compatibility and
   CMake keeps experimental target names.
7. Environments, legacy run outputs, checkpoints, local configuration and normal
   builds are ignored. CPU metric tests no longer import the SiMa QAT training
   package; site-specific Slurm module/partition/GPU constraints are optional.
8. Archived raw tensors and regenerable frame caches were pruned only after
   archive checks and the independent reproduction below.

## Independent reproduction

All 109 required files were archived, restored into a separate directory and
verified by SHA-256. Original FP32 predictions and all three compiled model
pairs were regenerated with the rebuilt ARM64 application on Modalix. All 32
replay input files matched the original serialization byte-for-byte.
All 128 FP32/compiled prediction-label files also match the originals
byte-for-byte, including cells outside the evaluation mask.

| Accuracy row | Reference mIoU | Rerun mIoU | Full confusion matrix |
| --- | ---: | ---: | --- |
| Original FP32 | 34.5818% | 34.5818% | Exact match |
| Original BF16-heavy | 34.6465% | 34.6465% | Exact match |
| BF16 camera + small mixed BEV | 33.5717% | 33.5717% | Exact match |
| QAT pair | 31.8731% | 31.8731% | Exact match |

Performance retained six 12 FPS inputs, six sequential camera calls, overlays,
2 Hz snapshots, 256 updates and 64 excluded warmups. Main columns disable the
logit scan. Both mean stage times and wall-clock throughput were regenerated.

| Configuration | Reference FPS | Rerun FPS | Change |
| --- | ---: | ---: | ---: |
| Previous + ARM | 4.476 | 4.461 | −0.33% |
| Previous + EV74 | 6.504 | 6.537 | +0.50% |
| QAT + ARM | 5.033 | 5.050 | +0.34% |
| QAT + EV74 | 8.305 | 8.340 | +0.42% |
| QAT + EV74, scan enabled | 6.378 | 6.436 | +0.92% |

The baseline was not updated to these new timings. Raw rerun outputs remain in
`artifacts/cleanup-validation/reproduction/`; compact results, environment and
source/runtime fingerprints are preserved in this report directory.

The normal `build/modalix` build and workspace-relative QAT preset also passed
a separate 64-update smoke with the logit scan enabled and zero invalid camera
markers. That check found and fixed relative input-directory forwarding over
SSH; benchmark config paths are now made absolute before device execution.

## Retained archives and disk cleanup

Local asset storage, outside the Git repository:

- `/workspace/benchmark-archives/flashocc_20261027-assets.tar.gz` and its JSON
  inventory/checksum sidecar. Restore using `scripts/comparison_assets.py`.
- `/workspace/benchmark-archives/flashocc-legacy-replay-tensors.tar.gz` and its
  JSON inventory/checksum sidecar. This restores 64 old raw tensor dumps to
  their original workspace-relative paths; use `tar -xzf ARCHIVE -C ROOT`
  after checking the archive hash in `archived-tensors.json`.

887.2 MiB of legacy raw tensors were archived and every payload read back and
verified. An additional 942.2 MiB of duplicate/regenerable serialized frames
were pruned after all 32 were regenerated identically. Total intermediates
pruned: **1,829.4 MiB (1.79 GiB)**. The pinned source NPZs, models, videos,
geometry, output labels and numeric benchmark evidence remain intact.
Old calibration sets and other unique experiments were retained.

The temporary independently restored asset cache was removed after this
validation; the immutable bundle remains available. A public release URL has
not been published. Existing binary blobs remain in prior Git history.

## Checks and limits

Passed: six metadata/runtime configuration tests; three QAT CPU tests; three
viewer HTTP/snapshot tests; ARM64 application builds; Modalix NEON pooling,
argmax, raycasting and stream/snapshot correctness tests; active document links;
Python/shell syntax; frozen-record recalculation; asset checksum verification;
and fresh accuracy/full-app reproduction.

The full raw trainval dataset is not local. Trainval metadata selection and
streaming have synthetic tests; the real mini path and all 32 supplied prepared
inputs were validated. Full trainval preparation, GPU QAT training and model
recompilation were not rerun for this cleanup. Compiled model/ELF identities and
operator computations were retained.
