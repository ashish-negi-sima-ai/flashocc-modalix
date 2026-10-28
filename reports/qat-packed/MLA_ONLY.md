# Direct MLA ELF profiling: sequential versus packed-3

Measured on Modalix `10.42.0.102` on 2026-10-28. The test extracted the original
ELFs from the deployed QAT camera archives and the compatible QAT BEV archive,
then ran them directly with the installed `mla-rt` revision `5382955`.
**AppComplex was disabled and stopped throughout profiling.** The MLA clock
remained at **1 GHz**. Neither model archives nor installed runtime libraries
were changed.

## Whole-ELF hardware execution

Each ELF had three independent trials of 1,000 executions, with the model order
reversed in the middle trial. Values below are the means of `mla-rt`'s
**`tile.lat` hardware execution timer**, with no performance counters requested.
No warmup samples were removed. Loading, initialization and process startup are
outside this timer. Inputs used the CLI's default allocation without an IFM
file; this is a performance test, not an accuracy evaluation.

| Measurement | Sequential camera | Packed-3 camera |
| --- | ---: | ---: |
| Cameras represented by one ELF execution | 1 | 3 |
| Hardware time per ELF execution | **1.283 ms** | **4.467 ms** |
| Executions per six-camera update | 6 | 2 |
| Camera hardware time per six views | **7.698 ms** | **8.934 ms** |
| Shared BEV ELF hardware time | 4.076 ms | 4.076 ms |
| Camera + BEV hardware time per update | **11.774 ms** | **13.010 ms** |

The per-update rows multiply the separately measured per-ELF means by the
required call counts; they are not measurements of a complete application loop.
Packing uses **16.06% more camera MLA execution time** despite requiring fewer
calls. The common BEV network has the same ELF in both configurations.

For comparison, direct CLI API latency was 1.380 ms per sequential-camera call,
4.539 ms per packed-3 call and 4.148 ms per BEV call. These are separate from
`tile.lat` and include host runtime work. They are also different from the Neat
camera-route and BEV-route timings measured inside the full application.

## Why packing does not reduce MLA time

An additional 1,000-execution run per ELF collected hardware counters
`tile_active`, `sync_frz`, `iq_sync_frz` and `d_sync_frz` using `-e 2 -e 5 -e 6 -e 7`.
The main latency table above uses runs without these extra counters.

| Counter, scaled to six camera views | Sequential | Packed-3 |
| --- | ---: | ---: |
| Active tile execution | 7.389 ms | 7.780 ms |
| Synchronization-freeze counter | 0.302 ms | 1.148 ms |

The packed network has more active execution work and substantially more
synchronization stalls. The packed compiler graph contains split convolution
segments, slices, concatenations, broadcasts and multiplies, and processes the
guard columns between cameras. Those structures are consistent with the extra
work and stalls, but this experiment does not attribute the stalls to an
individual memory transaction or prove which graph rewrite is responsible.
The individual freeze counters can overlap; do not add them together.

The compiler YAML's final cycle positions are 1,231,529 for the sequential
camera, 3,890,019 for packed-3 and 3,638,407 for BEV. At 1 GHz these correspond
closely to measured active execution time. Actual hardware latency is higher
because it includes stalls. Compiler cycle counts are not complete application
or runtime latency measurements.

## Per-layer profiles

Each ELF also had three separate `-p` profiling runs using its matching compiler
YAML. All returned layer IDs, names and cycle boundaries matched the compiler
inputs. The profiles contain **81 sequential-camera**, **169 packed-camera** and
**76 BEV** executable layer records, excluding placeholders. These counts describe
the compiled graphs at different levels of splitting, not a direct count of
equivalent source-model operations.

Per-layer profiling pauses/resumes execution at layer boundaries. Its summed
times include profiling effects and should not replace uninterrupted `tile.lat`
in the comparison. Per-layer means, counters, individual trial statistics,
model/ELF hashes and raw-evidence hashes are retained in
[`mla-only-20261028.json`](mla-only-20261028.json). Raw logs, runtime YAML files
and the complete unaggregated results are local under
`artifacts/mla-only-20261028/validated/`.

## Relationship to the full application

EV74 is outside this ELF-only test. In the earlier visualization-disabled app
benchmark, the complete camera stage took 31.48 ms with sequential EV74 and
32.20 ms with packed-3 EV74. Those include preprocessing, tensor boundaries,
runtime coordination, postprocessing and output copies. This measurement
isolates only the MLA portion: 7.70 ms versus 8.93 ms.

Reducing the number of MLA calls can reduce call overhead, while the compiled
packed graph still requires more hardware execution. The direct results explain
why fewer calls alone do not guarantee a faster application. They do not identify
the cost of each remaining Neat or EV74 stage.

## Reproduction and service restoration

Run the new helper on an idle Modalix from the shared repository directory.
The device's system Python needs PyYAML and passwordless `sudo`:

```bash
ssh sima@10.42.0.102
cd /workspace/GitHub/occupancy-prediction
python3 scripts/benchmark_mla_elf.py \
  --camera artifacts/compile_with_qat_better_accuracy/flashocc_m0_camera_mpk.tar.gz \
  --packed-camera artifacts/compile_with_qat_better_accuracy/compile-qat-packed3/flashocc_m0_camera_mpk.tar.gz \
  --bev artifacts/qat-app/compatible/flashocc_m0_bev_mpk.tar.gz \
  --output-dir artifacts/mla-only-repeat \
  --samples 1000 --repeats 3 --layer-repeats 3
```

Use a fresh output directory. The helper extracts only ELF and matching stats
files, records hashes, disables/stops `simaai-appcomplex.service`, and saves raw
logs and structured results. Representative commands it issues are:

```bash
# Uninterrupted hardware execution and direct API statistics.
sudo mla-rt -a mod,a65 -m 1000 -vs:usec:3 \
  --stats whole.yaml flashocc_m0_camera_stage1_mla.elf

# Separate per-layer run, using a copy of the archive's compiler YAML.
sudo mla-rt -a mod,a65 -vs:usec:3 \
  -p layers.yaml flashocc_m0_camera_stage1_mla.elf
```

The helper restores the original AppComplex enablement and active state in a
`finally` block. Initial profiling exposed a restart failure in the service's
memory-initialization step, which uses `-N` to avoid resetting the MLA. Running
the standard initialization ELF with a reset recovered it. The helper now does
that initialization before restarting an originally active AppComplex service
and checkpoints measurements before restoration.

**Final state: AppComplex active and enabled, MLA still at 1 GHz.** After
restoration, the EV74 synthetic preprocessing checker passed and a complete
packed-camera → ARM pooling → BEV replay passed with finite-logit checking.
Its occupancy labels were byte-identical to the earlier run on the same input.
All benchmark and validation processes have exited.
