# FlashOCC CPU BEV pooling

`bev_pool.hpp` / `bev_pool.cpp` provide a standalone C++20 library. There are no
Neat or OpenMP dependencies. `PoolPlan` validates sorted ranks and builds cell
intervals; `Pooler` owns persistent worker threads. Work is divided by contribution
count plus output-write cost. Each worker exclusively owns complete output cells,
so no atomic floating-point operations or locks are needed inside the kernel.

The optimized kernel accumulates 64 channels with sixteen NEON vector registers.
`Kernel::neon` is the default and uses separate multiplication/addition, preserving
reference FP32 rounding. `Kernel::neon_fma` explicitly uses fused multiply-add.
Other channel counts up to 64 use a scalar fallback. Compile with
`-ffp-contract=off` as the supplied CMake target does.

## Tensor contract

- Depth: flat float32 values, with ranks addressing that exact buffer layout.
- Camera features: contiguous float32 `[6,16,44,64]` (NHWC).
- Output: contiguous float32 `[1,200,200,64]` (NHWC), including zeros in empty cells.
- Rank arrays: equally sized int32 arrays, sorted by BEV destination.
- Output must not overlap either input; the library checks this.

Upstream `ranks_depth` address NCHW depth, whereas SDK camera outputs use NHWC
`[6,16,44,44]`. Remap depth ranks once when preparing the geometry:

```cpp
#include "bev_pool.hpp"

auto rd_nhwc = flashocc::depth_ranks_nhwc(ranks_depth, 44, 16 * 44);
flashocc::PoolPlan plan(rd_nhwc, ranks_feat, ranks_bev,
                       depth_nhwc.size(), 6 * 16 * 44);
flashocc::Pooler pooler(std::move(plan), 8);  // Default: non-fused NEON.
std::vector<float> bev(200 * 200 * 64);
pooler.run(depth_nhwc, features_nhwc, bev);
```

Keep the `Pooler` alive across frames. If motion compensation or calibration
changes ranks, call `update_plan(new_plan)` between frames; workers are retained.
Plan updates and concurrent `run` calls are serialized. Buffer lifetimes must
extend through the call. The implementation does not pin threads or change CPU
frequency. Use the channel-last output directly for the compiled BEV stage.

## Build and run

In this SDK workspace, ARM64 cross-build:

```bash
cmake -S cpp -B artifacts/build-pool-arm64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++-12
cmake --build artifacts/build-pool-arm64 --parallel 4
ssh sima@10.42.0.73 '/workspace/GitHub/occupancy-prediction/artifacts/build-pool-arm64/pool_bench --self-test'
```

The normal `dk` launcher could not chmod the host-owned binary on the shared NFS
mount. Direct SSH invocation succeeded using the existing executable permissions.

Prepare fixtures on the host and run the benchmark on the device:

```bash
source /home/ashishnegi/.bashrc
activate-model-compiler
python scripts/export_pool_fixtures.py
mkdir -p artifacts/pool-device-final
chmod 777 artifacts/pool-device-final
ssh sima@10.42.0.73 'python3 /workspace/GitHub/occupancy-prediction/scripts/benchmark_pool.py --binary /workspace/GitHub/occupancy-prediction/artifacts/build-pool-arm64/pool_bench --fixtures /workspace/GitHub/occupancy-prediction/artifacts/pool-fixtures --output /workspace/GitHub/occupancy-prediction/artifacts/pool-device-final --iterations 100'
python scripts/validate_pool_device.py
```

The writable output directory allows the device's different NFS UID to save its
results. Source, binaries, input fixtures, and results stay in the shared workspace.

For native host tests, explicitly clear SDK compiler/linker flags:

```bash
cmake -S cpp -B artifacts/build-pool-native -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/x86_64-linux-gnu-g++-13 -DCMAKE_CXX_FLAGS= -DCMAKE_EXE_LINKER_FLAGS=
cmake --build artifacts/build-pool-native --parallel 4
ctest --test-dir artifacts/build-pool-native --output-on-failure
```

## Measured on Modalix, 2026-10-14

Device `10.42.0.73`: Cortex-A65, eight physical cores, sixteen logical CPUs.
Eight real nuScenes holdout samples, each with approximately 151,000–152,000
contributions and 40,000 output cells. Each configuration used ten warmups and
100 timed iterations per sample.

| Kernel | Threads | Median of sample medians | Worst sample p95 |
| --- | ---: | ---: | ---: |
| Scalar | 1 | 59.96 ms | 63.44 ms |
| NEON | 1 | 20.29 ms | 21.03 ms |
| NEON | 4 | 5.72 ms | 6.13 ms |
| NEON | 8 | 3.26 ms | 5.08 ms |
| NEON | 16 | 3.21 ms | 4.14 ms |
| NEON FMA | 16 | 3.19 ms | 4.10 ms |

These are pooling timings including worker dispatch and all output writes. They
exclude input IO, rank generation, plan setup, tensor conversion, and model
inference. Initial plan/pooler setup was approximately 4–5 ms. Eight threads give
almost the same median as sixteen; full application scheduling should determine
the final thread count.

All 88 fixture/kernel/thread configurations passed. Scalar and non-fused NEON
matched every NumPy reference value exactly. FMA maximum absolute difference was
`3.05176e-5`, passing `atol=1e-4`, `rtol=1e-4`. Device FMA outputs from eight-thread
runs were fed through the host FP32 BEV model: all eight samples produced exactly
the same final labels as their original reference. This checks CPU pooling on
hardware; it does not run either neural network on MLA.

Self-tests passed both natively and on Modalix: published upstream fixture,
empty/stale outputs, rank/shape errors, input/output aliasing, repeated dispatch,
geometry updates, channel fallback, and concurrent-call serialization. Native
AddressSanitizer and UndefinedBehaviorSanitizer tests also passed.

Results: `artifacts/pool-device-final/summary.json` (binary/source hashes and raw
timings), and `artifacts/pool-device-validation.json` (downstream label checks).
