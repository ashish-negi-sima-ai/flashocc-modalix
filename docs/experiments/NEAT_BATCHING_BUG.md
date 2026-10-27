# Neat batching bug: counting the batch size twice

**Finding:** The inspected `develop` code still contains the output-conversion
sizing bug found with FlashOCC's batch-six camera model. Neat supplies the size
of six cameras where the conversion kernel expects the size of one camera.
The kernel multiplies that size by six again.

## A simple example

Imagine one camera produces **100 bytes**, and we process **six cameras**:

| Calculation | Result |
| --- | ---: |
| Correct: 100 bytes per camera × 6 cameras | **600 bytes** |
| Bug: tell the kernel that one camera needs 600 bytes, then multiply by 6 | **3,600 bytes** |

The memory contains the correct amount of data. The description of that memory
is wrong, so the runtime believes the buffer is too small and rejects it.

## What happens in FlashOCC

The camera network produces two outputs: depth probabilities and image features.
Its batch-six model pack specifies these physical input sizes for output conversion:

| Output | Bytes per camera | Bytes for six cameras |
| --- | ---: | ---: |
| Depth | 67,584 | 405,504 |
| Features | 90,112 | 540,672 |
| **Combined MLA output buffer** | **157,696** | **946,176** |

For the first output, the conversion descriptor incorrectly contains **405,504
bytes per camera**. The conversion code then calculates:

```text
Correct depth requirement:  67,584 × 6 =   405,504 bytes
Incorrect requirement:    405,504 × 6 = 2,433,024 bytes
Actual combined buffer:                  946,176 bytes
```

This explains the earlier runtime error:

```text
graph=detesscast role=input tensor=0
required_bytes=2433024 actual_bytes=946176
```

The first output really needs 405,504 bytes within the combined 946,176-byte
buffer. The incorrect requirement exceeds the entire buffer.

`detesscast` is the stage that rearranges the MLA's tiled output into a normal
tensor and converts BF16 values to FP32. The failure occurs at this output
conversion boundary, before conversion executes.

## Where the mismatch remains

| Code | What it does |
| --- | --- |
| [Core: descriptor geometry](https://github.com/sima-neat/core/blob/0297707f65493b8ab20710eb58a8e5b968dfb9e2/src/pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.cpp#L7295) | Builds kernel descriptors from a shape that still contains the full batch dimension. |
| [Core: descriptor byte count](https://github.com/sima-neat/core/blob/0297707f65493b8ab20710eb58a8e5b968dfb9e2/src/pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.cpp#L7390) | Copies the full batched transport size into `storage.nbytes`, which is used as a per-camera size. |
| [Internals: buffer sizing](https://github.com/sima-neat/internals/blob/a23678818ddf7d1e410c02ff9032c4763c890788/sima-ai-cvu-sw/graphs/detesscast/config/detesscast_buff_sz.cc#L121) | Multiplies the descriptor's byte count by the batch size. |

Internals expects a per-camera descriptor plus a separate batch count. Core's
fused `detesscast` route does not consistently supply that contract.

## What was checked

Checked on **16 October 2026**:

| Repository | Inspected `develop` commit | Remote verification |
| --- | --- | --- |
| Core | `0297707f` | Fetched successfully |
| Internals | `a2367881` | Local checkout; remote fetch failed authentication |

A host C++ test called the actual Internals buffer-sizing functions with values
from FlashOCC's batch-six MPK manifest. It reproduced the **2,433,024-byte**
requirement. Supplying per-camera sizes produced the correct **405,504-byte**
requirement; the batch-one control also passed.

**Validation scope:** source inspection and a host buffer-sizing check. These
`develop` branches were not rebuilt and tested on Modalix. Neither repository
nor the installed runtime was patched during this check.

Evidence: [audit report](../../artifacts/neat-develop-batch-audit/report.json),
[C++ check](../../artifacts/neat-develop-batch-audit/check_detesscast_sizes.cpp), and
[test output](../../artifacts/neat-develop-batch-audit/sizing-check.log).

## Fix direction and impact

Core should give the conversion kernel **one-camera geometry and physical storage
sizes**, retain `batch_size=6`, and preserve full-batch sizes and offsets in the
routing metadata. The resulting change needs regression tests and hardware
validation.

Batch one avoids the extra multiplication. The current spatially packed model
also keeps network batch size one, so it avoids this failure.

The compiler's MaxPool padding change is a **separate accuracy issue**. Fixing
Neat's buffer sizing alone does not fix that compiler behavior. See the
[earlier batch investigation](../../app/CAMERA_BATCH.md#follow-up-correctness-diagnosis).
