#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace flashocc {
enum class Kernel { scalar, neon, neon_fma };
bool neon_available() noexcept;

// Indices refer to flat depth and contiguous channel-last feature rows.
// BEV ranks must be sorted. Each destination cell appears in one interval.
struct PoolPlan {
  PoolPlan(std::span<const std::int32_t> depth_ranks,
           std::span<const std::int32_t> feature_ranks,
           std::span<const std::int32_t> bev_ranks,
           std::size_t depth_values, std::size_t feature_rows,
           std::size_t cells = 40000, std::size_t channels = 64);
  std::vector<std::int32_t> depth_ranks, feature_ranks;
  std::vector<std::size_t> offsets;
  std::size_t depth_values, feature_rows, cells, channels;
};

// Upstream depth ranks address N,D,H,W; SDK depth outputs use N,H,W,D.
std::vector<std::int32_t> depth_ranks_nhwc(
    std::span<const std::int32_t> ranks, std::size_t bins,
    std::size_t pixels_per_camera);

// Persistent workers and precomputed cost-balanced cell ranges. run() overwrites
// every output cell, including empty cells, without allocations or thread starts.
// Inputs must remain valid until run returns. Output must not alias either input.
// Concurrent run calls on one instance are serialized.
class Pooler {
 public:
  explicit Pooler(PoolPlan plan, std::size_t threads, Kernel kernel = Kernel::neon);
  ~Pooler();
  Pooler(const Pooler&) = delete;
  Pooler& operator=(const Pooler&) = delete;
  // Swap geometry between frames without recreating workers. Serialized with run.
  void update_plan(PoolPlan plan);
  void run(std::span<const float> depth, std::span<const float> features,
           std::span<float> output);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace flashocc
