#include "bev_pool.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace flashocc {
bool neon_available() noexcept {
#if defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

PoolPlan::PoolPlan(std::span<const std::int32_t> rd,
                   std::span<const std::int32_t> rf,
                   std::span<const std::int32_t> rb,
                   std::size_t dv, std::size_t fr, std::size_t n, std::size_t c)
    : depth_ranks(rd.begin(), rd.end()), feature_ranks(rf.begin(), rf.end()),
      depth_values(dv), feature_rows(fr), cells(n), channels(c) {
  if (!cells || !channels || channels > 64 || cells > std::numeric_limits<std::int32_t>::max() ||
      feature_rows > std::numeric_limits<std::size_t>::max()/channels ||
      cells > std::numeric_limits<std::size_t>::max()/channels)
    throw std::invalid_argument("Invalid pool dimensions (channels must be 1..64)");
  if (rd.size() != rf.size() || rd.size() != rb.size())
    throw std::invalid_argument("Rank lengths differ");
  offsets.assign(cells + 1, 0);
  for (std::size_t i = 0; i < rb.size(); ++i) {
    if (rd[i] < 0 || static_cast<std::size_t>(rd[i]) >= depth_values ||
        rf[i] < 0 || static_cast<std::size_t>(rf[i]) >= feature_rows ||
        rb[i] < 0 || static_cast<std::size_t>(rb[i]) >= cells ||
        (i && rb[i] < rb[i-1]))
      throw std::invalid_argument("Out-of-range or unsorted ranks");
    ++offsets[static_cast<std::size_t>(rb[i]) + 1];
  }
  for (std::size_t cell = 1; cell <= cells; ++cell)
    offsets[cell] += offsets[cell-1];
}

std::vector<std::int32_t> depth_ranks_nhwc(std::span<const std::int32_t> ranks,
                                         std::size_t bins, std::size_t pixels) {
  if (!bins || !pixels || bins > std::numeric_limits<std::int32_t>::max()/pixels)
    throw std::invalid_argument("Invalid depth dimensions");
  const auto per_camera = bins*pixels;
  std::vector<std::int32_t> out;
  out.reserve(ranks.size());
  for (auto rank : ranks) {
    if (rank < 0) throw std::invalid_argument("Negative depth rank");
    const auto r = static_cast<std::size_t>(rank);
    const auto mapped = (r/per_camera)*per_camera + (r%pixels)*bins + (r%per_camera)/pixels;
    if (mapped > std::numeric_limits<std::int32_t>::max())
      throw std::invalid_argument("Depth rank overflow");
    out.push_back(static_cast<std::int32_t>(mapped));
  }
  return out;
}

// Keep the comparison kernel scalar, and prevent implicit multiply-add fusion.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
static void scalar_cell(const PoolPlan& p, std::size_t cell, const float* depth,
                        const float* features, float* out, bool fused) {
  std::array<float,64> sum{};
  for (auto k = p.offsets[cell]; k < p.offsets[cell+1]; ++k) {
    const float d = depth[p.depth_ranks[k]];
    const float* f = features + static_cast<std::size_t>(p.feature_ranks[k])*p.channels;
    for (std::size_t c = 0; c < p.channels; ++c)
      sum[c] = fused ? std::fma(d,f[c],sum[c]) : sum[c] + d*f[c];
  }
  std::copy_n(sum.data(), p.channels, out);
}

static void cell(const PoolPlan& p, std::size_t index, const float* depth,
                 const float* features, float* output, Kernel kernel) {
  float* out = output + index*p.channels;
#if defined(__aarch64__)
  if (kernel != Kernel::scalar && p.channels == 64) {
    // Sixteen vector accumulators remain local until the cell is complete.
    // Contributions stay in their original order for each channel.
    float32x4_t sums[16];
    for (auto& v : sums) v = vdupq_n_f32(0);
    for (auto k = p.offsets[index]; k < p.offsets[index+1]; ++k) {
      const float d = depth[p.depth_ranks[k]];
      const float* f = features + static_cast<std::size_t>(p.feature_ranks[k])*64;
      if (kernel == Kernel::neon_fma) {
#pragma GCC unroll 16
        for (int v = 0; v < 16; ++v)
          sums[v] = vfmaq_n_f32(sums[v], vld1q_f32(f+4*v), d);
      } else {
#pragma GCC unroll 16
        for (int v = 0; v < 16; ++v)
          sums[v] = vaddq_f32(sums[v], vmulq_n_f32(vld1q_f32(f+4*v), d));
      }
    }
    for (int v = 0; v < 16; ++v) vst1q_f32(out+4*v, sums[v]);
    return;
  }
#endif
  scalar_cell(p, index, depth, features, out, kernel == Kernel::neon_fma);
}

struct Pooler::Impl {
  PoolPlan plan;
  Kernel kernel;
  std::vector<std::thread> workers;
  std::vector<std::size_t> boundaries;
  std::mutex dispatch_mutex, run_mutex;
  std::condition_variable ready, done;
  std::size_t generation = 0, finished = 0;
  bool stop = false;
  const float* depth = nullptr;
  const float* features = nullptr;
  float* output = nullptr;

  Impl(PoolPlan p, std::size_t threads, Kernel k) : plan(std::move(p)), kernel(k) {
    if (!threads || threads > 256) throw std::invalid_argument("threads must be 1..256");
    threads = std::min(threads, plan.cells);
    partition(threads);
    if (threads > 1) {
      try {
        for (std::size_t t=0; t<threads; ++t) workers.emplace_back([this,t] { work(t); });
      } catch (...) { shutdown(); throw; }
    }
  }
  void partition(std::size_t threads) {
    boundaries.assign(threads+1,0);
    // Prefix cost: contributions plus one unit per cell for output writes.
    const auto cost = plan.depth_ranks.size() + plan.cells;
    boundaries[threads] = plan.cells;
    for (std::size_t t = 1; t < threads; ++t) {
      const auto target = (cost/threads)*t + ((cost%threads)*t)/threads;
      std::size_t lo = boundaries[t-1], hi = plan.cells;
      while (lo < hi) {
        const auto mid = lo + (hi-lo)/2;
        if (plan.offsets[mid]+mid < target) lo=mid+1; else hi=mid;
      }
      boundaries[t] = lo;
    }
  }
  void shutdown() {
    { std::lock_guard lock(dispatch_mutex); stop=true; }
    ready.notify_all();
    for (auto& worker : workers) if (worker.joinable()) worker.join();
  }
  ~Impl() { shutdown(); }
  void range(std::size_t begin, std::size_t end) {
    for (auto i=begin; i<end; ++i) cell(plan,i,depth,features,output,kernel);
  }
  void work(std::size_t t) {
    std::size_t seen=0;
    for (;;) {
      std::unique_lock lock(dispatch_mutex);
      ready.wait(lock,[&] { return stop || generation!=seen; });
      if (stop) return;
      seen=generation;
      lock.unlock();
      range(boundaries[t],boundaries[t+1]);
      lock.lock();
      if (++finished == workers.size()) done.notify_one();
    }
  }
};

Pooler::Pooler(PoolPlan p, std::size_t threads, Kernel kernel)
    : impl_(std::make_unique<Impl>(std::move(p),threads,kernel)) {}
Pooler::~Pooler() = default;
void Pooler::update_plan(PoolPlan plan) {
  auto& s=*impl_;
  std::lock_guard serial(s.run_mutex);
  std::lock_guard lock(s.dispatch_mutex);
  s.plan=std::move(plan);
  s.partition(s.workers.empty()?1:s.workers.size());
}
void Pooler::run(std::span<const float> depth, std::span<const float> features,
                 std::span<float> output) {
  auto& s=*impl_;
  std::lock_guard serial(s.run_mutex);
  if (depth.size()!=s.plan.depth_values || features.size()!=s.plan.feature_rows*s.plan.channels ||
      output.size()!=s.plan.cells*s.plan.channels)
    throw std::invalid_argument("Tensor sizes do not match plan");
  const auto overlap=[](const float* a,std::size_t an,const float* b,std::size_t bn) {
    if (!an || !bn) return false;
    const auto av=reinterpret_cast<std::uintptr_t>(a),bv=reinterpret_cast<std::uintptr_t>(b);
    return av<=bv ? bv-av<an*sizeof(float) : av-bv<bn*sizeof(float);
  };
  if (overlap(output.data(),output.size(),depth.data(),depth.size()) ||
      overlap(output.data(),output.size(),features.data(),features.size()))
    throw std::invalid_argument("Output aliases input");
  std::unique_lock lock(s.dispatch_mutex);
  s.depth=depth.data(); s.features=features.data(); s.output=output.data();
  if (s.workers.empty()) { s.range(0,s.plan.cells); return; }
  s.finished=0; ++s.generation;
  s.ready.notify_all();
  s.done.wait(lock,[&] { return s.finished==s.workers.size(); });
}
}  // namespace flashocc
