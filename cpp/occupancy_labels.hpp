#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace flashocc {
// M0 logits are Y,X,Z*18; labels are X,Y,Z. First class wins ties.
class OccupancyLabels {
 public:
  explicit OccupancyLabels(std::size_t threads=8);
  ~OccupancyLabels();
  OccupancyLabels(const OccupancyLabels&)=delete;
  OccupancyLabels& operator=(const OccupancyLabels&)=delete;
  void run(std::span<const float> logits,std::span<std::uint8_t> labels);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace flashocc
