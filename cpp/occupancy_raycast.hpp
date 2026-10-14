#pragma once
#include <array>
#include <cstdint>
#include <span>

namespace flashocc {
// Labels use upstream X,Y,Z order. Directions are in the same key-ego frame.
int first_occupied_voxel(std::span<const std::uint8_t> labels,
                         const std::array<float,3>& origin,
                         const std::array<float,3>& direction,
                         float near_depth=1.f);
}
