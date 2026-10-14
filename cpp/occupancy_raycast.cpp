#include "occupancy_raycast.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace flashocc {
int first_occupied_voxel(std::span<const std::uint8_t> labels,
                        const std::array<float,3>& origin,
                        const std::array<float,3>& direction,float near_depth) {
  if(labels.size()!=200*200*16)throw std::invalid_argument("Invalid occupancy volume");
  constexpr std::array<float,3> lower{-40,-40,-1},upper{40,40,5.4f};
  constexpr std::array<int,3> size{200,200,16};
  float enter=near_depth,leave=std::numeric_limits<float>::infinity();
  std::array<int,3> cell{},step{};
  std::array<float,3> next{},delta{};
  bool moving=false;
  for(int axis=0;axis<3;++axis) {
    if(!std::isfinite(origin[axis]) || !std::isfinite(direction[axis]))return -1;
    if(std::abs(direction[axis])<1e-8f) {
      if(origin[axis]<lower[axis] || origin[axis]>=upper[axis])return -1;
    } else {
      moving=true;
      float a=(lower[axis]-origin[axis])/direction[axis],b=(upper[axis]-origin[axis])/direction[axis];
      enter=std::max(enter,std::min(a,b));leave=std::min(leave,std::max(a,b));
    }
  }
  if(!moving || enter>=leave)return -1;
  const float start=enter+1e-4f;
  for(int axis=0;axis<3;++axis) {
    const float position=origin[axis]+start*direction[axis];
    cell[axis]=std::clamp(static_cast<int>(std::floor((position-lower[axis])/.4f)),0,size[axis]-1);
    if(std::abs(direction[axis])<1e-8f) {
      step[axis]=0;next[axis]=delta[axis]=std::numeric_limits<float>::infinity();
    } else {
      step[axis]=direction[axis]>0?1:-1;
      const float boundary=lower[axis]+(cell[axis]+(step[axis]>0?1:0))*.4f;
      next[axis]=(boundary-origin[axis])/direction[axis];delta[axis]=.4f/std::abs(direction[axis]);
    }
  }
  for(int iteration=0;iteration<416;++iteration) {
    const int index=(cell[0]*200+cell[1])*16+cell[2];
    if(labels[index]!=17)return index;
    float distance=std::min({next[0],next[1],next[2]});
    if(distance>=leave)return -1;
    for(int axis=0;axis<3;++axis)if(next[axis]<=distance+1e-5f) {
      cell[axis]+=step[axis];next[axis]+=delta[axis];
      if(cell[axis]<0 || cell[axis]>=size[axis])return -1;
    }
  }
  return -1;
}
}
