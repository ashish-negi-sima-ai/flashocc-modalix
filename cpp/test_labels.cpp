#include "occupancy_labels.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  try {
    std::vector<float> logits(200*200*16*18,-9.f);
    std::vector<std::uint8_t> expected(200*200*16),actual(expected.size());
    for(std::size_t y=0;y<200;++y)for(std::size_t x=0;x<200;++x)for(std::size_t z=0;z<16;++z) {
      auto* p=logits.data()+((y*200+x)*16+z)*18;
      const auto c=(x*7+y*3+z)%18;p[c]=1.f;
      // Some cells have equal maxima in different NEON lanes/groups, including
      // the final two scalar channels. The lowest index must win.
      if((x+y+z)%5==0){p[1]=p[4]=p[17]=1.f;}
      expected[(x*200+y)*16+z]=static_cast<std::uint8_t>(std::max_element(p,p+18)-p);
    }
    for(auto threads:{1,8,16}) {
      flashocc::OccupancyLabels selector(threads);
      for(int repeat=0;repeat<3;++repeat) {
        selector.run(logits,actual);
        if(actual!=expected)throw std::runtime_error("Argmax/tie/layout comparison failed");
        std::fill(actual.begin(),actual.end(),255);
      }
    }
    std::cout<<"Occupancy label selection, tie handling, layout and repeat tests passed.\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
