#include "occupancy_labels.hpp"
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace flashocc {
static std::uint8_t select(const float* values) {
#if defined(__aarch64__)
  float32x4_t best=vld1q_f32(values);
  const std::uint32_t lanes[4]={0,1,2,3};
  uint32x4_t indices=vld1q_u32(lanes);
  for(std::uint32_t base=4;base<16;base+=4) {
    auto next=vld1q_f32(values+base);
    const auto greater=vcgtq_f32(next,best);
    best=vbslq_f32(greater,next,best);
    indices=vbslq_u32(greater,vaddq_u32(vld1q_u32(lanes),vdupq_n_u32(base)),indices);
  }
  float maximum=vmaxvq_f32(best);
  auto matching=vbslq_u32(vceqq_f32(best,vdupq_n_f32(maximum)),indices,vdupq_n_u32(std::numeric_limits<std::uint32_t>::max()));
  std::uint32_t index=vminvq_u32(matching);
  for(std::uint32_t c=16;c<18;++c)if(values[c]>maximum){maximum=values[c];index=c;}
  return static_cast<std::uint8_t>(index);
#else
  return static_cast<std::uint8_t>(std::max_element(values,values+18)-values);
#endif
}

struct OccupancyLabels::Impl {
  std::vector<std::thread> workers;
  std::vector<std::uint8_t> yxz=std::vector<std::uint8_t>(200*200*16);
  std::mutex mutex,serial;
  std::condition_variable ready,done;
  const float* logits=nullptr;
  bool stop=false;
  std::size_t generation=0,finished=0,count;
  explicit Impl(std::size_t threads):count(threads) {
    if(!threads || threads>256)throw std::invalid_argument("Label threads must be 1..256");
    try {
      for(std::size_t t=0;t<count;++t)workers.emplace_back([this,t] {
        std::size_t seen=0;
        for(;;) {
          std::unique_lock lock(mutex);ready.wait(lock,[&]{return stop || generation!=seen;});
          if(stop)return;
          seen=generation;lock.unlock();
          for(std::size_t cell=40000*t/count;cell<40000*(t+1)/count;++cell)
            for(std::size_t z=0;z<16;++z)yxz[cell*16+z]=select(logits+cell*288+z*18);
          lock.lock();if(++finished==count)done.notify_one();
        }
      });
    }catch(...){shutdown();throw;}
  }
  void shutdown() {
    {std::lock_guard lock(mutex);stop=true;}ready.notify_all();
    for(auto& t:workers)if(t.joinable())t.join();
  }
  ~Impl(){shutdown();}
};
OccupancyLabels::OccupancyLabels(std::size_t threads):impl_(std::make_unique<Impl>(threads)){}
OccupancyLabels::~OccupancyLabels()=default;
void OccupancyLabels::run(std::span<const float> logits,std::span<std::uint8_t> labels) {
  if(logits.size()!=200*200*16*18 || labels.size()!=200*200*16)throw std::invalid_argument("Invalid M0 label tensors");
  auto& s=*impl_;std::lock_guard serial(s.serial);
  std::unique_lock lock(s.mutex);s.logits=logits.data();s.finished=0;++s.generation;
  s.ready.notify_all();s.done.wait(lock,[&]{return s.finished==s.count;});lock.unlock();
  for(std::size_t x=0;x<200;++x)for(std::size_t y=0;y<200;++y)
    std::memcpy(labels.data()+(x*200+y)*16,s.yxz.data()+(y*200+x)*16,16);
}
}  // namespace flashocc
