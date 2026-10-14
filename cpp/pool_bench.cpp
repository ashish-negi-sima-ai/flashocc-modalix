#include "bev_pool.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using flashocc::Kernel;
using flashocc::Pooler;
using flashocc::PoolPlan;

struct Fixture {
  std::size_t cells, channels, feature_rows;
  std::vector<float> depth, features, expected;
  std::vector<std::int32_t> rd, rf, rb;
  PoolPlan plan() const { return PoolPlan(rd,rf,rb,depth.size(),feature_rows,cells,channels); }
};

template<class T> std::vector<T> read(std::ifstream& f, std::size_t count) {
  std::vector<T> out(count);
  if (!f.read(reinterpret_cast<char*>(out.data()),static_cast<std::streamsize>(count*sizeof(T))))
    throw std::runtime_error("Truncated fixture");
  return out;
}

Fixture load(const std::string& path) {
  if constexpr (std::endian::native != std::endian::little)
    throw std::runtime_error("Fixture reader requires little endian");
  std::ifstream f(path,std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open fixture");
  auto magic=read<char>(f,8);
  if (std::memcmp(magic.data(),"FOCCP01\0",8)) throw std::runtime_error("Bad fixture magic");
  auto h=read<std::uint64_t>(f,5);
  for (auto n:h) if (n>100000000) throw std::runtime_error("Fixture exceeds size limits");
  if (!h[2] || !h[3] || h[3]>64) throw std::runtime_error("Invalid fixture dimensions");
  const auto feature_count=h[1]*h[3], output_count=h[2]*h[3];
  if (feature_count>100000000 || output_count>100000000)
    throw std::runtime_error("Fixture tensors exceed size limits");
  const auto bytes=48+4*(h[0]+feature_count+3*h[4]+output_count);
  if (std::filesystem::file_size(path)!=bytes) throw std::runtime_error("Fixture length mismatch");
  Fixture out;
  out.feature_rows=h[1];out.cells=h[2];out.channels=h[3];
  out.depth=read<float>(f,h[0]);out.features=read<float>(f,feature_count);
  out.rd=read<std::int32_t>(f,h[4]);out.rf=read<std::int32_t>(f,h[4]);
  out.rb=read<std::int32_t>(f,h[4]);out.expected=read<float>(f,output_count);
  return out;
}

struct Errors { double max_abs=0; std::size_t different=0, failed=0; };
Errors compare(std::span<const float> actual, std::span<const float> expected) {
  if (actual.size()!=expected.size()) throw std::runtime_error("Output size mismatch");
  Errors e;
  for (std::size_t i=0;i<actual.size();++i) {
    const double d=std::abs(static_cast<double>(actual[i])-expected[i]);
    e.max_abs=std::max(e.max_abs,d);
    e.different+=(actual[i]!=expected[i]);
    e.failed+=(!std::isfinite(actual[i]) || d>1e-4+1e-4*std::abs(expected[i]));
  }
  return e;
}

void self_test() {
  Fixture fixture;
  fixture.cells=4;fixture.channels=64;fixture.feature_rows=4;
  fixture.depth={.3f,.4f,.2f,.1f,.7f,.6f,.8f,.9f};
  fixture.features.assign(4*64,1.f);
  fixture.rd={0,4,1,6};fixture.rf={0,0,1,2};fixture.rb={0,0,1,1};
  fixture.expected.assign(4*64,0);
  std::fill_n(fixture.expected.begin(),64,1.f);
  std::fill_n(fixture.expected.begin()+64,64,1.2f);
  for (auto k:{Kernel::scalar,Kernel::neon,Kernel::neon_fma}) {
    for (auto t:{1,2,4,8}) {
      Pooler p(fixture.plan(),t,k);
      std::vector<float> out(4*64,std::numeric_limits<float>::quiet_NaN());
      for (int repeat=0;repeat<20;++repeat) {
        p.run(fixture.depth,fixture.features,out);
        if (compare(out,fixture.expected).failed) throw std::runtime_error("Published fixture failed");
        std::fill(out.begin(),out.end(),-123.f);
      }
    }
  }
  // Empty contributions must overwrite every cell, including stale output.
  PoolPlan empty({}, {}, {}, 0,0,3,64);
  Pooler empty_pool(empty,2);
  std::vector<float> out(192,17.f);
  empty_pool.run({}, {}, out);
  if (std::any_of(out.begin(),out.end(),[](float v){return v!=0;}))
    throw std::runtime_error("Empty pooling failed");
  // Public contracts: invalid ranks, dimensions, thread count and run shapes.
  auto reject=[](auto operation) {
    try { operation(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Invalid input was accepted");
  };
  reject([&]{Pooler p(fixture.plan(),0);});
  reject([&]{auto copy=fixture;copy.rb[0]=3; auto p=copy.plan(); (void)p;});
  reject([&]{auto copy=fixture;copy.rd[0]=-1; auto p=copy.plan(); (void)p;});
  reject([&]{auto copy=fixture;copy.rf[0]=4; auto p=copy.plan(); (void)p;});
  reject([&]{auto copy=fixture;copy.rd.pop_back(); auto p=copy.plan(); (void)p;});
  reject([&]{empty_pool.run({}, {}, std::span<float>(out).first(3));});
  reject([&]{Pooler p(fixture.plan(),2); p.run(fixture.depth,fixture.features,fixture.features);});
  // Geometry updates reuse the existing workers, including changes in cell count
  // and a non-64-channel scalar fallback.
  Pooler reusable(fixture.plan(),8);
  reusable.update_plan(empty);
  reusable.run({}, {}, out);
  if (std::any_of(out.begin(),out.end(),[](float v){return v!=0;}))
    throw std::runtime_error("Plan update failed");
  fixture.channels=3;fixture.features.assign(12,1.f);fixture.expected.assign(12,0);
  std::fill_n(fixture.expected.begin(),3,1.f);std::fill_n(fixture.expected.begin()+3,3,1.2f);
  reusable.update_plan(fixture.plan());out.assign(12,0);
  reusable.run(fixture.depth,fixture.features,out);
  if (compare(out,fixture.expected).failed) throw std::runtime_error("Channel fallback failed");
  std::atomic<bool> concurrent_failed=false;
  auto caller=[&](float scale) {
    auto depth=fixture.depth;for(auto& value:depth)value*=scale;
    auto expected=fixture.expected;for(auto& value:expected)value*=scale;
    std::vector<float> output(expected.size());
    for(int i=0;i<32;++i) {
      reusable.run(depth,fixture.features,output);
      if(compare(output,expected).failed) concurrent_failed=true;
    }
  };
  std::thread a(caller,1.f), b(caller,2.f);a.join();b.join();
  if(concurrent_failed) throw std::runtime_error("Concurrent run serialization failed");
  const std::vector<std::int32_t> canonical{0,1,2,3,4,5,6,7};
  const auto remapped=flashocc::depth_ranks_nhwc(canonical,2,2);
  if (remapped!=std::vector<std::int32_t>{0,2,1,3,4,6,5,7})
    throw std::runtime_error("Depth layout remapping failed");
  std::cout<<"{\"self_tests\":\"passed\",\"neon_available\":"
           <<(flashocc::neon_available()?"true":"false")<<"}\n";
}

int main(int argc,char** argv) {
  try {
    if (argc==2 && std::string(argv[1])=="--self-test") {self_test();return 0;}
    if (argc<5 || argc>6) {
      std::cerr<<"Usage: pool_bench fixture.pool threads scalar|neon|neon-fma iterations [output.f32]\n";
      return 2;
    }
    const std::string name=argv[3];
    Kernel kernel;
    if (name=="scalar") kernel=Kernel::scalar;
    else if (name=="neon") kernel=Kernel::neon;
    else if (name=="neon-fma") kernel=Kernel::neon_fma;
    else throw std::invalid_argument("Unknown kernel");
    const auto threads=std::stoul(argv[2]), iterations=std::stoul(argv[4]);
    if (!iterations || iterations>100000) throw std::invalid_argument("Invalid iteration count");
    auto fixture=load(argv[1]);
    const auto start=std::chrono::steady_clock::now();
    Pooler pool(fixture.plan(),threads,kernel);
    const double setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::vector<float> output(fixture.cells*fixture.channels);
    for (int i=0;i<10;++i) pool.run(fixture.depth,fixture.features,output);
    std::vector<double> durations;
    durations.reserve(iterations);
    for (std::size_t i=0;i<iterations;++i) {
      const auto before=std::chrono::steady_clock::now();
      pool.run(fixture.depth,fixture.features,output);
      durations.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count());
    }
    const auto errors=compare(output,fixture.expected);
    std::sort(durations.begin(),durations.end());
    const auto percentile=[&](double q){return durations[static_cast<std::size_t>(q*(durations.size()-1))];};
    std::cout<<"{\"fixture\":\""<<std::filesystem::path(argv[1]).filename().string()
      <<"\",\"kernel\":\""<<name<<"\",\"threads\":"<<threads
      <<",\"neon_available\":"<<(flashocc::neon_available()?"true":"false")
      <<",\"contributions\":"<<fixture.rd.size()<<",\"iterations\":"<<iterations
      <<",\"setup_ms\":"<<setup_ms<<",\"median_ms\":"<<percentile(.5)
      <<",\"p95_ms\":"<<percentile(.95)<<",\"min_ms\":"<<durations.front()
      <<",\"max_abs_error\":"<<errors.max_abs<<",\"different_values\":"<<errors.different
      <<",\"failed_values\":"<<errors.failed<<"}\n";
    if (argc==6) {
      std::ofstream f(argv[5],std::ios::binary);
      if (!f.write(reinterpret_cast<const char*>(output.data()),static_cast<std::streamsize>(output.size()*sizeof(float))))
        throw std::runtime_error("Could not save output");
    }
    return errors.failed?1:0;
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
  }
}
