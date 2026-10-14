#pragma once
#include <neat.h>
#include "bev_pool.hpp"
#include "occupancy_labels.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace neat=simaai::neat;
namespace fs=std::filesystem;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
constexpr std::size_t pixels=16*44, depth_bins=44, channels=64, cells=200*200, voxels=cells*16;
volatile std::sig_atomic_t stop_signal=0;
void signal_handler(int signal) {stop_signal=signal;}
double elapsed(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}

struct Options {
  fs::path workspace="/workspace/GitHub/occupancy-prediction",camera,bev,inputs,output;
  int threads=8,repeats=1,warmup=1,timeout=20000,limit=0;
  std::string cvu_target="AUTO";
  bool inspect=false,dump=false;
};
Options parse(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    std::string key=argv[i];
    if(key=="--inspect") {o.inspect=true;continue;}
    if(key=="--dump-tensors") {o.dump=true;continue;}
    if(key=="--help") {
      std::cout<<"flashocc_replay [--workspace PATH] [--camera-model TAR] [--bev-model TAR]\n"
                  "  [--input-dir DIR] [--output-dir DIR] [--threads N] [--repeat N]\n"
                  "  [--warmup N] [--timeout-ms N] [--limit N] [--cvu-target EV74|A65|AUTO]\n"
                  "  [--inspect] [--dump-tensors]\n";
      std::exit(0);
    }
    if(i+1==argc) throw std::invalid_argument("Missing value for "+key);
    const std::string value=argv[++i];
    if(key=="--workspace") o.workspace=value;
    else if(key=="--camera-model") o.camera=value;
    else if(key=="--bev-model") o.bev=value;
    else if(key=="--input-dir") o.inputs=value;
    else if(key=="--output-dir") o.output=value;
    else if(key=="--threads") o.threads=std::stoi(value);
    else if(key=="--repeat") o.repeats=std::stoi(value);
    else if(key=="--warmup") o.warmup=std::stoi(value);
    else if(key=="--timeout-ms") o.timeout=std::stoi(value);
    else if(key=="--limit") o.limit=std::stoi(value);
    else if(key=="--cvu-target") o.cvu_target=value;
    else throw std::invalid_argument("Unknown option: "+key);
  }
  if(o.threads<1 || o.threads>256 || o.repeats<1 || o.repeats>10000 || o.warmup<0 || o.warmup>100 || o.timeout<1 || o.limit<0)
    throw std::invalid_argument("Invalid numeric option");
  if(o.cvu_target!="EV74" && o.cvu_target!="A65" && o.cvu_target!="AUTO")throw std::invalid_argument("Invalid CVU target");
  if(o.camera.empty()) o.camera=o.workspace/"artifacts/compile-real/camera/flashocc_m0_camera_mpk.tar.gz";
  if(o.bev.empty()) o.bev=o.workspace/"artifacts/compile-real-mixed/bev/flashocc_m0_bev_mpk.tar.gz";
  if(o.inputs.empty()) o.inputs=o.workspace/"artifacts/app-replay-inputs";
  if(o.output.empty()) o.output=o.workspace/"artifacts/app-device-results";
  for(const auto& path:{o.camera,o.bev}) if(!fs::is_regular_file(path)) throw std::runtime_error("Missing model: "+path.string());
  if(!o.inspect && !fs::is_directory(o.inputs)) throw std::runtime_error("Missing input directory: "+o.inputs.string());
  return o;
}

template<class T> std::vector<T> read(std::ifstream& f,std::size_t count) {
  std::vector<T> data(count);
  if(!f.read(reinterpret_cast<char*>(data.data()),static_cast<std::streamsize>(count*sizeof(T))))
    throw std::runtime_error("Truncated replay sample");
  return data;
}
struct Frame {
  std::vector<float> images;
  std::vector<std::int32_t> rd,rf,rb;
  std::vector<std::uint8_t> reference,sdk_reference;
};
Frame load_frame(const fs::path& path) {
  if constexpr(std::endian::native!=std::endian::little) throw std::runtime_error("Replay format requires little endian");
  std::ifstream f(path,std::ios::binary);if(!f) throw std::runtime_error("Cannot read sample");
  auto magic=read<char>(f,8);auto h=read<std::uint64_t>(f,6);
  if(std::memcmp(magic.data(),"FOCCAPP1",8) || h[0]!=6 || h[1]!=256 || h[2]!=704 || h[3]>6*pixels*depth_bins ||
     h[4]!=voxels || (h[5]!=0 && h[5]!=voxels)) throw std::runtime_error("Invalid M0 replay header");
  if(fs::file_size(path)!=56+6*256*704*3*4+3*h[3]*4+h[4]+h[5]) throw std::runtime_error("Replay sample length mismatch");
  Frame frame;frame.images=read<float>(f,6*256*704*3);
  if(!std::all_of(frame.images.begin(),frame.images.end(),[](float v){return std::isfinite(v);}))
    throw std::runtime_error("Non-finite input image");
  frame.rd=read<std::int32_t>(f,h[3]);frame.rf=read<std::int32_t>(f,h[3]);frame.rb=read<std::int32_t>(f,h[3]);
  frame.reference=read<std::uint8_t>(f,h[4]);frame.sdk_reference=read<std::uint8_t>(f,h[5]);
  return frame;
}

neat::Model::Options model_options(int height,int width,int depth,const std::string& name,const Options& options) {
  neat::Model::Options o;
  o.preprocess.kind=neat::InputKind::Tensor;
  o.preprocess.input_max_height=height;o.preprocess.input_max_width=width;o.preprocess.input_max_depth=depth;
  o.preprocess.normalize.enable=neat::AutoFlag::Off;
  o.preprocess.resize.enable=neat::AutoFlag::Off;
  o.preprocess.color_convert.enable=neat::AutoFlag::Off;
  o.name_suffix=name;
  o.async_queue_depth=1;
  o.processcvu.pre_run_target=options.cvu_target;
  o.processcvu.post_run_target=options.cvu_target;
  return o;
}
Json specs(const std::vector<neat::TensorSpec>& list) {
  Json j=Json::array();for(const auto& s:list) {Json types=Json::array();for(auto t:s.dtypes)types.push_back(static_cast<int>(t));j.push_back({{"shape",s.shape},{"rank",s.rank},{"dtype_codes",types}});}return j;
}
Json tensor_spec(const neat::Tensor& t) {return Json{{"shape",t.shape},{"dtype_code",static_cast<int>(t.dtype)},{"layout_code",static_cast<int>(t.layout)}};}
void copy_output(const neat::Tensor& t,std::span<float> destination,const std::vector<std::int64_t>& expected) {
  auto shape=t.shape;
  if(shape.size()==expected.size()+1 && shape.front()==1)shape.erase(shape.begin());
  if(t.dtype!=neat::TensorDType::Float32 || shape!=expected || !t.is_contiguous())
    throw std::runtime_error("Unexpected model output: "+tensor_spec(t).dump());
  auto mapped=t.view_read();
  if(!mapped.data || mapped.size_bytes<destination.size_bytes())throw std::runtime_error("Short model output buffer");
  std::memcpy(destination.data(),mapped.data,destination.size_bytes());
}
void save(const fs::path& path,const void* data,std::size_t bytes) {
  std::ofstream f(path,std::ios::binary);
  if(!f.write(static_cast<const char*>(data),static_cast<std::streamsize>(bytes)))throw std::runtime_error("Cannot write "+path.string());
}
struct RunnerGuard {
  neat::Model::Runner runner;
  ~RunnerGuard() {try{runner.close();}catch(const std::exception& e){std::cerr<<"Runner close: "<<e.what()<<'\n';}}
};

struct Pipeline {
  Options options;
  neat::Model camera,bev;
  RunnerGuard camera_runner,bev_runner;
  std::unique_ptr<flashocc::Pooler> pooler;
  flashocc::OccupancyLabels selector;
  std::vector<float> depth=std::vector<float>(6*pixels*depth_bins),features=std::vector<float>(6*pixels*channels);
  std::vector<float> pooled=std::vector<float>(cells*channels),logits=std::vector<float>(cells*16*18);
  bool built=false;
  explicit Pipeline(Options o):options(std::move(o)),camera(options.camera.string(),model_options(256,704,3,"camera",options)),
                               bev(options.bev.string(),model_options(200,200,64,"bev",options)),selector(options.threads) {}
  Json contracts() {return Json{{"camera",{{"inputs",specs(camera.input_specs())},{"outputs",specs(camera.output_specs())}}},
                               {"bev",{{"inputs",specs(bev.input_specs())},{"outputs",specs(bev.output_specs())}}}};}
  Json run(const Frame& frame,const std::string& token,bool write,int repeat,
           std::vector<std::uint8_t>* result_labels=nullptr) {
    if(frame.images.size()!=6*256*704*3 ||
       (!frame.reference.empty() && frame.reference.size()!=voxels) ||
       (!frame.sdk_reference.empty() && frame.sdk_reference.size()!=voxels))
      throw std::runtime_error("Invalid pipeline frame sizes");
    const auto total=Clock::now();Json metrics;
    auto begin=Clock::now();
    auto mapped=flashocc::depth_ranks_nhwc(frame.rd,depth_bins,pixels);
    flashocc::PoolPlan plan(mapped,frame.rf,frame.rb,depth.size(),6*pixels,cells,channels);
    if(pooler)pooler->update_plan(std::move(plan));else pooler=std::make_unique<flashocc::Pooler>(std::move(plan),options.threads);
    metrics["geometry_plan_ms"]=elapsed(begin);
    begin=Clock::now();neat::TensorList inputs;
    constexpr std::size_t image_values=256*704*3;
    for(std::size_t i=0;i<6;++i) {
      std::vector<float> values(frame.images.begin()+i*image_values,frame.images.begin()+(i+1)*image_values);
      inputs.push_back(neat::Tensor::from_vector(values,{256,704,3},neat::TensorMemory::EV74));
    }
    metrics["camera_input_ms"]=elapsed(begin);
    if(!built) {
      neat::RunOptions run_options;run_options.queue_depth=1;
      camera_runner.runner=camera.build(neat::TensorList{inputs[0]},neat::Model::RouteOptions{},run_options);
      auto seed=neat::Tensor::from_vector(pooled,{200,200,64},neat::TensorMemory::EV74);
      bev_runner.runner=bev.build(neat::TensorList{seed},neat::Model::RouteOptions{},run_options);built=true;
    }
    begin=Clock::now();
    for(std::size_t i=0;i<6;++i) {
      auto outputs=camera_runner.runner.run(neat::TensorList{inputs[i]},options.timeout);
      if(outputs.size()!=2)throw std::runtime_error("Camera must return depth and features; got "+std::to_string(outputs.size()));
      // Select by shape rather than assuming physical output ordering.
      bool have_depth=false,have_features=false;
      for(const auto& t:outputs) {
        auto shape=t.shape;if(shape.size()==4 && shape.front()==1)shape.erase(shape.begin());
        if(shape==std::vector<std::int64_t>{16,44,44}) {copy_output(t,std::span<float>(depth).subspan(i*pixels*depth_bins,pixels*depth_bins),{16,44,44});have_depth=true;}
        else if(shape==std::vector<std::int64_t>{16,44,64}) {copy_output(t,std::span<float>(features).subspan(i*pixels*channels,pixels*channels),{16,44,64});have_features=true;}
        else throw std::runtime_error("Unexpected camera tensor: "+tensor_spec(t).dump());
      }
      if(!have_depth || !have_features)throw std::runtime_error("Missing/duplicate camera outputs");
    }
    metrics["camera_six_views_ms"]=elapsed(begin);
    begin=Clock::now();pooler->run(depth,features,pooled);metrics["pooling_ms"]=elapsed(begin);
    begin=Clock::now();auto bev_input=neat::Tensor::from_vector(pooled,{200,200,64},neat::TensorMemory::EV74);
    metrics["bev_input_ms"]=elapsed(begin);
    begin=Clock::now();auto outputs=bev_runner.runner.run(neat::TensorList{bev_input},options.timeout);
    if(outputs.size()!=1)throw std::runtime_error("BEV must return one logits tensor");
    copy_output(outputs[0],logits,{200,200,288});metrics["bev_inference_ms"]=elapsed(begin);
    begin=Clock::now();std::vector<std::uint8_t> labels(voxels);
    selector.run(logits,labels);
    metrics["argmax_ms"]=elapsed(begin);metrics["compute_ms"]=elapsed(total);
    begin=Clock::now();
    if(!std::all_of(logits.begin(),logits.end(),[](float v){return std::isfinite(v);}))throw std::runtime_error("Non-finite logits");
    std::size_t equal=0,occupied=0,occupied_equal=0;
    if(!frame.reference.empty()) {
    for(std::size_t i=0;i<voxels;++i) {equal+=(labels[i]==frame.reference[i]);if(labels[i]!=17 || frame.reference[i]!=17){++occupied;occupied_equal+=(labels[i]==frame.reference[i]);}}
    metrics["fp32_label_agreement"]=static_cast<double>(equal)/voxels;
    metrics["occupied_union_agreement"]=occupied?static_cast<double>(occupied_equal)/occupied:1.;
    }
    if(!frame.sdk_reference.empty()) {
      equal=0;for(std::size_t i=0;i<voxels;++i)equal+=(labels[i]==frame.sdk_reference[i]);
      metrics["host_sdk_label_agreement"]=static_cast<double>(equal)/voxels;
    }
    metrics["token"]=token;metrics["repeat"]=repeat;metrics["threads"]=options.threads;
    metrics["cvu_target"]=options.cvu_target;
    metrics["validation_ms"]=elapsed(begin);begin=Clock::now();
    if(write) {
      save(options.output/(token+".labels.u8"),labels.data(),labels.size());
      if(options.dump) {
        save(options.output/(token+".logits.f32"),logits.data(),logits.size()*4);
        save(options.output/(token+".depth.f32"),depth.data(),depth.size()*4);
        save(options.output/(token+".features.f32"),features.data(),features.size()*4);
        save(options.output/(token+".pooled.f32"),pooled.data(),pooled.size()*4);
      }
    }
    metrics["output_io_ms"]=elapsed(begin);metrics["frame_ms"]=elapsed(total);
    if(result_labels)*result_labels=std::move(labels);
    return metrics;
  }
};

