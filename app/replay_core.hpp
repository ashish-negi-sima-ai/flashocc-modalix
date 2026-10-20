#pragma once
#include <neat.h>
#include "bev_pool.hpp"
#include "occupancy_labels.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
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
  std::string camera_input="arm-fp32",camera_pre_target,camera_post_target,bev_pre_target,bev_post_target;
  std::string camera_model_order="BGR";
  std::string camera_mode="sequential";
  int camera_guard_width=32;
  int camera_pack_size=6;
  int pool_threads=8,argmax_threads=8,render_threads=6;
  std::array<float,3> mean{123.675f,116.28f,103.53f},stddev{58.395f,57.12f,57.375f};
  bool visualization=true;
  bool inspect=false,dump=false;
};
Options finalize_execution(Options o);
int camera_input_width(const Options& o) {return o.camera_mode=="packed"?o.camera_pack_size*704+(o.camera_pack_size-1)*o.camera_guard_width:704;}
int camera_output_width(const Options& o) {return o.camera_mode=="packed"?o.camera_pack_size*44+(o.camera_pack_size-1)*(o.camera_guard_width/16):44;}
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
                  "  [--camera-pre-target TARGET] [--camera-post-target TARGET]\n"
                  "  [--camera-mode sequential|packed] [--camera-guard-width N]\n"
                  "  [--camera-pack-size 2|3|6]\n"
                  "  [--bev-pre-target TARGET] [--bev-post-target TARGET]\n"
                  "  [--inspect] [--dump-tensors]\n";
      std::exit(0);
    }
    if(i+1==argc) throw std::invalid_argument("Missing value for "+key);
    const std::string value=argv[++i];
    if(key=="--workspace") o.workspace=value;
    else if(key=="--camera-model") o.camera=value;
    else if(key=="--camera-mode") o.camera_mode=value;
    else if(key=="--camera-guard-width") o.camera_guard_width=std::stoi(value);
    else if(key=="--camera-pack-size") o.camera_pack_size=std::stoi(value);
    else if(key=="--bev-model") o.bev=value;
    else if(key=="--input-dir") o.inputs=value;
    else if(key=="--output-dir") o.output=value;
    else if(key=="--threads") o.threads=std::stoi(value);
    else if(key=="--repeat") o.repeats=std::stoi(value);
    else if(key=="--warmup") o.warmup=std::stoi(value);
    else if(key=="--timeout-ms") o.timeout=std::stoi(value);
    else if(key=="--limit") o.limit=std::stoi(value);
    else if(key=="--cvu-target") o.cvu_target=value;
    else if(key=="--camera-pre-target") o.camera_pre_target=value;
    else if(key=="--camera-post-target") o.camera_post_target=value;
    else if(key=="--bev-pre-target") o.bev_pre_target=value;
    else if(key=="--bev-post-target") o.bev_post_target=value;
    else throw std::invalid_argument("Unknown option: "+key);
  }
  if(o.threads<1 || o.threads>256 || o.repeats<1 || o.repeats>10000 || o.warmup<0 || o.warmup>100 || o.timeout<1 || o.limit<0)
    throw std::invalid_argument("Invalid numeric option");
  if(o.cvu_target!="EV74" && o.cvu_target!="A65" && o.cvu_target!="AUTO")throw std::invalid_argument("Invalid CVU target");
  for(const auto& target:{o.camera_pre_target,o.camera_post_target,o.bev_pre_target,o.bev_post_target})
    if(!target.empty() && target!="EV74" && target!="A65" && target!="AUTO")throw std::invalid_argument("Invalid per-stage CVU target");
  o.pool_threads=o.argmax_threads=o.threads;
  o=finalize_execution(std::move(o));
  if(o.inputs.empty()) o.inputs=o.workspace/"artifacts/app-replay-inputs";
  if(o.output.empty()) o.output=o.workspace/"artifacts/app-device-results";
  for(const auto& path:{o.camera,o.bev}) if(!fs::is_regular_file(path)) throw std::runtime_error("Missing model: "+path.string());
  if(!o.inspect && !fs::is_directory(o.inputs)) throw std::runtime_error("Missing input directory: "+o.inputs.string());
  return o;
}

std::string target_or(const std::string& specific,const Options& o) {return specific.empty()?o.cvu_target:specific;}
Json execution_config(const Options& o) {
  return {{"camera_input",o.camera_input},{"camera_model_order",o.camera_model_order},{"camera_mode",o.camera_mode},
          {"camera_guard_width",o.camera_guard_width},{"camera_pack_size",o.camera_pack_size},
          {"camera_model",o.camera.string()},{"bev_model",o.bev.string()},
          {"camera_pre_target",target_or(o.camera_pre_target,o)},
          {"camera_post_target",target_or(o.camera_post_target,o)},{"bev_pre_target",target_or(o.bev_pre_target,o)},
          {"bev_post_target",target_or(o.bev_post_target,o)},{"pool_threads",o.pool_threads},
          {"argmax_threads",o.argmax_threads},{"render_threads",o.render_threads},{"visualization",o.visualization},
          {"mean",o.mean},{"stddev",o.stddev},{"timeout_ms",o.timeout}};
}
void configure_execution(Options& o,const Json& config) {
  if(!config.is_object())throw std::invalid_argument("execution must be an object");
  const std::vector<std::string> keys{"camera_input","camera_pre_target","camera_post_target","bev_pre_target","bev_post_target",
    "pool_threads","argmax_threads","render_threads","visualization","mean","stddev","timeout_ms","camera_model","bev_model","camera_model_order",
    "camera_mode","camera_guard_width","camera_pack_size"};
  for(const auto& [key,value]:config.items())if(std::find(keys.begin(),keys.end(),key)==keys.end())
    throw std::invalid_argument("Unknown execution setting: "+key);
  auto text_option=[&](const char* key,std::string& destination){if(config.contains(key))destination=config.at(key).get<std::string>();};
  text_option("camera_input",o.camera_input);text_option("camera_pre_target",o.camera_pre_target);
  text_option("camera_post_target",o.camera_post_target);text_option("bev_pre_target",o.bev_pre_target);text_option("bev_post_target",o.bev_post_target);
  text_option("camera_model_order",o.camera_model_order);
  text_option("camera_mode",o.camera_mode);
  if(o.camera_mode!="sequential" && o.camera_mode!="packed")throw std::invalid_argument("camera_mode must be sequential or packed");
  if(config.contains("camera_pack_size")) {
    if(!config.at("camera_pack_size").is_number_integer())throw std::invalid_argument("camera_pack_size must be an integer");
    o.camera_pack_size=config.at("camera_pack_size");
  }
  if(o.camera_pack_size!=2 && o.camera_pack_size!=3 && o.camera_pack_size!=6)
    throw std::invalid_argument("camera_pack_size must be 2, 3 or 6 and match the compiled model");
  if(config.contains("camera_guard_width")) {
    if(!config.at("camera_guard_width").is_number_integer())throw std::invalid_argument("camera_guard_width must be an integer");
    o.camera_guard_width=config.at("camera_guard_width");
  }
  if(o.camera_guard_width<32 || o.camera_guard_width>1024 || o.camera_guard_width%32)
    throw std::invalid_argument("camera_guard_width must be a multiple of 32 in 32..1024 and match the compiled model");
  if(o.camera_mode=="packed" && o.camera_input!="arm-fp32")
    throw std::invalid_argument("Packed mode requires camera_input=arm-fp32; normalize before inserting zero guards");
  if(o.camera_input!="arm-fp32" && o.camera_input!="ev74-bgr" && o.camera_input!="ev74-nv12")throw std::invalid_argument("Invalid camera_input");
  if(o.camera_model_order!="BGR" && o.camera_model_order!="RGB")throw std::invalid_argument("Invalid camera_model_order");
  if(o.camera_input=="ev74-nv12" && o.camera_model_order!="RGB")throw std::invalid_argument("ev74-nv12 requires the RGB-input camera model and camera_model_order=RGB");
  for(const auto& target:{o.camera_pre_target,o.camera_post_target,o.bev_pre_target,o.bev_post_target})
    if(!target.empty() && target!="AUTO" && target!="A65" && target!="EV74")throw std::invalid_argument("Invalid per-stage CVU target");
  if(o.camera_input!="arm-fp32" && target_or(o.camera_pre_target,o)=="A65")
    throw std::invalid_argument("Image preproc requires EV74; use arm-fp32 for ARM normalization");
  auto integer_option=[&](const char* key,int& destination,int maximum){
    if(config.contains(key)) {if(!config.at(key).is_number_integer())throw std::invalid_argument(std::string(key)+" must be integer");destination=config.at(key);}
    if(destination<1 || destination>maximum)throw std::invalid_argument(std::string("Invalid ")+key);
  };
  integer_option("pool_threads",o.pool_threads,256);integer_option("argmax_threads",o.argmax_threads,256);
  integer_option("render_threads",o.render_threads,6);integer_option("timeout_ms",o.timeout,120000);
  if(config.contains("visualization"))o.visualization=config.at("visualization").get<bool>();
  for(const auto* key:{"mean","stddev"})if(config.contains(key) && (!config.at(key).is_array() || config.at(key).size()!=3))
    throw std::invalid_argument(std::string(key)+" must have three channel values");
  if(config.contains("mean"))o.mean=config.at("mean").get<std::array<float,3>>();
  if(config.contains("stddev"))o.stddev=config.at("stddev").get<std::array<float,3>>();
  for(int c=0;c<3;++c)if(!std::isfinite(o.mean[c]) || !std::isfinite(o.stddev[c]) || o.stddev[c]<=0)
    throw std::invalid_argument("Invalid normalization statistics");
  if(o.camera_input!="arm-fp32")for(int c=0;c<3;++c)
    if(o.mean[c]<0 || o.mean[c]>255 || o.stddev[c]>255)throw std::invalid_argument("EV74 normalization statistics outside supported pixel range");
  if(config.contains("camera_model"))o.camera=config.at("camera_model").get<std::string>();
  if(config.contains("bev_model"))o.bev=config.at("bev_model").get<std::string>();
}
Options finalize_execution(Options o) {
  configure_execution(o,Json::object());
  if(o.camera.empty()) {
    const std::string variant=o.camera_pack_size==6?"compile-packed":"compile-packed"+std::to_string(o.camera_pack_size);
    o.camera=o.workspace/(o.camera_mode=="packed"?
      "artifacts/"+variant+"/camera/flashocc_m0_camera_mpk.tar.gz":
      "artifacts/compile-real/camera/flashocc_m0_camera_mpk.tar.gz");
  }
  if(o.bev.empty())o.bev=o.workspace/"artifacts/compile-real-mixed/bev/flashocc_m0_bev_mpk.tar.gz";
  if(o.camera_mode=="packed" && o.camera_pre_target.empty() && o.cvu_target=="AUTO")o.camera_pre_target="A65";
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
  const bool camera=name=="camera";
  if(camera && options.camera_input!="arm-fp32") {
    o.preprocess.kind=neat::InputKind::Image;
    o.preprocess.normalize.enable=neat::AutoFlag::On;
    // Image preproc normalizes uint8 to [0,1] before mean/std. Public
    // application settings stay in raw pixel units, matching the ARM path.
    for(int c=0;c<3;++c) {int source=options.camera_model_order=="RGB"?2-c:c;
      o.preprocess.normalize.mean[c]=options.mean[source]/255.f;o.preprocess.normalize.stddev[c]=options.stddev[source]/255.f;}
    o.preprocess.normalize.has_explicit_stats=true;
    o.preprocess.color_convert.enable=options.camera_input=="ev74-nv12"?neat::AutoFlag::On:neat::AutoFlag::Off;
    o.preprocess.color_convert.input_format=options.camera_input=="ev74-nv12"?neat::PreprocessColorFormat::NV12:neat::PreprocessColorFormat::RGB;
    o.preprocess.color_convert.output_format=neat::PreprocessColorFormat::RGB;
  }
  o.name_suffix=name;
  o.async_queue_depth=1;
  o.processcvu.pre_run_target=target_or(camera?options.camera_pre_target:options.bev_pre_target,options);
  o.processcvu.post_run_target=target_or(camera?options.camera_post_target:options.bev_post_target,options);
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
void pack_camera_values(std::span<const float> images,std::span<float> packed,int guard_width,int cameras=6) {
  if(cameras<1 || cameras>6)throw std::invalid_argument("Invalid camera pack size");
  const std::size_t width=cameras*704+(cameras-1)*guard_width;
  if(guard_width<32 || guard_width%32 || images.size()!=static_cast<std::size_t>(cameras)*256*704*3 || packed.size()!=256*width*3)
    throw std::invalid_argument("Invalid packed camera input buffers");
  std::fill(packed.begin(),packed.end(),0.f);
  for(std::size_t view=0;view<static_cast<std::size_t>(cameras);++view)for(std::size_t row=0;row<256;++row)
    std::copy_n(images.data()+(view*256+row)*704*3,704*3,
                packed.data()+(row*width+view*(704+guard_width))*3);
}
void unpack_camera_values(std::span<const float> packed,std::span<float> images,std::size_t count,int guard_width,int cameras=6) {
  if(cameras<1 || cameras>6)throw std::invalid_argument("Invalid camera pack size");
  const std::size_t width=cameras*44+(cameras-1)*(guard_width/16);
  if(guard_width<32 || guard_width%32 || packed.size()!=16*width*count || images.size()!=cameras*pixels*count)
    throw std::invalid_argument("Invalid packed camera output buffers");
  for(std::size_t view=0;view<static_cast<std::size_t>(cameras);++view)for(std::size_t row=0;row<16;++row)
    std::copy_n(packed.data()+(row*width+view*(44+guard_width/16))*count,44*count,
                images.data()+(view*pixels+row*44)*count);
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
  std::vector<float> packed_images;
  bool built=false;
  explicit Pipeline(Options o):options(finalize_execution(std::move(o))),
                               camera(options.camera.string(),model_options(256,camera_input_width(options),3,"camera",options)),
                               bev(options.bev.string(),model_options(200,200,64,"bev",options)),selector(options.argmax_threads) {
    if(camera.compiled_batch_size()!=1)throw std::invalid_argument("Application camera modes require a batch-one archive");
    const auto output_specs=camera.output_specs();bool have_depth=false,have_features=false;
    for(const auto& spec:output_specs) {
      auto shape=spec.shape;if(shape.size()==4 && shape.front()==1)shape.erase(shape.begin());
      have_depth|=shape==std::vector<std::int64_t>{16,camera_output_width(options),44};
      have_features|=shape==std::vector<std::int64_t>{16,camera_output_width(options),64};
    }
    if(output_specs.size()!=2 || !have_depth || !have_features)
      throw std::invalid_argument("Camera archive outputs do not match camera_mode, camera_pack_size and camera_guard_width: "+specs(output_specs).dump());
    if(options.camera_mode=="packed")packed_images.resize(256*camera_input_width(options)*3);
  }
  Json contracts() {return Json{{"camera",{{"inputs",specs(camera.input_specs())},{"outputs",specs(camera.output_specs())}}},
                               {"bev",{{"inputs",specs(bev.input_specs())},{"outputs",specs(bev.output_specs())}}}};}
  Json run(const Frame& frame,const std::string& token,bool write,int repeat,
           std::vector<std::uint8_t>* result_labels=nullptr,const neat::TensorList* camera_images=nullptr) {
    if((options.camera_mode=="packed" && camera_images) ||
       (camera_images?camera_images->size()!=6:frame.images.size()!=6*256*704*3) ||
       (!frame.reference.empty() && frame.reference.size()!=voxels) ||
       (!frame.sdk_reference.empty() && frame.sdk_reference.size()!=voxels))
      throw std::runtime_error("Invalid pipeline frame sizes");
    const auto total=Clock::now();Json metrics;
    auto begin=Clock::now();
    auto mapped=flashocc::depth_ranks_nhwc(frame.rd,depth_bins,pixels);
    flashocc::PoolPlan plan(mapped,frame.rf,frame.rb,depth.size(),6*pixels,cells,channels);
    if(pooler)pooler->update_plan(std::move(plan));else pooler=std::make_unique<flashocc::Pooler>(std::move(plan),options.pool_threads);
    metrics["geometry_plan_ms"]=elapsed(begin);
    begin=Clock::now();neat::TensorList inputs;
    constexpr std::size_t image_values=256*704*3;
    metrics["camera_pack_ms"]=0.;
    if(options.camera_mode=="packed") {
      double pack_ms=0;
      for(std::size_t first=0;first<6;first+=options.camera_pack_size) {
        const auto pack_start=Clock::now();
        pack_camera_values(std::span<const float>(frame.images).subspan(first*image_values,options.camera_pack_size*image_values),
                           packed_images,options.camera_guard_width,options.camera_pack_size);
        pack_ms+=elapsed(pack_start);
        inputs.push_back(neat::Tensor::from_vector(packed_images,{256,camera_input_width(options),3},neat::TensorMemory::EV74));
      }
      metrics["camera_pack_ms"]=pack_ms;
    } else if(camera_images)inputs=*camera_images;
    else for(std::size_t i=0;i<6;++i) {
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
    begin=Clock::now();double route_ms=0,unpack_ms=0;
    const int output_width=camera_output_width(options);
    for(std::size_t i=0;i<inputs.size();++i) {
      const auto call_start=Clock::now();
      auto outputs=camera_runner.runner.run(neat::TensorList{inputs[i]},options.timeout);
      route_ms+=elapsed(call_start);const auto unpack_start=Clock::now();
      if(outputs.size()!=2)throw std::runtime_error("Camera must return depth and features; got "+std::to_string(outputs.size()));
      // Select by shape rather than assuming physical output ordering.
      bool have_depth=false,have_features=false;
      for(const auto& t:outputs) {
        auto shape=t.shape;if(shape.size()==4 && shape.front()==1)shape.erase(shape.begin());
        auto copy=[&](std::vector<float>& destination,std::size_t count) {
          if(options.camera_mode=="sequential") {
            copy_output(t,std::span<float>(destination).subspan(i*pixels*count,pixels*count),{16,44,static_cast<std::int64_t>(count)});
          } else {
            if(t.dtype!=neat::TensorDType::Float32 || !t.is_contiguous())throw std::runtime_error("Invalid packed output dtype/layout");
            const auto mapped=t.view_read();const auto size=16*output_width*count;
            if(!mapped.data || mapped.size_bytes<size*sizeof(float))throw std::runtime_error("Short packed camera output");
            unpack_camera_values({reinterpret_cast<const float*>(mapped.data),size},
              std::span<float>(destination).subspan(i*options.camera_pack_size*pixels*count,options.camera_pack_size*pixels*count),
              count,options.camera_guard_width,options.camera_pack_size);
          }
        };
        if(shape==std::vector<std::int64_t>{16,output_width,44}) {if(have_depth)throw std::runtime_error("Duplicate depth output");copy(depth,depth_bins);have_depth=true;}
        else if(shape==std::vector<std::int64_t>{16,output_width,64}) {if(have_features)throw std::runtime_error("Duplicate feature output");copy(features,channels);have_features=true;}
        else throw std::runtime_error("Unexpected camera tensor: "+tensor_spec(t).dump());
      }
      if(!have_depth || !have_features)throw std::runtime_error("Missing/duplicate camera outputs");
      unpack_ms+=elapsed(unpack_start);
    }
    metrics["camera_six_views_ms"]=elapsed(begin);
    metrics["camera_calls"]=inputs.size();metrics["camera_route_ms"]=route_ms;metrics["camera_unpack_ms"]=unpack_ms;
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
    metrics["execution"]=execution_config(options);
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

