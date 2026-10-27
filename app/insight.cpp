#include "replay_core.hpp"
#include "occupancy_raycast.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <array>
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>

struct StreamFrame {int marker; cv::Mat image;};
struct Synchronizer {
  std::mutex mutex;
  std::condition_variable changed;
  std::array<std::map<int,cv::Mat>,6> frames;
  std::array<std::uint64_t,6> received{},invalid{};
  std::array<std::vector<std::uint64_t>,6> marker_received;
  std::string error;
  std::atomic<bool> stopping=false;
};

int decode_marker(const cv::Mat& image,int camera,int count) {
  if(image.cols!=704 || image.rows!=288 || (image.type()!=CV_8UC3 && image.type()!=CV_8UC1))return -1;
  std::array<int,5> bytes{};
  for(int bit=0;bit<40;++bit) {
    auto value=cv::mean(image(cv::Rect(bit*16+4,272,8,8)));
    const double luminance=image.channels()==1?value[0]:(value[0]+value[1]+value[2])/3;
    if(luminance>70 && luminance<180)return -1;
    bytes[bit/8]=(bytes[bit/8]<<1)|(luminance>128);
  }
  if(bytes[0]!=0xA5 || bytes[2]!=(bytes[1]^255) || bytes[3]!=((camera<<4)|1) ||
     bytes[4]!=(bytes[1]^camera^0x5A) || bytes[1]>=count)return -1;
  return bytes[1];
}

struct CameraReader {
  neat::Run run;
  std::thread worker;
  Synchronizer& sync;
  CameraReader(const Json& clip,int index,int count,int fps,Synchronizer& state,bool nv12=false):sync(state) {
    neat::nodes::groups::RtspDecodedInputOptions options;
    options.url=clip.at("url");options.tcp=true;options.latency_ms=150;
    options.source_fps=fps;options.h264_width=704;options.h264_height=288;
    options.h264_fps=fps;options.auto_caps_from_stream=false;
    options.num_buffers=4;options.decoder_input_buffers=4;
    options.decoder_name="flashocc_decode_"+std::to_string(index);
    options.out_format=neat::FormatTag::NV12;
    neat::Graph graph("flashocc_camera_"+std::to_string(index));
    graph.connect(neat::nodes::groups::RtspDecodedInput(options),
                  neat::nodes::Output("camera",neat::OutputOptions::EveryFrame(4)));
    neat::RunOptions runtime;runtime.queue_depth=4;
    run=graph.build(runtime);
    worker=std::thread([this,index,count,nv12] {
      try {
        int last=-1,cycle=0;auto last_sample=Clock::now();
        while(!sync.stopping) {
          auto sample=run.pull("camera",500);
          if(!sample) {
            if(elapsed(last_sample)>20000)throw std::runtime_error("Camera "+std::to_string(index)+" stalled for 20 seconds");
            continue;
          }
          last_sample=Clock::now();auto tensors=neat::tensors_from_sample(*sample,true);
          if(tensors.size()!=1 || !tensors[0].is_nv12())throw std::runtime_error("Expected decoded NV12 camera frame");
          auto bytes=tensors[0].copy_nv12_contiguous();
          if(bytes.size()!=704*288*3/2)throw std::runtime_error("Decoded camera dimensions changed");
          cv::Mat yuv(288*3/2,704,CV_8UC1,bytes.data()),image;
          int marker;
          if(nv12) {
            marker=decode_marker(yuv.rowRange(0,288),index,count);
            // Copy both planes independently: UV starts after all 288 Y rows.
            image=cv::Mat(256*3/2,704,CV_8UC1);
            yuv.rowRange(0,256).copyTo(image.rowRange(0,256));
            yuv.rowRange(288,288+128).copyTo(image.rowRange(256,384));
          } else {
            cv::Mat bgr;cv::cvtColor(yuv,bgr,cv::COLOR_YUV2BGR_NV12);
            marker=decode_marker(bgr,index,count);
            image=bgr(cv::Rect(0,0,704,256)).clone();
          }
          std::lock_guard lock(sync.mutex);
          ++sync.received[index];
          if(marker<0){++sync.invalid[index];continue;}
          ++sync.marker_received[index][marker];
          if(last>=0 && marker<last)++cycle;
          last=marker;
          auto& cache=sync.frames[index];cache[cycle*count+marker]=std::move(image);
          while(cache.size()>12)cache.erase(cache.begin());
          sync.changed.notify_one();
        }
      } catch(const std::exception& e) {
        std::lock_guard lock(sync.mutex);sync.error=e.what();sync.changed.notify_one();
      }
    });
  }
  ~CameraReader(){if(worker.joinable())worker.join();try{run.close();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';}}
};
struct Readers {
  Synchronizer& sync;
  std::vector<std::unique_ptr<CameraReader>> cameras;
  explicit Readers(Synchronizer& s):sync(s){}
  ~Readers(){sync.stopping=true;sync.changed.notify_all();cameras.clear();}
};
bool synchronized_frame(Synchronizer& sync,int& sequence,int count,std::array<cv::Mat,6>& images) {
  std::unique_lock lock(sync.mutex);const auto deadline=Clock::now()+std::chrono::seconds(20);
  while(!stop_signal && !sync.stopping) {
    if(!sync.error.empty())throw std::runtime_error(sync.error);
    // The clips repeat identical images/geometry. Readers can join different loop
    // generations; match the encoded marker within their bounded recent caches.
    // Drain matching groups in order. A decoder can release several frames at
    // once; selecting the latest group would skip valid replay positions.
    for(auto candidate=sync.frames[0].begin();candidate!=sync.frames[0].end();++candidate) {
      const int key=candidate->first;
      if(key<=sequence)continue;
      std::array<int,6> keys{};keys[0]=key;bool have=true;
      for(int i=1;i<6;++i) {
        bool found=false;
        for(auto entry=sync.frames[i].rbegin();entry!=sync.frames[i].rend();++entry)
          if(entry->first%count==key%count){keys[i]=entry->first;found=true;break;}
        have=have && found;
      }
      if(!have)continue;
      sequence=key;
      for(int i=0;i<6;++i) {
        auto& cache=sync.frames[i];images[i]=cache.at(keys[i]);
        cache.erase(cache.begin(),cache.upper_bound(keys[i]));
      }
      return true;
    }
    if(Clock::now()>deadline)throw std::runtime_error("No matching six-camera frame within 20 seconds");
    sync.changed.wait_for(lock,std::chrono::milliseconds(100));
  }
  return false;
}

constexpr int ray_width=176,ray_height=64,rays_per_camera=ray_width*ray_height;
struct Geometry {
  Frame frame;
  std::vector<std::array<float,3>> origins,directions;
};
Geometry load_geometry(const fs::path& path) {
  std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("Missing geometry: "+path.string());
  auto magic=read<char>(file,8);auto count=read<std::uint64_t>(file,1)[0];
  if(std::memcmp(magic.data(),"FOCCGEO2",8) || count>6*pixels*depth_bins ||
     fs::file_size(path)!=16+count*12+(6+6*rays_per_camera)*3*4)throw std::runtime_error("Invalid geometry file");
  Geometry g;g.frame.rd=read<std::int32_t>(file,count);g.frame.rf=read<std::int32_t>(file,count);
  g.frame.rb=read<std::int32_t>(file,count);
  static_assert(sizeof(std::array<float,3>)==12);
  g.origins=read<std::array<float,3>>(file,6);
  g.directions=read<std::array<float,3>>(file,6*rays_per_camera);
  for(const auto& values:{std::span(g.origins),std::span(g.directions)})
    for(const auto& point:values)for(float v:point)if(!std::isfinite(v))throw std::runtime_error("Non-finite camera ray");
  return g;
}
void normalize_views(const std::array<cv::Mat,6>& views,Frame& frame,const Options& options=Options{}) {
  const auto& mean=options.mean;const auto& stddev=options.stddev;
  frame.images.resize(6*704*256*3);
  for(int camera=0;camera<6;++camera) {
    if(views[camera].rows!=256 || views[camera].cols!=704 || views[camera].type()!=CV_8UC3)
      throw std::runtime_error("Invalid cropped camera frame");
    for(int y=0;y<256;++y) {
      auto* row=views[camera].ptr<cv::Vec3b>(y);
      for(int x=0;x<704;++x)for(int c=0;c<3;++c) {
        const int source=options.camera_model_order=="RGB"?2-c:c;
        frame.images[((camera*256+y)*704+x)*3+c]=(static_cast<float>(row[x][source])-mean[source])/stddev[source];
      }
    }
  }
}
neat::Tensor camera_image_tensor(const cv::Mat& image,bool nv12,bool rgb_model=false) {
  // BF16 preproc's identity path requires RGB transport. In ev74-bgr mode
  // these are deliberately raw model-order channels: no color swap is run.
  if(!nv12) {
    if(rgb_model){cv::Mat rgb;cv::cvtColor(image,rgb,cv::COLOR_BGR2RGB);return neat::Tensor::from_cv_mat(rgb,neat::ImageSpec::PixelFormat::RGB,neat::TensorMemory::EV74);}
    return neat::Tensor::from_cv_mat(image,neat::ImageSpec::PixelFormat::RGB,neat::TensorMemory::EV74);
  }
  if(image.rows!=384 || image.cols!=704 || image.type()!=CV_8UC1 || !image.isContinuous())
    throw std::invalid_argument("Invalid cropped NV12 image");
  std::vector<std::uint8_t> bytes(image.data,image.data+384*704);
  auto tensor=neat::Tensor::from_vector(bytes,{384,704},neat::TensorMemory::EV74);
  tensor.shape={256,704};tensor.strides_bytes={704,1};tensor.layout=neat::TensorLayout::HW;tensor.axis_semantics.clear();
  tensor.semantic.image=neat::ImageSpec{neat::ImageSpec::PixelFormat::NV12,{}};
  tensor.planes={{neat::PlaneRole::Y,{256,704},{704,1},0},
                 {neat::PlaneRole::UV,{128,704},{704,1},256*704}};
  return tensor;
}
const std::array<cv::Vec3b,18> colors{{{119,119,119},{0,158,255},{255,97,255},{0,255,255},
  {255,140,0},{255,255,0},{255,0,255},{0,0,255},{0,200,255},{250,150,100},
  {255,255,80},{75,50,150},{150,0,255},{75,0,175},{75,0,75},{200,200,200},{0,175,0},{245,245,245}}};
const std::array<std::string,18> names{"other","barrier","bicycle","bus","car","construction vehicle",
  "motorcycle","pedestrian","traffic cone","trailer","truck","road","other flat","sidewalk",
  "terrain","structure","vegetation","free"};
void text(cv::Mat& image,const std::string& value,int x,int y,double size=.55,cv::Scalar color={230,230,230}) {
  cv::putText(image,value,{x,y},cv::FONT_HERSHEY_SIMPLEX,size,color,1,cv::LINE_AA);
}
cv::Mat render(const std::array<cv::Mat,6>& views,const std::vector<std::uint8_t>& labels,
               const Geometry& geometry,const Json& config,int marker,double compute_ms,int render_threads=6) {
  cv::Mat canvas(864,1536,CV_8UC3,cv::Scalar(22,25,30));
  text(canvas,"FlashOCC on Modalix | "+config.at("scene").get<std::string>()+" | frame "+std::to_string(marker)+
       " | compute "+std::to_string(static_cast<int>(compute_ms))+" ms",16,25,.65);
  std::array<cv::Mat,6> annotated;
  std::array<std::thread,6> workers;
  for(int worker=0;worker<render_threads;++worker)workers[worker]=std::thread([&,worker] {
   for(int camera=worker;camera<6;camera+=render_threads) {
    cv::Mat hits(ray_height,ray_width,CV_32SC1);
    for(int row=0;row<ray_height;++row)for(int col=0;col<ray_width;++col) {
      const int ray=row*ray_width+col;
      hits.at<int>(row,col)=flashocc::first_occupied_voxel(labels,geometry.origins[camera],geometry.directions[camera*rays_per_camera+ray]);
    }
    cv::resize(views[camera],annotated[camera],{512,186});
    cv::Mat expanded;cv::resize(hits,expanded,{512,186},0,0,cv::INTER_NEAREST);
    for(int row=0;row<186;++row)for(int col=0;col<512;++col) {
      const int voxel=expanded.at<int>(row,col);if(voxel<0)continue;
      auto& pixel=annotated[camera].at<cv::Vec3b>(row,col);
      const auto color=colors[labels[voxel]];
      for(int channel=0;channel<3;++channel)pixel[channel]=cv::saturate_cast<std::uint8_t>(pixel[channel]*.65+color[channel]*.35);
    }
   }
  });
  for(auto& worker:workers)if(worker.joinable())worker.join();
  for(int camera=0;camera<6;++camera) {
    const int left=(camera%3)*512,top=40+(camera/3)*208;
    annotated[camera].copyTo(canvas(cv::Rect(left,top+20,512,186)));
    text(canvas,config.at("clips")[camera].at("camera").get<std::string>(),left+10,top+15,.48);
  }
  text(canvas,"Camera overlay: full 3D occupancy | nearest visible voxel along each calibrated ray",16,478,.55);
  for(int slice=0;slice<2;++slice) {
    const int z=slice?5:2,left=30+slice*455,top=510;
    cv::Mat grid(200,200,CV_8UC3);
    for(int x=0;x<200;++x)for(int y=0;y<200;++y)grid.at<cv::Vec3b>(199-x,199-y)=colors[labels[(x*200+y)*16+z]];
    cv::resize(grid,canvas(cv::Rect(left,top,320,320)),{320,320},0,0,cv::INTER_NEAREST);
    for(int step=0;step<=8;++step) {
      const int offset=step*40;cv::line(canvas,{left+offset,top},{left+offset,top+320},{100,100,100},1);
      cv::line(canvas,{left,top+offset},{left+320,top+offset},{100,100,100},1);
    }
    cv::drawMarker(canvas,{left+160,top+160},{0,0,0},cv::MARKER_CROSS,14,2);
    text(canvas,slice?"BEV height: 1.2 m":"BEV height: 0.0 m",left,503,.56);
    text(canvas,"80 x 80 m | up: +x | left: +y",left,850,.45);
  }
  for(int klass=0;klass<18;++klass) {
    const int x=930+(klass/9)*300,y=518+(klass%9)*30;
    cv::rectangle(canvas,{x,y-12,16,16},cv::Scalar(colors[klass]),cv::FILLED);
    text(canvas,names[klass],x+24,y,.47);
  }
  text(canvas,"Live MLA inference + ARM NEON",930,820,.52);
  text(canvas,std::to_string(config.at("fps").get<int>())+" FPS replay | predicted occupancy",930,846,.43);
  return canvas;
}
struct RunGuard {neat::Run run;~RunGuard(){try{run.close();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';}}};

void publish_volume(const fs::path& directory,std::span<const std::uint8_t> labels,
                    const std::string& token,std::uint64_t sequence,std::uint64_t marker) {
  if constexpr(std::endian::native!=std::endian::little)throw std::runtime_error("Occupancy transport requires little endian");
  if(labels.size()!=voxels || token.size()!=32)throw std::runtime_error("Invalid volume snapshot");
  const auto timestamp=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
  const std::array<std::uint64_t,3> header{sequence,marker,timestamp};
  const auto temporary=directory/"latest.occ.tmp";
  {
    std::ofstream file(temporary,std::ios::binary);
    file.write("FOCC3D01",8);file.write(reinterpret_cast<const char*>(header.data()),sizeof(header));
    file.write(token.data(),32);file.write(reinterpret_cast<const char*>(labels.data()),labels.size());
    file.close();if(!file)throw std::runtime_error("Cannot publish 3D occupancy snapshot");
  }
  fs::rename(temporary,directory/"latest.occ");
}

#ifndef FLASHOCC_INSIGHT_TEST
int main(int argc,char** argv) {
  std::cout.setf(std::ios::unitbuf);std::cerr.setf(std::ios::unitbuf);
  try {
    fs::path config_path="/workspace/GitHub/occupancy-prediction/artifacts/insight-demo/config.json";
    fs::path output="/workspace/GitHub/occupancy-prediction/artifacts/insight-device";
    int max_frames=0,volume_fps=-1;Json overrides=Json::object();
    for(int i=1;i<argc;++i) {
      std::string key=argv[i];
      if(key=="--help"){std::cout<<"flashocc_insight [--config JSON] [--output-dir DIR] [--frames N] [--volume-fps N]\n"
        "  [--execution JSON] [--camera-input arm-fp32|ev74-bgr|ev74-nv12]\n"
        "  [--camera-mode sequential|packed] [--camera-guard-width N]\n"
        "  [--camera-pack-size 2|3|6]\n"
        "  [--camera-pre-target AUTO|EV74|A65] [--camera-post-target AUTO|EV74|A65]\n"
        "  [--bev-pre-target AUTO|EV74|A65] [--bev-post-target AUTO|EV74|A65]\n"
        "  [--camera-model TAR] [--camera-model-order BGR|RGB] [--bev-model TAR] [--timeout-ms N]\n"
        "  [--pool-threads N] [--argmax-threads N] [--render-threads N] [--visualization true|false]\n"
        "  [--check-finite-logits true|false]\n";return 0;}
      if(i+1==argc)throw std::invalid_argument("Missing argument");
      std::string value=argv[++i];
      if(key=="--config")config_path=value;else if(key=="--output-dir")output=value;
      else if(key=="--frames")max_frames=std::stoi(value);else if(key=="--volume-fps")volume_fps=std::stoi(value);
      else if(key=="--execution") {Json file;std::ifstream f(value);f>>file;if(!file.is_object())throw std::invalid_argument("Execution override must be object");overrides.update(file);}
      else if(key=="--camera-input")overrides["camera_input"]=value;
      else if(key=="--camera-mode")overrides["camera_mode"]=value;
      else if(key=="--camera-guard-width")overrides["camera_guard_width"]=std::stoi(value);
      else if(key=="--camera-pack-size")overrides["camera_pack_size"]=std::stoi(value);
      else if(key=="--camera-model-order")overrides["camera_model_order"]=value;
      else if(key=="--camera-model")overrides["camera_model"]=value;
      else if(key=="--bev-model")overrides["bev_model"]=value;
      else if(key=="--timeout-ms")overrides["timeout_ms"]=std::stoi(value);
      else if(key=="--camera-pre-target")overrides["camera_pre_target"]=value;
      else if(key=="--camera-post-target")overrides["camera_post_target"]=value;
      else if(key=="--bev-pre-target")overrides["bev_pre_target"]=value;
      else if(key=="--bev-post-target")overrides["bev_post_target"]=value;
      else if(key=="--pool-threads")overrides["pool_threads"]=std::stoi(value);
      else if(key=="--argmax-threads")overrides["argmax_threads"]=std::stoi(value);
      else if(key=="--render-threads")overrides["render_threads"]=std::stoi(value);
      else if(key=="--visualization") {if(value!="true" && value!="false")throw std::invalid_argument("visualization must be true or false");overrides["visualization"]=value=="true";}
      else if(key=="--check-finite-logits")overrides["check_finite_logits"]=parse_boolean(value,"check_finite_logits");
      else throw std::invalid_argument("Unknown option: "+key);
    }
    if(max_frames<0 || volume_fps < -1 || volume_fps>30)throw std::invalid_argument("Invalid frame or volume FPS limit");
    Json config;{std::ifstream f(config_path);f>>config;}
    if(config.at("clips").size()!=6 || config.at("frames").empty() || config.at("frames").size()>256 ||
       !config.at("fps").is_number_integer() || config.at("fps").get<int>()<1 ||
       config.at("fps").get<int>()>30 || config.at("width")!=704 || config.at("height")!=288)
      throw std::runtime_error("Invalid six-camera demo configuration");
    std::signal(SIGINT,signal_handler);std::signal(SIGTERM,signal_handler);std::signal(SIGHUP,signal_handler);
    fs::create_directories(output);
    Options options;options.output=output;
    Json execution=config.value("execution",Json::object());
    if(!execution.is_object())throw std::invalid_argument("execution must be object");
    execution.update(overrides);configure_execution(options,execution);options=finalize_execution(std::move(options));
    if(volume_fps<0)volume_fps=config.value("volume_fps",2);
    if(volume_fps<0 || volume_fps>30)throw std::invalid_argument("Invalid configured volume FPS");
    const bool nv12=options.camera_input=="ev74-nv12";
    std::cout<<Json{{"execution",execution_config(options)},{"volume_fps",volume_fps}}.dump()<<'\n';
    {std::ofstream f(output/"execution.json");f<<execution_config(options).dump(2)<<'\n';}
    Pipeline pipeline(options);
    const int fps=config.at("fps");
    auto sender_options=neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(1536,864,fps);
    sender_options.host=config.at("insight_host");sender_options.channel=config.at("channel");
    sender_options.video_port_base=config.at("video_port");sender_options.encoder.bitrate_kbps=3500;
    neat::InputOptions input;input.payload_type=neat::PayloadType::Image;input.format=neat::FormatTag::BGR;
    input.width=1536;input.height=864;input.fps_n=fps;input.memory_policy=neat::InputMemoryPolicy::Ev74;
    neat::Graph video("flashocc_insight_video");video.connect(neat::nodes::Input("video",input),neat::nodes::groups::VideoSender(sender_options));
    RunGuard sender;neat::RunOptions runtime;runtime.queue_depth=2;if(options.visualization)sender.run=video.build(runtime);
    Synchronizer sync;Readers readers(sync);
    const int frame_count=config.at("frames").size();
    for(auto& counts:sync.marker_received)counts.resize(frame_count,0);
    for(int camera=0;camera<6;++camera)readers.cameras.push_back(std::make_unique<CameraReader>(config.at("clips")[camera],camera,frame_count,fps,sync,nv12));
    std::ofstream log(output/"frames.jsonl");Json records=Json::array();
    std::vector<std::uint64_t> marker_counts(frame_count,0);
    int sequence=-1,processed=0;const auto measurement_start=Clock::now();
    auto next_volume=Clock::now();
    while(!stop_signal && (max_frames==0 || processed<max_frames)) {
      std::array<cv::Mat,6> views;
      if(!synchronized_frame(sync,sequence,frame_count,views))break;
      const auto start=Clock::now();const int marker=sequence%frame_count;
      const auto& record=config.at("frames")[marker];
      auto geometry=load_geometry(record.at("geometry").get<std::string>());
      neat::TensorList image_inputs;
      if(options.camera_input=="arm-fp32")normalize_views(views,geometry.frame,options);
      else for(const auto& view:views)image_inputs.push_back(camera_image_tensor(view,nv12,options.camera_model_order=="RGB"));
      auto prepare_ms=elapsed(start);
      std::vector<std::uint8_t> labels;
      auto result=pipeline.run(geometry.frame,record.at("token"),false,sequence/frame_count,&labels,
                               image_inputs.empty()?nullptr:&image_inputs);
      auto published=Clock::now();
      if(volume_fps>0 && published>=next_volume) {
        publish_volume(output,labels,record.at("token").get<std::string>(),sequence,marker);
        const auto period=std::chrono::microseconds(1000000/volume_fps);
        do{next_volume+=period;}while(next_volume<=published);
      }
      result["volume_publish_ms"]=elapsed(published);
      auto draw=Clock::now();
      if(options.visualization) {
      if(nv12)for(auto& view:views){cv::Mat bgr;cv::cvtColor(view,bgr,cv::COLOR_YUV2BGR_NV12);view=std::move(bgr);}
      auto rendered=render(views,labels,geometry,config,marker,result.at("compute_ms"),options.render_threads);
      result["render_ms"]=elapsed(draw);result["preparation_ms"]=prepare_ms;
      auto encoded=neat::Tensor::from_cv_mat(rendered,neat::ImageSpec::PixelFormat::BGR,neat::TensorMemory::EV74);
      if(!sender.run.push(neat::TensorList{encoded}))throw std::runtime_error("Insight video push rejected");
      if(processed==0 || processed%20==0) {
        if(!cv::imwrite((output/"latest.jpg").string(),rendered))throw std::runtime_error("Cannot save preview");
        save(output/"latest.labels.u8",labels.data(),labels.size());
      }
      } else {result["render_ms"]=0.;result["preparation_ms"]=prepare_ms;}
      result["sequence"]=sequence;result["marker"]=marker;result["application_ms"]=elapsed(start);
      result["completed_ms"]=elapsed(measurement_start);result["input_fps"]=fps;
      if(processed%20==0){std::lock_guard lock(sync.mutex);result["camera_marker_counts"]=sync.marker_received;
        result["camera_received"]=sync.received;result["invalid_markers"]=sync.invalid;}
      records.push_back(result);if(records.size()>256)records.erase(records.begin());
      ++marker_counts[marker];log<<result.dump()<<'\n';log.flush();std::cout<<result.dump()<<'\n';++processed;
    }
    Json summary={{"frames",records},{"signal",static_cast<int>(stop_signal)},
                  {"config",config_path.string()},{"ground_truth_accuracy",false},
                  {"total_processed",processed},{"marker_counts",marker_counts},
                  {"retained_frame_records",256},{"execution",execution_config(options)}};
    {std::lock_guard lock(sync.mutex);summary["camera_received"]=sync.received;summary["invalid_markers"]=sync.invalid;
     summary["camera_marker_counts"]=sync.marker_received;}
    {std::ofstream file(output/"summary.json");file<<summary.dump(2)<<'\n';}
    return stop_signal?128+stop_signal:0;
  } catch(const neat::NeatError& e){std::cerr<<e.what()<<'\n'<<e.report().to_json()<<'\n';return 1;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
