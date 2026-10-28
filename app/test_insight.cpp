#define FLASHOCC_INSIGHT_TEST
#include "insight.cpp"
#include <limits>

void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main() {
  try {
    Options validity;
    require(validity.check_finite_logits,"Logit validity check must default to enabled");
    configure_execution(validity,{{"check_finite_logits",false}});
    require(!execution_config(validity).at("check_finite_logits").get<bool>(),"Disabled validity setting lost");
    const std::array<float,4> nonfinite{0.f,std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()};
    validate_finite_logits(nonfinite,validity.check_finite_logits);
    configure_execution(validity,{{"check_finite_logits",true}});
    for(float value:nonfinite) {
      const std::array<float,1> logits{value};bool failed=false;
      try{validate_finite_logits(logits,validity.check_finite_logits);}catch(const std::runtime_error&){failed=true;}
      require(failed==!std::isfinite(value),"Enabled check failed to distinguish finite and non-finite logits");
    }
    for(const Json invalid:{Json("false"),Json(0),Json(nullptr)}) {
      bool failed=false;
      try{configure_execution(validity,{{"check_finite_logits",invalid}});}catch(const std::exception&){failed=true;}
      require(failed,"Non-boolean validity setting accepted");
    }
    require(parse_boolean("true","check_finite_logits") && !parse_boolean("false","check_finite_logits"),"Boolean CLI values lost");
    bool invalid_boolean=false;
    try{parse_boolean("0","check_finite_logits");}catch(const std::invalid_argument&){invalid_boolean=true;}
    require(invalid_boolean,"Invalid boolean CLI value accepted");
    Synchronizer sync;std::array<cv::Mat,6> views;int sequence=24;
    for(int camera=0;camera<6;++camera)for(int marker=25;marker<=29;++marker)
      sync.frames[camera][marker+(camera?0:40)]=cv::Mat(1,1,CV_8UC1,cv::Scalar(marker));
    sequence=64;
    for(int expected=25;expected<=29;++expected) {
      require(synchronized_frame(sync,sequence,40,views),"Missing frame from decoder burst");
      require(sequence==40+expected,"Skipped frame in decoder burst");
      for(auto& view:views)require(view.at<std::uint8_t>(0,0)==expected,"Mixed camera frame markers");
    }
    for(const auto& cache:sync.frames)require(cache.empty(),"Processed frames were retained");
    stop_signal=SIGINT;
    require(!synchronized_frame(sync,sequence,40,views),"Signal did not stop synchronization");
    stop_signal=0;
    cv::Mat frame(288,704,CV_8UC3,cv::Scalar(0));
    const std::array<int,5> bytes{0xA5,39,39^255,0x31,39^3^0x5A};
    for(int bit=0;bit<40;++bit)if(bytes[bit/8]&(1<<(7-bit%8)))
      frame(cv::Rect(bit*16,264,16,24)).setTo(cv::Scalar(255,255,255));
    require(decode_marker(frame,3,40)==39,"Valid marker rejected");
    cv::Mat gray;cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
    require(decode_marker(gray,3,40)==39,"Luma marker rejected");
    require(decode_marker(frame,4,40)==-1,"Wrong camera identity accepted");
    require(decode_marker(frame,3,39)==-1,"Out-of-range frame accepted");
    frame(cv::Rect(0,264,16,24)).setTo(cv::Scalar(0));
    require(decode_marker(frame,3,40)==-1,"Corrupted marker accepted");
    Options settings;configure_execution(settings,{{"camera_input","ev74-nv12"},{"camera_model_order","RGB"},{"camera_pre_target","EV74"},
      {"bev_post_target","A65"},{"pool_threads",4},{"argmax_threads",2},{"visualization",false}});
    auto camera_options=model_options(256,704,3,"camera",settings);
    auto bev_options=model_options(200,200,64,"bev",settings);
    require(camera_options.preprocess.kind==neat::InputKind::Image,"Image preprocessing not enabled");
    require(camera_options.preprocess.color_convert.output_format==neat::PreprocessColorFormat::RGB,"Wrong RGB model channel order");
    require(camera_options.processcvu.pre_run_target=="EV74" && bev_options.processcvu.post_run_target=="A65","Lost independent placement");
    bool rejected=false;try{configure_execution(settings,{{"camera_pre_target","A65"}});}catch(const std::exception&){rejected=true;}
    require(rejected,"Unsupported image preproc target accepted");
    rejected=false;try{Options invalid;configure_execution(invalid,{{"pool_threads",0}});}catch(const std::exception&){rejected=true;}
    require(rejected,"Invalid thread count accepted");
    rejected=false;try{Options invalid;configure_execution(invalid,{{"typo",true}});}catch(const std::exception&){rejected=true;}
    require(rejected,"Unknown execution setting accepted");
    rejected=false;try{Options invalid;configure_execution(invalid,{{"camera_input","ev74-nv12"}});}catch(const std::exception&){rejected=true;}
    require(rejected,"NV12 accepted without RGB camera contract");
    rejected=false;try{Options invalid;configure_execution(invalid,{{"stddev",{1.,0.,1.}}});}catch(const std::exception&){rejected=true;}
    require(rejected,"Zero normalization standard deviation accepted");
    rejected=false;try{Options invalid;configure_execution(invalid,{{"mean",{1.,2.}}});}catch(const std::exception&){rejected=true;}
    require(rejected,"Short normalization statistics accepted");
    Options legacy;configure_execution(legacy,Json::object());
    auto legacy_options=model_options(256,704,3,"camera",legacy);
    require(legacy_options.preprocess.kind==neat::InputKind::Tensor && legacy_options.preprocess.normalize.enable==neat::AutoFlag::Off,"Changed default preprocessing path");
    Options packed_settings;configure_execution(packed_settings,{{"camera_mode","packed"}});
    packed_settings=finalize_execution(packed_settings);
    require(packed_settings.camera_pre_target=="A65" && camera_input_width(packed_settings)==4384 &&
            camera_output_width(packed_settings)==274,"Packed defaults lost");
    require(packed_settings.camera.string().find("compile-packed")!=std::string::npos,"Packed mode selected sequential archive");
    Options triple;configure_execution(triple,{{"camera_mode","packed"},{"camera_pack_size",3}});
    triple=finalize_execution(triple);
    require(camera_input_width(triple)==2176 && camera_output_width(triple)==136 &&
            triple.camera.string().find("compile-packed3")!=std::string::npos,"Three-camera model contract lost");
    configure_execution(packed_settings,{{"camera_pre_target","EV74"}});
    require(finalize_execution(packed_settings).camera_pre_target=="EV74","Explicit packed target override lost");
    require(finalize_execution(legacy).camera_mode=="sequential" && camera_input_width(legacy)==704,"Changed sequential defaults");
    Options packed_ev74;configure_execution(packed_ev74,{{"camera_mode","packed"},{"camera_pack_size",3},
      {"camera_input","ev74-bgr"},{"camera_pre_target","EV74"}});
    auto packed_model_options=model_options(256,2176,3,"camera",packed_ev74);
    auto normalize_options=PackedCameraPreprocess::normalizer_options(packed_ev74);
    require(packed_model_options.preprocess.kind==neat::InputKind::Tensor &&
            normalize_options.preprocess.kind==neat::InputKind::Image &&
            normalize_options.preprocess.resize.width==704,"Packed EV74 preprocessing contract lost");
    for(const Json invalid:{Json{{"camera_mode","other"}},Json{{"camera_mode","packed"},{"camera_input","ev74-nv12"}},
                           Json{{"camera_mode","packed"},{"camera_guard_width",31}},Json{{"camera_guard_width",32.5}},
                           Json{{"camera_pack_size",4}},Json{{"camera_pack_size",3.5}}}) {
      rejected=false;try{Options o;configure_execution(o,invalid);}catch(const std::exception&){rejected=true;}
      require(rejected,"Incompatible packed configuration accepted");
    }
    for(int cameras:{2,3,6})for(int gap:{32,64}) {
      std::vector<float> images(cameras*256*704*3);
      for(std::size_t i=0;i<images.size();++i)images[i]=static_cast<float>(i+1);
      const int width=cameras*704+(cameras-1)*gap;std::vector<float> packed(256*width*3,-1.f);
      pack_camera_values(images,packed,gap,cameras);
      for(int row=0;row<256;++row)for(int view=0;view<cameras;++view) {
        const auto input=images.begin()+(view*256+row)*704*3;
        const auto output=packed.begin()+(row*width+view*(704+gap))*3;
        require(std::equal(input,input+704*3,output),"Packing changed camera pixels or order");
        if(view<cameras-1)require(std::all_of(output+704*3,output+(704+gap)*3,[](float x){return x==0.f;}),"Normalized guards are not zero");
      }
      for(std::size_t count:{44,64}) {
        const int out_width=cameras*44+(cameras-1)*(gap/16);std::vector<float> wide(16*out_width*count),unpacked(cameras*pixels*count);
        for(std::size_t i=0;i<wide.size();++i)wide[i]=static_cast<float>(i);
        unpack_camera_values(wide,unpacked,count,gap,cameras);
        for(std::size_t view=0;view<static_cast<std::size_t>(cameras);++view)for(std::size_t row=0;row<16;++row)
          for(std::size_t x=0;x<44;++x)for(std::size_t c=0;c<count;++c)
            require(unpacked[((view*16+row)*44+x)*count+c]==wide[(row*out_width+view*(44+gap/16)+x)*count+c],
                    "Unpacking changed camera, pixel or channel order");
      }
    }
    cv::Mat nv12(384,704,CV_8UC1,cv::Scalar(128));nv12.rowRange(0,256).setTo(cv::Scalar(80));
    auto tensor=camera_image_tensor(nv12,true);auto copy=tensor.copy_nv12_contiguous();
    require(tensor.width()==704 && tensor.height()==256 && tensor.is_nv12(),"NV12 image contract corrupted");
    require(copy.size()==704*384 && copy[0]==80 && copy[704*256]==128,"NV12 planes corrupted");
    const auto temporary=fs::temp_directory_path()/("flashocc-snapshot-"+std::to_string(Clock::now().time_since_epoch().count()));
    fs::create_directory(temporary);std::vector<std::uint8_t> labels(voxels,17);labels[123]=4;
    publish_volume(temporary,labels,std::string(32,'a'),1,1);
    std::ifstream first(temporary/"latest.occ",std::ios::binary);
    labels[123]=7;publish_volume(temporary,labels,std::string(32,'b'),2,2);
    auto old=read<std::uint8_t>(first,640064);
    std::ifstream second(temporary/"latest.occ",std::ios::binary);auto current=read<std::uint8_t>(second,640064);
    require(std::memcmp(old.data(),"FOCC3D01",8)==0,"Invalid snapshot magic");
    require(old[64+123]==4 && current[64+123]==7,"Atomic replacement mixed snapshots");
    require(!fs::exists(temporary/"latest.occ.tmp"),"Snapshot temporary file was retained");fs::remove_all(temporary);
    std::cout<<"Logit validity settings, non-finite rejection, burst ordering, loop generation, camera pairing, marker validation and atomic snapshot tests passed.\n";
    return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
