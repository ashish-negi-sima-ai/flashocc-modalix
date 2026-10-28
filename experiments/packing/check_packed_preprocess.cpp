// Hardware check: EV74 pixel math, camera ordering and exact quantized-zero gaps.
#define FLASHOCC_INSIGHT_TEST
#include "insight.cpp"

int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::invalid_argument("check_packed_preprocess EXECUTION_JSON OUTPUT_DIR");
    Json execution;{std::ifstream file(argv[1]);file>>execution;}
    Options options;configure_execution(options,execution);options=finalize_execution(options);
    if(options.camera_mode!="packed" || options.camera_input!="ev74-bgr")
      throw std::invalid_argument("Select packed mode with EV74 BGR preprocessing");
    neat::Model model(options.camera.string(),model_options(256,camera_input_width(options),3,"camera",options));
    PackedCameraPreprocess preprocess(options,model);neat::TensorList images;
    for(int camera=0;camera<6;++camera) {
      cv::Mat image(256,704,CV_8UC3);
      for(int y=0;y<256;++y)for(int x=0;x<704;++x)for(int c=0;c<3;++c)
        image.at<cv::Vec3b>(y,x)[c]=(x+3*y+c*31+camera*17)%256;
      images.push_back(camera_image_tensor(image,false,options.camera_model_order=="RGB"));
    }
    double normalize_ms=0.,pack_ms=0.;auto inputs=preprocess.run(images,normalize_ms,pack_ms);
    const int width=camera_input_width(options),pack_size=options.camera_pack_size;
    if(inputs.size()!=static_cast<std::size_t>(6/pack_size))throw std::runtime_error("Incorrect packed call count");
    std::size_t exact=0,values=0,guards=0;int max_error=0;
    fs::create_directories(argv[2]);
    for(std::size_t group=0;group<inputs.size();++group) {
      const auto view=inputs[group].view_read();const auto* data=static_cast<const std::int8_t*>(view.data);
      if(!data || view.size_bytes<static_cast<std::size_t>(256*width*3))throw std::runtime_error("Short packed input");
      for(int row=0;row<256;++row)for(int camera=0;camera<pack_size;++camera) {
        const auto offset=(row*width+camera*(704+options.camera_guard_width))*3;
        for(int x=0;x<704;++x)for(int c=0;c<3;++c) {
          const int source=options.camera_model_order=="RGB"?2-c:c;
          const float pixel=(x+3*row+source*31+(group*pack_size+camera)*17)%256;
          const float normalized=(pixel-options.mean[source])/options.stddev[source];
          const int expected=std::clamp(static_cast<int>(std::nearbyint(normalized*preprocess.multiplier))+preprocess.zero,-128,127);
          const int error=std::abs(static_cast<int>(data[offset+x*3+c])-expected);
          max_error=std::max(max_error,error);exact+=error==0;++values;
        }
        if(camera+1<pack_size)for(int i=0;i<options.camera_guard_width*3;++i) {
          if(data[offset+704*3+i]!=preprocess.zero)throw std::runtime_error("Packed gap is not quantized zero");
          ++guards;
        }
      }
      save(fs::path(argv[2])/("input-"+std::to_string(group)+".i8"),data,256*width*3);
    }
    if(max_error>1)throw std::runtime_error("EV74 normalized pixels differ by more than one quantization level");
    Json report={{"passed",true},{"camera_images",6},{"max_quantized_pixel_error",max_error},
      {"exact_pixel_fraction",static_cast<double>(exact)/values},{"pixel_values",values},
      {"guard_values_checked",guards},{"quantized_zero",preprocess.zero},{"quantization_multiplier",preprocess.multiplier},
      {"normalize_ms",normalize_ms},{"pack_ms",pack_ms},{"execution",execution_config(options)}};
    std::ofstream file(fs::path(argv[2])/"validation.json");file<<report.dump(2)<<'\n';std::cout<<report.dump()<<std::endl;
    return 0;
  }catch(const neat::NeatError& e){std::cerr<<e.what()<<'\n'<<e.report().to_json()<<'\n';return 1;}
   catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
