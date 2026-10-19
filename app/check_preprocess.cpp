// Compare camera preprocessing routes on identical decoded NV12 fixtures.
#define FLASHOCC_INSIGHT_TEST
#include "insight.cpp"

int main(int argc,char** argv) {
  std::cout.setf(std::ios::unitbuf);
  try {
    if(argc!=5)throw std::invalid_argument("check_preprocess FIXTURES_JSON EXECUTION_JSON OUTPUT_DIR REPEATS");
    Json fixtures,execution;
    {std::ifstream f(argv[1]);f>>fixtures;}{std::ifstream f(argv[2]);f>>execution;}
    Options options;options.output=argv[3];options.dump=true;
    configure_execution(options,execution);options=finalize_execution(std::move(options));fs::create_directories(options.output);
    Pipeline pipeline(options);Json results=Json::array();
    const int repeats=std::stoi(argv[4]);if(repeats<1 || repeats>1000)throw std::invalid_argument("Invalid repeats");
    for(int repeat=-1;repeat<repeats;++repeat)for(const auto& item:fixtures) {
      const auto begin=Clock::now();auto geometry=load_geometry(item.at("geometry"));
      std::array<cv::Mat,6> images;neat::TensorList tensors;
      for(int camera=0;camera<6;++camera) {
        std::ifstream f(item.at("images")[camera].get<std::string>(),std::ios::binary);
        auto bytes=read<std::uint8_t>(f,704*288*3/2);
        cv::Mat full(432,704,CV_8UC1,bytes.data());
        if(options.camera_input=="ev74-nv12") {
          images[camera]=cv::Mat(384,704,CV_8UC1);
          full.rowRange(0,256).copyTo(images[camera].rowRange(0,256));
          full.rowRange(288,416).copyTo(images[camera].rowRange(256,384));
        } else {
          cv::Mat bgr;cv::cvtColor(full,bgr,cv::COLOR_YUV2BGR_NV12);
          images[camera]=bgr(cv::Rect(0,0,704,256)).clone();
        }
      }
      if(options.camera_input=="arm-fp32")normalize_views(images,geometry.frame,options);
      else for(const auto& image:images)tensors.push_back(camera_image_tensor(image,options.camera_input=="ev74-nv12",options.camera_model_order=="RGB"));
      const double preparation_ms=elapsed(begin);
      auto metrics=pipeline.run(geometry.frame,item.at("token"),repeat==0,repeat,nullptr,tensors.empty()?nullptr:&tensors);
      metrics["preparation_ms"]=preparation_ms;metrics["check_total_ms"]=elapsed(begin);
      if(repeat>=0)results.push_back(metrics);
    }
    std::ofstream f(options.output/"summary.json");f<<Json{{"execution",execution_config(options)},{"frames",results}}.dump(2)<<'\n';
    return 0;
  } catch(const neat::NeatError& e){std::cerr<<e.what()<<'\n'<<e.report().to_json()<<'\n';return 1;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
