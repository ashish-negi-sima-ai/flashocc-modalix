// Time the public camera preprocess, raw-MLA and complete model routes separately.
#include "replay_core.hpp"

int main(int argc,char** argv) {
  std::cout.setf(std::ios::unitbuf);
  try {
    if(argc<3)throw std::invalid_argument("profile_packed_camera ARCHIVE OUTPUT_JSON [PRE_TARGET] [POST_TARGET]");
    Options options;options.camera=argv[1];options.timeout=120000;
    if(argc>3)options.camera_pre_target=argv[3];
    if(argc>4)options.camera_post_target=argv[4];
    constexpr int width=4384;
    auto frame=load_frame(options.workspace/"artifacts/app-replay-inputs/281b92269fd648d4b52d06ac06ca6d65.frame");
    std::vector<float> packed(256*width*3,0.f);
    for(int row=0;row<256;++row)for(int view=0;view<6;++view)
      std::copy_n(frame.images.data()+view*256*704*3+row*704*3,704*3,
                  packed.data()+(static_cast<std::size_t>(row)*width+view*736)*3);
    neat::TensorList inputs{neat::Tensor::from_vector(packed,{256,width,3},neat::TensorMemory::EV74)};
    Json report={{"pre_target",target_or(options.camera_pre_target,options)},
                 {"post_target",target_or(options.camera_post_target,options)}};
    for(const std::string mode:{"preprocess","preprocess_and_mla","full"}) {
      auto settings=model_options(256,width,3,"camera",options);
      settings.name_suffix="profile_"+mode;
      if(mode=="preprocess_and_mla")settings.inference_terminal.mla_only=true;
      neat::Model model(options.camera.string(),settings);
      std::vector<double> times;
      const auto measure=[&](auto& runner) {
        runner.run(inputs,options.timeout);
        for(int i=0;i<2;++i) {
          const auto begin=Clock::now();auto outputs=runner.run(inputs,options.timeout);
          if(outputs.empty())throw std::runtime_error("No profiling outputs");
          times.push_back(elapsed(begin));
        }
      };
      if(mode=="preprocess") {
        neat::GraphOptions graph_options;
        graph_options.processcvu.pre_run_target=target_or(options.camera_pre_target,options);
        graph_options.processcvu.post_run_target=target_or(options.camera_post_target,options);
        neat::Graph graph(graph_options);graph.add(neat::nodes::Input(model.input_appsrc_options(true)));
        graph.add(model.preprocess());graph.add(neat::nodes::Output());
        auto runner=graph.build(inputs);measure(runner);runner.close();
      } else {
        RunnerGuard guard;guard.runner=model.build(inputs);measure(guard.runner);
      }
      report[mode+"_ms"]=times;
      std::ofstream file(argv[2]);file<<report.dump(2)<<'\n';
      std::cout<<mode<<" "<<Json(times).dump()<<'\n';
    }
    return 0;
  } catch(const neat::NeatError& e){std::cerr<<e.what()<<'\n'<<e.report().to_json()<<'\n';return 1;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
