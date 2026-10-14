#include "replay_core.hpp"

int main(int argc,char** argv) {
  std::cout.setf(std::ios::unitbuf);std::cerr.setf(std::ios::unitbuf);
  try {
    auto options=parse(argc,argv);
    std::signal(SIGINT,signal_handler);std::signal(SIGTERM,signal_handler);std::signal(SIGHUP,signal_handler);
    fs::create_directories(options.output);
    std::vector<fs::path> files;
    if(!options.inspect) {
      for(const auto& e:fs::directory_iterator(options.inputs))if(e.is_regular_file() && e.path().extension()==".frame")files.push_back(e.path());
      std::sort(files.begin(),files.end());if(options.limit && files.size()>static_cast<std::size_t>(options.limit))files.resize(options.limit);
      if(files.empty())throw std::runtime_error("No .frame replay inputs");
    }
    Pipeline pipeline(options);const auto contracts=pipeline.contracts();
    std::cout<<Json{{"contracts",contracts}}.dump()<<'\n';
    {std::ofstream f(options.output/"contracts.json");f<<contracts.dump(2)<<'\n';}
    if(options.inspect)return 0;
    const auto first=load_frame(files.front());
    for(int i=0;i<options.warmup && !stop_signal;++i) {pipeline.run(first,files.front().stem().string(),false,-1);std::cout<<"warmup "<<i+1<<'\n';}
    Json results=Json::array();std::ofstream log(options.output/"frames.jsonl");
    for(int repeat=0;repeat<options.repeats && !stop_signal;++repeat)for(const auto& file:files) {
      if(stop_signal)break;
      const auto start=Clock::now();const auto frame=load_frame(file);const auto io_ms=elapsed(start);
      auto result=pipeline.run(frame,file.stem().string(),true,repeat);result["input_io_ms"]=io_ms;
      results.push_back(result);log<<result.dump()<<'\n';log.flush();std::cout<<result.dump()<<'\n';
    }
    Json summary={{"device_neon",flashocc::neon_available()},{"camera_model",options.camera.string()},
                  {"bev_model",options.bev.string()},{"reference_is_ground_truth",false},
                  {"signal",static_cast<int>(stop_signal)},{"frames",results},
                  {"timing_scope","compute_ms includes plan, input copies, six camera inferences, pooling, BEV inference, output copies and argmax; excludes validation comparisons and file IO. frame_ms includes validation and output writes; input_io_ms is separate."}};
    {std::ofstream f(options.output/"summary.json");f<<summary.dump(2)<<'\n';}
    return stop_signal?128+stop_signal:0;
  } catch(const neat::NeatError& e) {
    std::cerr<<e.what()<<'\n'<<e.report().to_json()<<'\n';return 1;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
