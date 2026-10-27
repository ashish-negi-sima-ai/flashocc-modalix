// Validate compiled camera batches on the existing six-camera holdouts.
#include "replay_core.hpp"

Json error(const std::vector<float>& actual,const fs::path& path) {
  std::ifstream f(path,std::ios::binary);auto reference=read<float>(f,actual.size());
  double squared=0,energy=0,absolute=0,maximum=0;
  for(std::size_t i=0;i<actual.size();++i) {
    if(!std::isfinite(actual[i]))throw std::runtime_error("Nonfinite camera output");
    const double delta=static_cast<double>(actual[i])-reference[i];
    squared+=delta*delta;energy+=static_cast<double>(reference[i])*reference[i];
    absolute+=std::abs(delta);maximum=std::max(maximum,std::abs(delta));
  }
  return {{"relative_rmse",std::sqrt(squared/std::max(energy,1e-30))},
          {"mae",absolute/actual.size()},{"max_absolute_error",maximum}};
}

int main(int argc,char** argv) {
  std::cout.setf(std::ios::unitbuf);
  try {
    if(argc<4)
      throw std::invalid_argument("check_camera_batch CAMERA_ARCHIVE REFERENCE_DIR OUTPUT_DIR [--packed] [--guard-width N] [--dump-boundaries] [--execution JSON]");
    bool dump_boundaries=false,packed=false;int guard_width=32;
    Options options;options.camera=argv[1];options.output=argv[3];
    options.bev=options.workspace/"artifacts/compile-real-mixed/bev/flashocc_m0_bev_mpk.tar.gz";
    for(int i=4;i<argc;++i) {
      const std::string key=argv[i];
      if(key=="--dump-boundaries")dump_boundaries=true;
      else if(key=="--packed")packed=true;
      else if(key=="--guard-width" && i+1<argc)guard_width=std::stoi(argv[++i]);
      else if(key=="--execution" && i+1<argc) {
        std::ifstream file(argv[++i]);Json config;file>>config;configure_execution(options,config);
      } else throw std::invalid_argument("Unknown or incomplete argument: "+key);
    }
    if(options.camera_input!="arm-fp32")throw std::invalid_argument("Saved replay tensors require arm-fp32 input mode");
    if(guard_width<32 || guard_width%32)throw std::invalid_argument("Guard width must be a positive multiple of 32");
    if(packed && dump_boundaries)throw std::invalid_argument("Boundary capture is only implemented for ordinary batches");
    const int input_width=packed?6*704+5*guard_width:704;
    const int output_width=packed?6*44+5*(guard_width/16):44;
    fs::create_directories(options.output);const fs::path reference_dir=argv[2];
    neat::Model camera(options.camera.string(),model_options(256,input_width,3,"camera",options));
    neat::Model bev(options.bev.string(),model_options(200,200,64,"bev",options));
    const int batch=camera.compiled_batch_size();
    if(batch<1 || batch>6 || 6%batch)throw std::runtime_error("Resolved camera batch does not divide six: "+std::to_string(batch));
    if(packed && batch!=1)throw std::runtime_error("Spatially packed camera model must use batch one");
    std::cout<<"Resolved camera batch: "<<batch<<'\n';
    std::vector<float> depth(6*pixels*depth_bins),features(6*pixels*channels),pooled(cells*channels),logits(voxels*18);
    std::vector<std::uint8_t> labels(voxels);flashocc::OccupancyLabels selector(8);
    RunnerGuard camera_runner,bev_runner;bool built=false;Json reports=Json::array();
    std::vector<fs::path> files;
    for(const auto& entry:fs::directory_iterator(options.workspace/"artifacts/app-replay-inputs"))
      if(entry.path().extension()==".frame")files.push_back(entry.path());
    std::sort(files.begin(),files.end());
    for(const auto& path:files) {
      const auto frame=load_frame(path);const auto token=path.stem().string();
      const auto prepare_begin=Clock::now();
      neat::TensorList inputs;constexpr std::size_t image_values=256*704*3;
      if(packed) {
        std::vector<float> values(static_cast<std::size_t>(256)*input_width*3,0.f);
        for(int row=0;row<256;++row)for(int view=0;view<6;++view)
          std::copy_n(frame.images.data()+view*image_values+row*704*3,704*3,
                      values.data()+(static_cast<std::size_t>(row)*input_width+view*(704+guard_width))*3);
        inputs.push_back(neat::Tensor::from_vector(values,{256,input_width,3},neat::TensorMemory::EV74));
      } else for(int first=0;first<6;first+=batch) {
        std::vector<float> values(frame.images.begin()+first*image_values,frame.images.begin()+(first+batch)*image_values);
        std::vector<std::int64_t> shape{256,704,3};if(batch>1)shape.insert(shape.begin(),batch);
        inputs.push_back(neat::Tensor::from_vector(values,shape,neat::TensorMemory::EV74));
      }
      const double input_prepare_ms=elapsed(prepare_begin);
      if(!built) {
        if(dump_boundaries) {
          Json boundary_reports=Json::array();
          for(std::size_t group=0;group<inputs.size();++group) {
            neat::Graph graph;
            graph.add(neat::nodes::Input(camera.input_appsrc_options(true)));
            graph.add(camera.preprocess());graph.add(neat::nodes::Output());
            const auto tensors=graph.run(neat::TensorList{inputs[group]});
            for(std::size_t i=0;i<tensors.size();++i) {
              // The preprocess fragment emits tiled transport. Record the
              // exposed backing mapping; public mappings may omit tile padding.
              const auto mapped=tensors[i].storage->map(neat::MapMode::Read);
              const auto name="premla-"+std::to_string(group)+"-"+std::to_string(i)+".bin";
              save(options.output/name,mapped.data,mapped.size_bytes);
              boundary_reports.push_back({{"file",name},{"tensor",tensor_spec(tensors[i])},{"bytes",mapped.size_bytes}});
            }
          }
          auto raw_options=model_options(256,704,3,"camera_raw",options);
          raw_options.inference_terminal.mla_only=true;
          neat::Model raw(options.camera.string(),raw_options);
          for(std::size_t group=0;group<inputs.size();++group) {
            const auto tensors=raw.run(neat::TensorList{inputs[group]},options.timeout);
            for(std::size_t i=0;i<tensors.size();++i) {
              const auto mapped=tensors[i].view_read();
              const auto name="rawmla-"+std::to_string(group)+"-"+std::to_string(i)+".bin";
              save(options.output/name,mapped.data,mapped.size_bytes);
              boundary_reports.push_back({{"file",name},{"tensor",tensor_spec(tensors[i])},{"bytes",mapped.size_bytes}});
            }
          }
          std::ofstream f(options.output/"boundaries.json");f<<boundary_reports.dump(2)<<'\n';
        }
        neat::RunOptions runtime;runtime.queue_depth=1;
        camera_runner.runner=camera.build(neat::TensorList{inputs[0]},neat::Model::RouteOptions{},runtime);
        auto seed=neat::Tensor::from_vector(pooled,{200,200,64},neat::TensorMemory::EV74);
        bev_runner.runner=bev.build(neat::TensorList{seed},neat::Model::RouteOptions{},runtime);
        for(int i=0;i<3;++i)camera_runner.runner.run(neat::TensorList{inputs[0]},options.timeout);
        built=true;
      }
      const auto begin=Clock::now();double route_ms=0,unpack_ms=0;
      for(std::size_t group=0;group<inputs.size();++group) {
        const auto inference_begin=Clock::now();
        const auto outputs=camera_runner.runner.run(neat::TensorList{inputs[group]},options.timeout);
        route_ms+=elapsed(inference_begin);const auto unpack_begin=Clock::now();
        if(outputs.size()!=2)throw std::runtime_error("Expected two camera outputs");
        std::vector<std::int64_t> depth_shape{16,output_width,44},feature_shape{16,output_width,64};
        if(batch>1){depth_shape.insert(depth_shape.begin(),batch);feature_shape.insert(feature_shape.begin(),batch);}
        bool have_depth=false,have_features=false;
        for(const auto& tensor:outputs) {
          auto shape=tensor.shape;if(shape.size()==4 && batch==1 && shape.front()==1)shape.erase(shape.begin());
          const auto unpack=[&](std::vector<float>& destination,std::size_t count,const auto& expected) {
            if(!packed) {copy_output(tensor,std::span<float>(destination).subspan(group*batch*pixels*count,batch*pixels*count),expected);return;}
            std::vector<float> wide(16*output_width*count);
            copy_output(tensor,wide,expected);
            for(int view=0;view<6;++view)for(int row=0;row<16;++row)
              std::copy_n(wide.data()+(static_cast<std::size_t>(row)*output_width+view*(44+guard_width/16))*count,44*count,
                          destination.data()+(static_cast<std::size_t>(view)*pixels+row*44)*count);
          };
          if(shape==depth_shape){unpack(depth,depth_bins,depth_shape);have_depth=true;}
          else if(shape==feature_shape){unpack(features,channels,feature_shape);have_features=true;}
          else throw std::runtime_error("Unexpected batch output: "+tensor_spec(tensor).dump());
        }
        if(!have_depth || !have_features)throw std::runtime_error("Missing batch camera output");
        unpack_ms+=elapsed(unpack_begin);
      }
      const double camera_ms=elapsed(begin);
      save(options.output/(token+".depth.f32"),depth.data(),depth.size()*sizeof(float));
      save(options.output/(token+".features.f32"),features.data(),features.size()*sizeof(float));
      auto rd=flashocc::depth_ranks_nhwc(frame.rd,depth_bins,pixels);
      flashocc::Pooler pooler(flashocc::PoolPlan(rd,frame.rf,frame.rb,depth.size(),6*pixels,cells,channels),8);
      pooler.run(depth,features,pooled);
      auto input=neat::Tensor::from_vector(pooled,{200,200,64},neat::TensorMemory::EV74);
      auto outputs=bev_runner.runner.run(neat::TensorList{input},options.timeout);
      if(outputs.size()!=1)throw std::runtime_error("Expected one BEV output");
      copy_output(outputs[0],logits,{200,200,288});selector.run(logits,labels);
      std::size_t equal=0,occupied=0,occupied_equal=0;
      for(std::size_t i=0;i<voxels;++i){equal+=labels[i]==frame.reference[i];if(labels[i]!=17 || frame.reference[i]!=17){++occupied;occupied_equal+=labels[i]==frame.reference[i];}}
      Json record={{"token",token},{"batch",batch},{"spatial_packed",packed},{"camera_calls",inputs.size()},
        {"six_camera_ms",camera_ms},{"input_prepare_ms",input_prepare_ms},{"camera_route_ms",route_ms},
        {"output_unpack_ms",unpack_ms},{"camera_with_input_prepare_ms",input_prepare_ms+camera_ms},
        {"depth",error(depth,reference_dir/(token+".depth.f32"))},
        {"features",error(features,reference_dir/(token+".features.f32"))},
        {"fp32_label_agreement",static_cast<double>(equal)/voxels},
        {"occupied_union_label_agreement",occupied?static_cast<double>(occupied_equal)/occupied:1.}};
      save(options.output/(token+".labels.u8"),labels.data(),labels.size());reports.push_back(record);
      {std::ofstream f(options.output/"accuracy.json");f<<Json{{"batch",batch},{"camera_model",options.camera.string()},
        {"bev_model",options.bev.string()},{"execution",execution_config(options)},
        {"reference_is_ground_truth",false},{"samples",reports}}.dump(2)<<'\n';}
      std::cout<<record.dump()<<'\n';
    }
    return 0;
  } catch(const neat::NeatError& e){std::cerr<<e.what()<<'\n'<<e.report().to_json()<<'\n';return 1;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
