// Run the camera MLA fragment with independently packed inputs, bypassing CVU adapters.
#include "replay_core.hpp"

int main(int argc,char** argv) {
  std::cout.setf(std::ios::unitbuf);
  try {
    if(argc!=4)throw std::invalid_argument("check_camera_transport CAMERA_ARCHIVE PACKED_INPUT OUTPUT_DIR");
    Options options;options.camera=argv[1];options.output=argv[3];
    auto model_options_raw=model_options(256,704,3,"transport",options);
    model_options_raw.inference_terminal.mla_only=true;
    neat::Model model(options.camera.string(),model_options_raw);
    const int batch=model.compiled_batch_size();
    constexpr std::size_t frame_bytes=1082272;
    std::ifstream file(argv[2],std::ios::binary);
    auto bytes=read<std::int8_t>(file,batch*frame_bytes);
    auto tensor=neat::Tensor::from_vector(bytes,{batch,static_cast<std::int64_t>(frame_bytes)},neat::TensorMemory::EV74);
    neat::InputOptions input;
    input.payload_type=neat::PayloadType::Tensor;input.format="INT8";
    input.memory_policy=neat::InputMemoryPolicy::Ev74;
    input.caps_override="application/vnd.simaai.tensor,format=EVXX_INT8,rank=2,dim0="+std::to_string(batch)+",dim1="+std::to_string(frame_bytes);
    neat::Graph graph;
    graph.add(neat::nodes::Input(input));graph.add(model.inference());graph.add(neat::nodes::Output());
    fs::create_directories(options.output);
    const auto outputs=graph.run(neat::TensorList{tensor});
    Json report=Json::array();
    for(std::size_t i=0;i<outputs.size();++i) {
      const auto mapped=outputs[i].view_read();
      const auto name="head-"+std::to_string(i)+".bin";
      save(options.output/name,mapped.data,mapped.size_bytes);
      report.push_back({{"file",name},{"bytes",mapped.size_bytes},{"tensor",tensor_spec(outputs[i])}});
    }
    std::ofstream summary(options.output/"outputs.json");summary<<report.dump(2)<<'\n';
    std::cout<<report.dump(2)<<'\n';
  } catch(const neat::NeatError& e){std::cerr<<e.what()<<'\n'<<e.report().to_json()<<'\n';return 1;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
