#include "occupancy_raycast.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv) {
  try {
    if(argc!=4)throw std::runtime_error("Usage: raycast_volume labels.u8 geometry.geom hits.i32");
    std::vector<std::uint8_t> labels(200*200*16);
    std::ifstream input(argv[1],std::ios::binary);
    if(!input.read(reinterpret_cast<char*>(labels.data()),labels.size()) || input.peek()!=EOF)
      throw std::runtime_error("Invalid label file size");
    for(auto label:labels)if(label>17)throw std::runtime_error("Invalid occupancy class");
    std::ifstream geometry(argv[2],std::ios::binary);
    char magic[8];std::uint64_t count=0;
    if(!geometry.read(magic,8) || std::memcmp(magic,"FOCCGEO2",8) ||
       !geometry.read(reinterpret_cast<char*>(&count),8) || count>6*44*16*44)
      throw std::runtime_error("Invalid geometry header");
    geometry.seekg(static_cast<std::streamoff>(count*12),std::ios::cur);
    std::array<std::array<float,3>,6> origins;
    std::vector<std::array<float,3>> directions(6*176*64);
    if(!geometry.read(reinterpret_cast<char*>(origins.data()),sizeof(origins)) ||
       !geometry.read(reinterpret_cast<char*>(directions.data()),directions.size()*sizeof(directions[0])))
      throw std::runtime_error("Truncated geometry");
    std::vector<std::int32_t> hits(directions.size());
    for(std::size_t i=0;i<hits.size();++i)
      hits[i]=flashocc::first_occupied_voxel(labels,origins[i/(176*64)],directions[i]);
    std::ofstream output(argv[3],std::ios::binary);
    if(!output.write(reinterpret_cast<char*>(hits.data()),hits.size()*sizeof(hits[0])))
      throw std::runtime_error("Cannot write ray hits");
    return 0;
  } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
