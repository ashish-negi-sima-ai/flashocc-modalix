#include "occupancy_raycast.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>

int index(int x,int y,int z){return (x*200+y)*16+z;}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main() {
  try {
    std::vector<std::uint8_t> labels(200*200*16,17);
    using flashocc::first_occupied_voxel;
    require(first_occupied_voxel(labels,{0,0,2},{1,0,0})==-1,"Empty ray has a hit");
    labels[index(130,100,7)]=4;labels[index(150,100,7)]=15;
    require(first_occupied_voxel(labels,{0,0,2},{1,0,0})==index(130,100,7),"Far object occluded near object");
    require(first_occupied_voxel(labels,{39,0,2},{-1,0,0})==index(150,100,7),"Reverse ray ordering wrong");
    require(first_occupied_voxel(labels,{0,0,2},{0,1,0})==-1,"X/Y axes swapped");
    require(first_occupied_voxel(labels,{45,0,2},{1,0,0})==-1,"Outside ray entered volume");
    require(first_occupied_voxel(labels,{45,0,2},{-1,0,0})==index(150,100,7),"Box entry failed");
    labels.assign(labels.size(),17);labels[index(100,100,3)]=11;
    require(first_occupied_voxel(labels,{0,0,4},{0,0,-1})==index(100,100,3),"Ground surface or Z indexing wrong");
    labels.assign(labels.size(),17);labels[index(130,110,7)]=4;
    require(first_occupied_voxel(labels,{0,0,2},{12.2f,4.2f,.0f},0)==index(130,110,7),"Asymmetric voxel projection wrong");
    require(first_occupied_voxel(labels,{0,0,2},{0,0,0})==-1,"Degenerate ray accepted");
    std::cout<<"Raycast occlusion, axis order, ground height, box entry and empty rays passed.\n";
    return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
