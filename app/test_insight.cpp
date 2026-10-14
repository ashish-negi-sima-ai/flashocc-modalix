#define FLASHOCC_INSIGHT_TEST
#include "insight.cpp"

void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main() {
  try {
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
    require(decode_marker(frame,4,40)==-1,"Wrong camera identity accepted");
    require(decode_marker(frame,3,39)==-1,"Out-of-range frame accepted");
    frame(cv::Rect(0,264,16,24)).setTo(cv::Scalar(0));
    require(decode_marker(frame,3,40)==-1,"Corrupted marker accepted");
    std::cout<<"Burst ordering, loop generation, camera pairing and marker validation passed.\n";
    return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
