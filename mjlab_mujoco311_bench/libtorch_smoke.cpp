#include <torch/script.h>
#include <iostream>
#include <iomanip>
int main(int argc,char** argv) {
 auto actor=torch::jit::load(argv[1],torch::kCPU); torch::NoGradGuard guard;
 std::cout<<std::setprecision(9);
 while (true) {
  auto x=torch::zeros({1,31});auto p=x.data_ptr<float>();
  for(int i=0;i<31;i++)if(!(std::cin>>p[i]))return 0;
  auto y=actor.forward({x}).toTensor().contiguous();auto out=y.data_ptr<float>();
  for(int i=0;i<6;i++)std::cout<<(i?" ":"")<<out[i];std::cout<<std::endl;
 }
}
