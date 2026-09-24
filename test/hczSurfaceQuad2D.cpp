// Isolated OCCA surface-kernel test. No bns_t, solver timestep, or MPI setup.
#include "hczQuad2DKernelTest.hpp"

using namespace hczKernelTest;

namespace {
double compare(occa::device& device,occa::kernel& kernel,Fixture& f,
               const std::vector<int>& ids,double tolerance) {
  auto o_ids=device.malloc<int>(ids.size(),ids.data());
  auto o_sgeo=device.malloc<double>(f.geometry.size(),f.geometry.data());
  auto o_mapM=device.malloc<int>(f.mapM.size(),f.mapM.data());
  auto o_mapP=device.malloc<int>(f.mapP.size(),f.mapP.data());
  auto o_bc=device.malloc<int>(f.boundaries.size(),f.boundaries.data());
  auto o_xyz=device.malloc<double>(f.coordinates.size(),f.coordinates.data());
  auto o_q=device.malloc<double>(f.q.size(),f.q.data());
  auto o_rhs=device.malloc<double>(f.seed.size(),f.seed.data());
  auto o_lift=device.malloc<double>(1);
  Vec expected=f.reference(ids), result(f.seed.size());
  double largestError=0;
  // Repeated calls check additive RHS semantics and fresh shared storage.
  for(int repeat=1;repeat<=2;repeat++) {
    kernel(int(ids.size()),o_ids,o_sgeo,o_lift,o_mapM,o_mapP,o_bc,
           o_xyz,o_xyz,o_xyz,0.0,soundSpeed,0.01,o_q,o_rhs);
    o_rhs.copyTo(result.data());
    double error=0,scale=1;
    for(size_t a=0;a<result.size();a++) {
      if(!std::isfinite(result[a]))throw std::runtime_error("nonfinite surface result");
      error=std::max(error,std::abs(result[a]-f.seed[a]-repeat*expected[a]));
      scale=std::max(scale,std::abs(repeat*expected[a]));
    }
    if(error>tolerance*scale) {
      std::cerr<<"surface error="<<std::setprecision(17)<<error<<" scale="<<scale<<" tolerance="<<tolerance<<"\n";
      throw std::runtime_error("OCCA surface differs from CPU reference");
    }
    largestError=std::max(largestError,error);
    if(f.fields==12 && ids.size()==elements) {
      Vec surface=result;
      for(size_t a=0;a<surface.size();a++)surface[a]=(surface[a]-f.seed[a])/repeat;
      f.checkTransportConservation(surface);
    }
  }
  Vec qAfter(f.q.size());o_q.copyTo(qAfter.data());
  if(qAfter!=f.q)throw std::runtime_error("surface kernel changed its input state");
  return largestError;
}
} // namespace

int main(int argc,char** argv) {
  if(argc<2 || argc>3) {std::cerr<<"Usage: hczSurfaceQuad2D REPO_ROOT [Serial|OpenMP|CUDA]\n";return 2;}
  const std::string root=argv[1],mode=argc==3?argv[2]:"Serial";
  try {
    occa::device device;
    occa::json config;config["mode"]=mode;
    if(mode=="CUDA")config["device_id"]=0;
    device.setup(config);
    std::vector<int> all={4,1,5,0,3,2},subset={5,0,3};
    for(int degree: {1,2,3}) {
      Vec r,w,qr,qw;quadrature(degree,r,w,qr,qw);
      double largestError=0;
      auto kernel=device.buildKernel(root+"/solvers/bns/okl/bnsSurfaceQuad2D.okl",
                                     "bnsSurfaceQuad2D",kernelProperties(root,degree,12));
      for(bool walls: {false,true}) {
        Fixture f(degree,12,walls,r,w,qr,qw);
        for(int pattern=0;pattern<4;pattern++) {
          f.fill(pattern);
          largestError=std::max(largestError,compare(device,kernel,f,all,2e-12));
          largestError=std::max(largestError,compare(device,kernel,f,subset,2e-12));
        }
      }
      auto legacy=device.buildKernel(root+"/solvers/bns/okl/bnsSurfaceQuad2D.okl",
                                     "bnsSurfaceQuad2D",kernelProperties(root,degree,6));
      Fixture f(degree,6,false,r,w,qr,qw);f.fill(1);
      compare(device,legacy,f,all,3e-7);
      compare(device,legacy,f,subset,3e-7);
      std::cout<<mode<<" N="<<degree<<": max HCZ error="<<std::setprecision(4)<<largestError
               <<"; walls, periodic/halo traces, corners, subsets, additive RHS, conservation and legacy BNS passed\n";
    }
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<"\n";return 1;
  }
  return 0;
}
