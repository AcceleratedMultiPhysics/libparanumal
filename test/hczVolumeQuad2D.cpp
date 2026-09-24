// Isolated volume and assembled volume+surface transport; no solver or MPI.
#include "hczQuad2DKernelTest.hpp"

using namespace hczKernelTest;

namespace {
occa::json volumeProperties(const std::string& root,int degree,int fields) {
  auto props=kernelProperties(root,degree,fields);
  props["defines/p_NblockV"]=4;
  props["defines/p_Nvgeo"]=4;
  props["defines/p_RXID"]=0;
  props["defines/p_RYID"]=1;
  props["defines/p_SXID"]=2;
  props["defines/p_SYID"]=3;
  return props;
}

Vec volumeReference(const Fixture& f,const std::vector<int>& ids) {
  Vec result(f.seed.size(),0);
  // Differentiate physical fluxes as in QuadOperator::dg, without sources
  // or collision. The kernel instead differentiates q before applying H2.
  for(int e:ids) for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++) {
    State out{};
    for(int axis=0;axis<2;axis++) for(int a=0;a<f.n;a++) {
      State flux=hcz::flux(f.cpu.get(f.state,e,axis?a:j,axis?i:a,f.n),axis,soundSpeed);
      double derivative=2/(axis?hy:hx)*f.cpu.D[(axis?j:i)*f.n+a];
      for(int k=0;k<f.fields;k++)out[k]-=derivative*flux[k];
    }
    for(int k=0;k<f.fields;k++)result[(e*f.fields+k)*f.np+j*f.n+i]=out[k];
  }
  return result;
}

Vec fillPolynomial(Fixture& f,const Vec& r,const std::vector<int>& ids) {
  const int degree=f.n-1;
  Vec exact(f.seed.size(),0);
  for(int e=0;e<elements;e++) for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++) {
    State dx{},dy{};
    for(int k=0;k<12;k++) {
      const double a=.013*(k+1)*(e+1), b=-.017*(k+2), d=.011*(e+k+1);
      f.state[f.cpu.id(e,j,i,k,f.n)]=.2*(k+1)+a*std::pow(r[i],degree)
                                   +b*std::pow(r[j],degree)+d*r[i]*r[j];
      dx[k]=2/hx*(a*degree*std::pow(r[i],degree-1)+d*r[j]);
      dy[k]=2/hy*(b*degree*std::pow(r[j],degree-1)+d*r[i]);
    }
    if(std::find(ids.begin(),ids.end(),e)!=ids.end()) {
      State fx=hcz::flux(dx,0,soundSpeed), fy=hcz::flux(dy,1,soundSpeed);
      for(int k=0;k<f.fields;k++)exact[(e*f.fields+k)*f.np+j*f.n+i]=-fx[k]-fy[k];
    }
  }
  for(int e=0;e<2*elements;e++) for(int a=0;a<f.np;a++) for(int k=0;k<f.fields;k++)
    f.q[(e*f.fields+k)*f.np+a]=f.state[((e%elements)*f.np+a)*12+k];
  return exact;
}

double check(const Vec& actual,const Vec& expected,double tolerance,const char* label) {
  double error=0,scale=1;
  for(size_t a=0;a<actual.size();a++) {
    if(!std::isfinite(actual[a]))throw std::runtime_error(std::string(label)+": nonfinite result");
    error=std::max(error,std::abs(actual[a]-expected[a]));
    scale=std::max(scale,std::abs(expected[a]));
  }
  if(error>tolerance*scale) {
    std::cerr<<label<<": error="<<std::setprecision(17)<<error<<", scale="<<scale<<"\n";
    throw std::runtime_error("OCCA transport comparison failed");
  }
  return error;
}

void checkMass(const Fixture& f,const Vec& transport) {
  double integral=0;
  for(int e=0;e<elements;e++) for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++)
    integral+=transport[e*f.fields*f.np+j*f.n+i]*hx*hy*.25*f.cpu.weights[i]*f.cpu.weights[j];
  if(std::abs(integral)>2e-11)throw std::runtime_error("assembled index transport does not conserve mass");
}

struct Errors {double volume=0, assembled=0;};
Errors compareTransport(occa::device& device,occa::kernel& volume,occa::kernel& surface,
                        Fixture& f,const std::vector<int>& ids,const Vec* analytic=nullptr) {
  Vec vgeo(elements*4*f.np,0);
  for(int e=0;e<elements;e++) for(int a=0;a<f.np;a++) {
    vgeo[e*4*f.np+a]=2/hx;
    vgeo[(e*4+3)*f.np+a]=2/hy;
  }
  // No primitive conversion is needed for HCZ: the build has no body-force
  // macro, and the fixtures include an entirely zero index or pressure block.
  auto o_ids=device.malloc<int>(ids.size(),ids.data());
  auto o_vgeo=device.malloc<double>(vgeo.size(),vgeo.data());
  auto o_D=device.malloc<double>(f.cpu.D.size(),f.cpu.D.data());
  auto o_sgeo=device.malloc<double>(f.geometry.size(),f.geometry.data());
  auto o_mapM=device.malloc<int>(f.mapM.size(),f.mapM.data());
  auto o_mapP=device.malloc<int>(f.mapP.size(),f.mapP.data());
  auto o_bc=device.malloc<int>(f.boundaries.size(),f.boundaries.data());
  auto o_xyz=device.malloc<double>(f.coordinates.size(),f.coordinates.data());
  auto o_q=device.malloc<double>(f.q.size(),f.q.data());
  auto o_rhs=device.malloc<double>(f.seed.size(),f.seed.data());
  auto o_lift=device.malloc<double>(1);
  Vec expected=volumeReference(f,ids), correction=f.reference(ids);
  if(analytic)check(expected,*analytic,2e-12,"reference vs analytic polynomial");
  // Unselected elements must keep their initial RHS, including after surface.
  for(int e=0;e<elements;e++) if(std::find(ids.begin(),ids.end(),e)==ids.end())
    for(int k=0;k<f.fields;k++) for(int a=0;a<f.np;a++)expected[(e*f.fields+k)*f.np+a]=f.seed[(e*f.fields+k)*f.np+a];
  Vec assembled=expected,result(expected.size());
  for(size_t a=0;a<assembled.size();a++)assembled[a]+=correction[a];
  Errors errors;
  for(int repeat=0;repeat<2;repeat++) {
    // Starts with a nonzero seed, then with the previous assembled RHS.
    // Both calls must overwrite selected volume entries, never accumulate.
    volume(int(ids.size()),o_ids,o_vgeo,o_D,o_xyz,o_xyz,o_xyz,
           .23+repeat,soundSpeed,.01+.1*repeat,o_q,o_rhs);
    o_rhs.copyTo(result.data());
    errors.volume=std::max(errors.volume,check(result,expected,2e-12,"volume"));
    surface(int(ids.size()),o_ids,o_sgeo,o_lift,o_mapM,o_mapP,o_bc,
            o_xyz,o_xyz,o_xyz,0.0,soundSpeed,0.01,o_q,o_rhs);
    o_rhs.copyTo(result.data());
    errors.assembled=std::max(errors.assembled,check(result,assembled,f.fields==12?2e-12:3e-7,"volume+surface"));
    if(ids.size()==elements && f.fields==12)checkMass(f,result);
  }
  Vec qAfter(f.q.size());o_q.copyTo(qAfter.data());
  if(qAfter!=f.q)throw std::runtime_error("transport changed input state");
  return errors;
}
} // namespace

int main(int argc,char** argv) {
  if(argc<2 || argc>3) {std::cerr<<"Usage: hczVolumeQuad2D REPO_ROOT [Serial|OpenMP|CUDA]\n";return 2;}
  const std::string root=argv[1],mode=argc==3?argv[2]:"Serial";
  try {
    occa::device device;
    occa::json config;config["mode"]=mode;
    if(mode=="CUDA")config["device_id"]=0;
    device.setup(config);
    const std::vector<std::vector<int>> lists={{4,1,5,0,3,2},{5,0,3},{2}};
    for(int degree: {1,2,3}) {
      Vec r,w,qr,qw;quadrature(degree,r,w,qr,qw);
      auto volume=device.buildKernel(root+"/solvers/bns/okl/bnsVolumeQuad2D.okl",
                                     "bnsVolumeQuad2D",volumeProperties(root,degree,12));
      auto surface=device.buildKernel(root+"/solvers/bns/okl/bnsSurfaceQuad2D.okl",
                                      "bnsSurfaceQuad2D",kernelProperties(root,degree,12));
      Errors largest;
      for(bool walls: {false,true}) for(const auto& ids:lists) for(int pattern=0;pattern<5;pattern++) {
        Fixture f(degree,12,walls,r,w,qr,qw);f.fill(pattern);
        Vec analytic;
        if(pattern==4)analytic=fillPolynomial(f,r,ids);
        if(pattern==0) {
          analytic=Vec(f.seed.size(),0);
          check(volumeReference(f,ids),analytic,2e-12,"constant volume");
        }
        Errors error=compareTransport(device,volume,surface,f,ids,analytic.empty()?nullptr:&analytic);
        largest.volume=std::max(largest.volume,error.volume);
        largest.assembled=std::max(largest.assembled,error.assembled);
      }
      auto legacyVolume=device.buildKernel(root+"/solvers/bns/okl/bnsVolumeQuad2D.okl",
                                           "bnsVolumeQuad2D",volumeProperties(root,degree,6));
      auto legacySurface=device.buildKernel(root+"/solvers/bns/okl/bnsSurfaceQuad2D.okl",
                                            "bnsSurfaceQuad2D",kernelProperties(root,degree,6));
      for(const auto& ids:lists) {
        Fixture f(degree,6,false,r,w,qr,qw);f.fill(1);
        compareTransport(device,legacyVolume,legacySurface,f,ids);
      }
      std::cout<<mode<<" N="<<degree<<": max HCZ volume error="<<std::setprecision(4)<<largest.volume
               <<", assembled error="<<largest.assembled
               <<"; constants, analytic polynomials, independent distributions, walls/periodic/halo traces, subsets, partial blocks, overwrite, conservation and legacy BNS passed\n";
    }
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<"\n";return 1;
  }
  return 0;
}
