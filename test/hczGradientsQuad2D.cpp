// Isolated HCZ nodal thermodynamics and central DG gradients; no solver/MPI.
#include "hczQuad2DKernelTest.hpp"

using namespace hczKernelTest;

namespace {
occa::json gradientProperties(const std::string& root,int degree) {
  auto props=kernelProperties(root,degree,12);
  props["defines/p_NblockV"]=4;
  props["defines/p_Nvgeo"]=4;
  props["defines/p_RXID"]=0;props["defines/p_RYID"]=1;
  props["defines/p_SXID"]=2;props["defines/p_SYID"]=3;
  return props;
}

double check(const Vec& actual,const Vec& expected,const char* label) {
  double error=0,scale=1;
  for(size_t a=0;a<actual.size();a++) {
    if(!std::isfinite(actual[a]) || !std::isfinite(expected[a]))
      throw std::runtime_error(std::string(label)+": nonfinite value");
    error=std::max(error,std::abs(actual[a]-expected[a]));
    scale=std::max(scale,std::abs(expected[a]));
  }
  if(error>2e-12*scale) {
    std::cerr<<label<<": error="<<std::setprecision(17)<<error<<" scale="<<scale<<"\n";
    throw std::runtime_error("HCZ thermodynamic/gradient comparison failed");
  }
  return error;
}

Vec seed(size_t size) {
  Vec out(size);
  for(size_t a=0;a<size;a++)out[a]=.01*std::cos(.13*(a+1));
  return out;
}

// Retain the seed in unselected owned elements and any output halo region.
Vec selected(const Vec& values,const Vec& initial,const std::vector<int>& ids,int fields,int np) {
  Vec out=initial;
  for(int e:ids) for(int f=0;f<fields;f++) for(int a=0;a<np;a++)
    out[(e*fields+f)*np+a]=values[(e*fields+f)*np+a];
  return out;
}

struct Thermodynamics {Vec rho,potentials,referencePotentials;double error;};
Thermodynamics thermodynamics(occa::device& device,occa::kernel& kernel,Fixture& f,
                              const hcz::Parameters& p,const std::vector<int>& ids) {
  const int total=2*elements;
  Vec rho(total*f.np),psi(total*2*f.np);
  for(int e=0;e<total;e++) for(int a=0;a<f.np;a++) {
    double phi=f.q[e*12*f.np+a];
    rho[e*f.np+a]=p.rho(phi);
    psi[e*2*f.np+a]=p.cs(phi)-p.c*p.c*phi;
    psi[(e*2+1)*f.np+a]=f.q[(e*12+6)*f.np+a]-p.c*p.c*rho[e*f.np+a];
  }
  Vec initialRho=seed(rho.size()),initialPsi=seed(psi.size());
  Vec expectedRho=selected(rho,initialRho,ids,1,f.np),expectedPsi=selected(psi,initialPsi,ids,2,f.np);
  auto o_ids=device.malloc<int>(ids.size(),ids.data());
  auto o_q=device.malloc<double>(f.q.size(),f.q.data());
  auto o_rho=device.malloc<double>(initialRho.size(),initialRho.data());
  auto o_psi=device.malloc<double>(initialPsi.size(),initialPsi.data());
  double error=0;
  for(int repeat=0;repeat<2;repeat++) {
    kernel(int(ids.size()),o_ids,p.c,p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.eosRatio,o_q,o_rho,o_psi);
    o_rho.copyTo(rho.data());o_psi.copyTo(psi.data());
    error=std::max(error,check(rho,expectedRho,"density"));
    error=std::max(error,check(psi,expectedPsi,"potentials"));
  }
  Vec after(f.q.size());o_q.copyTo(after.data());
  if(after!=f.q)throw std::runtime_error("thermodynamics changed moments");
  return {rho,psi,expectedPsi,error};
}

struct GradientReference {Vec volume,full;};
GradientReference reference(const Fixture& f,const Vec& psi) {
  Vec volume(elements*4*f.np,0),full(volume.size(),0);
  for(int field=0;field<2;field++) {
    Vec scalar(elements*f.np);
    for(int e=0;e<elements;e++) for(int a=0;a<f.np;a++)scalar[e*f.np+a]=psi[(e*2+field)*f.np+a];
    for(int axis=0;axis<2;axis++) {
      Vec gradient=f.cpu.derivative(scalar,axis);
      for(int e=0;e<elements;e++) for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++) {
        int out=(e*4+2*field+axis)*f.np+j*f.n+i;
        full[out]=gradient[e*f.np+j*f.n+i];
        for(int m=0;m<f.n;m++)volume[out]+=2/(axis?hy:hx)*f.cpu.D[(axis?j:i)*f.n+m]
          *scalar[e*f.np+(axis?m:j)*f.n+(axis?i:m)];
      }
    }
  }
  return {volume,full};
}

void copyHalo(Vec& psi,int np) {
  for(int e=0;e<elements;e++) for(int field=0;field<2;field++) for(int a=0;a<np;a++)
    psi[((e+elements)*2+field)*np+a]=psi[(e*2+field)*np+a];
}

// Globally continuous, periodic tensor polynomials. Degree-2/3 bubbles
// vanish at element endpoints, while bilinear vertex values match faces.
Vec polynomial(const Fixture& f,const Vec& r,Vec& psi) {
  Vec analytic(elements*4*f.np,0);
  for(int e=0;e<elements;e++) for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++) for(int k=0;k<2;k++) {
    int ex=e%nxElements,ey=e/nxElements;
    auto vertex=[&](int x,int y) {
      return .1*(k+1)*std::cos(.83*((x%nxElements)+3*(y%nyElements)+k));
    };
    double a=vertex(ex,ey),b=vertex(ex+1,ey),c=vertex(ex+1,ey+1),d=vertex(ex,ey+1);
    double x=r[i],y=r[j];
    double value=.25*((1-x)*(1-y)*a+(1+x)*(1-y)*b+(1+x)*(1+y)*c+(1-x)*(1+y)*d);
    double dx=.25*((1-y)*(b-a)+(1+y)*(c-d));
    double dy=.25*((1-x)*(d-a)+(1+x)*(c-b));
    if(f.n>2) {
      double ax=.017*(ex+1)*(k+1),ay=-.021*(ey+1)*(k+2);
      value+=ax*(1-x*x)*(f.n==4?x:1)+ay*(1-y*y)*(f.n==4?y:1);
      dx+=ax*(f.n==4?1-3*x*x:-2*x);
      dy+=ay*(f.n==4?1-3*y*y:-2*y);
    }
    psi[(e*2+k)*f.np+j*f.n+i]=value;
    analytic[(e*4+2*k)*f.np+j*f.n+i]=2/hx*dx;
    analytic[(e*4+2*k+1)*f.np+j*f.n+i]=2/hy*dy;
  }
  copyHalo(psi,f.np);
  return analytic;
}

struct GradientErrors {double volume=0,surface=0,full=0;};
GradientErrors gradients(occa::device& device,occa::kernel& volume,occa::kernel& surface,
                         Fixture& f,const Vec& psi,const std::vector<int>& ids,const Vec* analytic=nullptr,
                         const Vec* referencePsi=nullptr) {
  auto ref=reference(f,referencePsi?*referencePsi:psi);
  if(analytic) {
    check(ref.volume,*analytic,"analytic volume gradient");
    check(ref.full,*analytic,"analytic central gradient");
  }
  Vec vgeo(elements*4*f.np,0);
  for(int e=0;e<elements;e++) for(int a=0;a<f.np;a++) {
    vgeo[e*4*f.np+a]=2/hx;vgeo[(e*4+3)*f.np+a]=2/hy;
  }
  auto mapP=f.mapP;
  for(size_t sk=0;sk<mapP.size();sk++)if(f.boundaries[sk/f.n]==1)mapP[sk]=-1;
  // Invalid exterior wall maps deliberately verify that copied scalar
  // traces require no exterior read. Interior maps include supplied halos.
  auto o_ids=device.malloc<int>(ids.size(),ids.data());
  auto o_vgeo=device.malloc<double>(vgeo.size(),vgeo.data());
  auto o_D=device.malloc<double>(f.cpu.D.size(),f.cpu.D.data());
  auto o_sgeo=device.malloc<double>(f.geometry.size(),f.geometry.data());
  auto o_mapM=device.malloc<int>(f.mapM.size(),f.mapM.data());
  auto o_mapP=device.malloc<int>(mapP.size(),mapP.data());
  auto o_bc=device.malloc<int>(f.boundaries.size(),f.boundaries.data());
  auto o_psi=device.malloc<double>(psi.size(),psi.data());
  // Include output sentinels after the owned elements to catch halo writes.
  Vec initial=seed(2*elements*4*f.np),result(initial.size());
  auto o_grad=device.malloc<double>(initial.size(),initial.data());
  Vec expectedVolume=selected(ref.volume,initial,ids,4,f.np);
  Vec expectedFull=selected(ref.full,initial,ids,4,f.np);
  auto launchSurface=[&]() {surface(int(ids.size()),o_ids,o_sgeo,o_mapM,o_mapP,o_bc,o_psi,o_grad);};
  GradientErrors errors;
  for(int repeat=1;repeat<=2;repeat++) {
    launchSurface();o_grad.copyTo(result.data());
    Vec expected=initial;
    for(int e:ids) for(int k=0;k<4;k++) for(int a=0;a<f.np;a++) {
      int id=(e*4+k)*f.np+a;
      expected[id]+=repeat*(ref.full[id]-ref.volume[id]);
    }
    errors.surface=std::max(errors.surface,check(result,expected,"additive surface gradient"));
  }
  for(int repeat=0;repeat<2;repeat++) {
    volume(int(ids.size()),o_ids,o_vgeo,o_D,o_psi,o_grad);
    o_grad.copyTo(result.data());
    errors.volume=std::max(errors.volume,check(result,expectedVolume,"volume gradient overwrite"));
    launchSurface();o_grad.copyTo(result.data());
    errors.full=std::max(errors.full,check(result,expectedFull,"central gradient"));
  }
  // Exact preservation of unselected elements and the output halo.
  for(int e=0;e<2*elements;e++)if(std::find(ids.begin(),ids.end(),e)==ids.end())
    for(int k=0;k<4;k++) for(int a=0;a<f.np;a++) {
      int id=(e*4+k)*f.np+a;
      if(result[id]!=initial[id])throw std::runtime_error("gradient wrote an unselected element");
    }
  Vec after(psi.size());o_psi.copyTo(after.data());
  if(after!=psi)throw std::runtime_error("gradient changed potentials");
  return errors;
}
} // namespace

int main(int argc,char** argv) {
  if(argc<2 || argc>3) {std::cerr<<"Usage: hczGradientsQuad2D REPO_ROOT [Serial|OpenMP|CUDA]\n";return 2;}
  const std::string root=argv[1],mode=argc==3?argv[2]:"Serial";
  try {
    occa::device device;occa::json config;config["mode"]=mode;
    if(mode=="CUDA")config["device_id"]=0;
    device.setup(config);
    const std::vector<std::vector<int>> lists={{4,1,5,0,3,2},{5,0,3},{2}};
    // For the end-to-end test, prepare supplied halo values too. This is
    // fixture setup, not an implementation or validation of MPI exchange.
    const std::vector<int> withHalo={10,4,7,1,11,5,6,0,9,3,8,2};
    for(int degree: {1,2,3}) {
      Vec r,w,qr,qw;quadrature(degree,r,w,qr,qw);
      auto props=gradientProperties(root,degree);
      auto thermo=device.buildKernel(root+"/solvers/bns/okl/bnsHczThermodynamicsQuad2D.okl","bnsHczThermodynamicsQuad2D",props);
      auto volume=device.buildKernel(root+"/solvers/bns/okl/bnsHczGradVolumeQuad2D.okl","bnsHczGradVolumeQuad2D",props);
      auto surface=device.buildKernel(root+"/solvers/bns/okl/bnsHczGradSurfaceQuad2D.okl","bnsHczGradSurfaceQuad2D",props);
      double thermoError=0;GradientErrors largest;
      auto record=[&](GradientErrors e) {
        largest.volume=std::max(largest.volume,e.volume);
        largest.surface=std::max(largest.surface,e.surface);
        largest.full=std::max(largest.full,e.full);
      };
      for(bool walls: {false,true}) {
        Fixture f(degree,12,walls,r,w,qr,qw);
        for(int parameters=0;parameters<2;parameters++) for(int pattern=0;pattern<3;pattern++) {
          f.fill(1);
          hcz::Parameters p;p.c=parameters?.7:1.3;
          p.rhoL=parameters?.8:1.2;p.rhoH=parameters?2.2:3.7;
          p.phiL=.03;p.phiH=.26;p.eosB=parameters?3.2:4;p.eosRatio=parameters?2.1:3;
          for(int e=0;e<2*elements;e++) for(int a=0;a<f.np;a++) {
            double variation=pattern?std::sin(.73*(a+11*(e%elements))):0;
            f.q[e*12*f.np+a]=(pattern==2?.23:.14)+.02*variation;
            f.q[(e*12+6)*f.np+a]=pattern==2?0:.6+.08*variation;
          }
          for(const auto& ids:lists)thermoError=std::max(thermoError,thermodynamics(device,thermo,f,p,ids).error);
          auto prepared=thermodynamics(device,thermo,f,p,withHalo);
          thermoError=std::max(thermoError,prepared.error);
          for(const auto& ids:lists)record(gradients(device,volume,surface,f,prepared.potentials,ids,nullptr,&prepared.referencePotentials));
        }
        for(int pattern=0;pattern<4;pattern++) {
          Vec psi(2*elements*2*f.np,0),analytic;
          for(int e=0;e<elements;e++) for(int a=0;a<f.np;a++) for(int k=0;k<2;k++)
            psi[(e*2+k)*f.np+a]=pattern==0?.17*(k+1):.13*(e+1)*(k+1);
          if(pattern==2)for(int e=0;e<elements;e++) for(int a=0;a<f.np;a++)psi[(e*2+1)*f.np+a]=0;
          copyHalo(psi,f.np);
          if(pattern==0)analytic=Vec(elements*4*f.np,0);
          if(pattern==3)analytic=polynomial(f,r,psi);
          for(const auto& ids:lists)record(gradients(device,volume,surface,f,psi,ids,analytic.empty()?nullptr:&analytic));
        }
      }
      std::cout<<mode<<" N="<<degree<<": max thermodynamic error="<<std::setprecision(4)<<thermoError
               <<", volume="<<largest.volume<<", surface="<<largest.surface<<", central="<<largest.full
               <<"; EOS/maps, zero pressure, constants, analytic polynomials, jumps, walls/periodic/halo traces, corners, subsets, partial blocks, overwrite/addition and input preservation passed\n";
    }
  } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
  return 0;
}
