// Isolated kappa=0 sources and the complete OCCA DG RHS; no solver/MPI.
#include "hczQuad2DKernelTest.hpp"

using namespace hczKernelTest;

namespace {
occa::json sourceProperties(const std::string& root,int degree) {
  auto props=kernelProperties(root,degree,12);
  props["defines/p_NblockV"]=4;props["defines/p_NblockCub"]=4;
  props["defines/p_cubNq"]=degree+2;props["defines/p_cubNp"]=(degree+2)*(degree+2);
  props["defines/p_Nvgeo"]=6;
  props["defines/p_RXID"]=0;props["defines/p_RYID"]=1;
  props["defines/p_SXID"]=2;props["defines/p_SYID"]=3;
  props["defines/p_JWID"]=4;props["defines/p_IJWID"]=5;
  return props;
}

Vec seed(size_t size) {
  Vec out(size);
  for(size_t a=0;a<size;a++)out[a]=.01*std::cos(.13*(a+1));
  return out;
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
    std::cerr<<label<<": error="<<std::setprecision(17)<<error<<", scale="<<scale<<"\n";
    throw std::runtime_error("HCZ source/DG comparison failed");
  }
  return error;
}

void fill(Fixture& f,const hcz::Parameters& p,int pattern) {
  f.cpu.p=p;
  for(int e=0;e<elements;e++) for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++) {
    double variation=(pattern==2 || pattern==3)?std::sin(.71*(1+3*i+7*j+11*e)):0;
    State q{};q[0]=.12+.02*variation;q[6]=pattern==3?0:.6+.05*variation;
    if(pattern!=0 && pattern!=4) {
      q[7]=p.rho(q[0])*p.c*(.13+.04*variation);
      q[8]=p.rho(q[0])*p.c*(-.09+.03*variation);
    }
    if(pattern==4) {
      double y=(e/nxElements+.5+.5*f.cpu.nodes[j])*hy;
      q[6]=.6-p.rho(q[0])*p.gravity*y;
    }
    q=hcz::equilibrium(q,p);
    if(pattern==2 || pattern==3)
      for(int k: {1,2,3,4,5,9,10,11})q[k]+=.002*(k+1)*(1+.3*variation);
    for(int k=0;k<12;k++)f.state[f.cpu.id(e,j,i,k,f.n)]=q[k];
  }
  for(int e=0;e<2*elements;e++) for(int a=0;a<f.np;a++) for(int k=0;k<12;k++)
    f.q[(e*12+k)*f.np+a]=f.state[((e%elements)*f.np+a)*12+k];
}

void unchanged(const Fixture& f,const std::vector<int>& ids,const Vec& result,
               const Vec& before,bool sourceOnly) {
  for(int e=0;e<2*elements;e++) for(int k=0;k<12;k++) for(int a=0;a<f.np;a++) {
    bool selected=std::find(ids.begin(),ids.end(),e)!=ids.end();
    if(!selected || (sourceOnly && (k==0 || k==7))) {
      size_t id=(e*12+k)*f.np+a;
      if(result[id]!=before[id])throw std::runtime_error("unexpected write to a zero-source field, unselected element or halo");
    }
  }
}

void launchSource(occa::kernel& kernel,int count,occa::memory& ids,const hcz::Parameters& p,
                  occa::memory& q,occa::memory& gradients,occa::memory& rhs) {
  kernel(count,ids,p.c,p.gravity,p.rhoL,p.rhoH,p.phiL,p.phiH,q,gradients,rhs);
}

double isolated(occa::device& device,occa::kernel& kernel,Fixture& f,
                const std::vector<int>& ids,int gradientPattern,int statePattern) {
  Vec gradient(2*elements*4*f.np,0);
  for(int e=0;e<elements;e++) for(int k=0;k<4;k++) for(int a=0;a<f.np;a++) {
    bool active=gradientPattern==3 || (gradientPattern==1 && k<2) || (gradientPattern==2 && k>=2);
    if(active)gradient[(e*4+k)*f.np+a]=.031*(k+1)*std::sin(.73*(a+3*e+7*k+1));
  }
  const auto& p=f.cpu.p;
  Vec initial=seed(f.q.size()),co(initial.size(),0),result(initial.size());
  for(int e:ids) for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++) {
    int node=j*f.n+i,base=e*4*f.np+node;
    State q=f.cpu.get(f.state,e,j,i,f.n);
    State source=hcz::source(q,p,gradient[base],gradient[base+f.np],
                            gradient[base+2*f.np],gradient[base+3*f.np],0,0);
    for(int k=0;k<12;k++)co[(e*12+k)*f.np+node]=source[k];
    // At rest with zero gradients, only vertical momentum is forced.
    if(statePattern==0 && gradientPattern==0) {
      for(int k=0;k<12;k++) {
        double analytic=k==8?-p.c*p.rho(q[0])*p.gravity:0;
        if(std::abs(source[k]-analytic)>1e-14)throw std::runtime_error("gravity normalization reference check failed");
      }
    }
  }
  auto o_ids=device.malloc<int>(ids.size(),ids.data());
  auto o_q=device.malloc<double>(f.q.size(),f.q.data());
  auto o_gradient=device.malloc<double>(gradient.size(),gradient.data());
  auto o_rhs=device.malloc<double>(initial.size(),initial.data());
  double error=0;
  for(int repeat=1;repeat<=2;repeat++) {
    launchSource(kernel,int(ids.size()),o_ids,p,o_q,o_gradient,o_rhs);
    o_rhs.copyTo(result.data());
    Vec expected=initial;
    for(size_t a=0;a<expected.size();a++)expected[a]+=repeat*co[a];
    error=std::max(error,check(result,expected,"isolated additive source"));
    unchanged(f,ids,result,initial,true);
    if(gradientPattern==0 && p.gravity==0 && result!=initial)
      throw std::runtime_error("zero source changed the RHS");
  }
  Vec afterQ(f.q.size()),afterGradient(gradient.size());
  o_q.copyTo(afterQ.data());o_gradient.copyTo(afterGradient.data());
  if(afterQ!=f.q || afterGradient!=gradient)throw std::runtime_error("source changed an input buffer");
  return error;
}

struct Kernels {occa::kernel source,thermo,gradVolume,gradSurface,volume,surface,relaxation;};
struct Errors {double preparedSource=0,full=0;};
Errors assembled(occa::device& device,Kernels& kernels,Fixture& f,const Vec& qw,
                 const std::vector<int>& ids,int pattern) {
  const auto& p=f.cpu.p;const int nc=f.cpu.nc,ncp=nc*nc;
  Vec vgeo(elements*6*f.np,0),cubvgeo(elements*6*ncp,0),interp(nc*f.n),project(nc*f.n);
  for(int a=0;a<nc;a++) for(int i=0;i<f.n;i++) {
    interp[a+i*nc]=f.cpu.I[a*f.n+i];project[i+a*f.n]=f.cpu.I[a*f.n+i];
  }
  for(int e=0;e<elements;e++) {
    double J=hx*hy*.25;
    for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++) {
      int base=e*6*f.np+j*f.n+i;
      vgeo[base]=2/hx;vgeo[base+3*f.np]=2/hy;
      vgeo[base+4*f.np]=J*f.cpu.weights[i]*f.cpu.weights[j];
      vgeo[base+5*f.np]=1/vgeo[base+4*f.np];
    }
    for(int j=0;j<nc;j++) for(int i=0;i<nc;i++)cubvgeo[(e*6+4)*ncp+j*nc+i]=J*qw[i]*qw[j];
  }
  const std::vector<int> suppliedHalo={10,4,7,1,11,5,6,0,9,3,8,2};
  auto o_all=device.malloc<int>(suppliedHalo.size(),suppliedHalo.data());
  auto o_ids=device.malloc<int>(ids.size(),ids.data());
  auto o_q=device.malloc<double>(f.q.size(),f.q.data());
  auto o_vgeo=device.malloc<double>(vgeo.size(),vgeo.data());
  auto o_cubvgeo=device.malloc<double>(cubvgeo.size(),cubvgeo.data());
  auto o_interp=device.malloc<double>(interp.size(),interp.data());
  auto o_project=device.malloc<double>(project.size(),project.data());
  auto o_D=device.malloc<double>(f.cpu.D.size(),f.cpu.D.data());
  auto o_sgeo=device.malloc<double>(f.geometry.size(),f.geometry.data());
  auto o_mapM=device.malloc<int>(f.mapM.size(),f.mapM.data());
  auto o_mapP=device.malloc<int>(f.mapP.size(),f.mapP.data());
  auto o_bc=device.malloc<int>(f.boundaries.size(),f.boundaries.data());
  auto o_xyz=device.malloc<double>(f.coordinates.size(),f.coordinates.data());
  auto o_lift=device.malloc<double>(1);
  Vec rho=seed(2*elements*f.np),potentials=seed(2*elements*2*f.np),gradient=seed(elements*4*f.np);
  auto o_rho=device.malloc<double>(rho.size(),rho.data());
  auto o_potentials=device.malloc<double>(potentials.size(),potentials.data());
  auto o_gradient=device.malloc<double>(gradient.size(),gradient.data());
  Vec initial=seed(f.q.size()),result(initial.size());
  auto o_rhs=device.malloc<double>(initial.size(),initial.data());
  // Prepare supplied moment halos as part of the fixture, without MPI.
  kernels.thermo(int(suppliedHalo.size()),o_all,p.c,p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.eosRatio,o_q,o_rho,o_potentials);
  kernels.gradVolume(int(ids.size()),o_ids,o_vgeo,o_D,o_potentials,o_gradient);
  kernels.gradSurface(int(ids.size()),o_ids,o_sgeo,o_mapM,o_mapP,o_bc,o_potentials,o_gradient);
  o_gradient.copyTo(gradient.data());
  Vec cpuSources(f.state.size(),0);f.cpu.addSources(f.state,cpuSources,false);
  Vec sourceExpected=initial;
  for(int e:ids) for(int a=0;a<f.np;a++) for(int k=0;k<12;k++)
    sourceExpected[(e*12+k)*f.np+a]+=cpuSources[(e*f.np+a)*12+k];
  launchSource(kernels.source,int(ids.size()),o_ids,p,o_q,o_gradient,o_rhs);
  o_rhs.copyTo(result.data());
  Errors errors;errors.preparedSource=check(result,sourceExpected,"thermodynamics+gradients+source");
  unchanged(f,ids,result,initial,true);

  Vec cpu=f.cpu.dg(f.state),expected=initial;
  for(int e:ids) for(int a=0;a<f.np;a++) for(int k=0;k<12;k++)
    expected[(e*12+k)*f.np+a]=cpu[(e*f.np+a)*12+k];
  for(int repeat=0;repeat<2;repeat++) {
    kernels.volume(int(ids.size()),o_ids,o_vgeo,o_D,o_xyz,o_xyz,o_xyz,0.0,p.c,.01,o_q,o_rhs);
    kernels.surface(int(ids.size()),o_ids,o_sgeo,o_lift,o_mapM,o_mapP,o_bc,o_xyz,o_xyz,o_xyz,0.0,p.c,.01,o_q,o_rhs);
    kernels.relaxation(int(ids.size()),o_ids,o_vgeo,o_cubvgeo,o_interp,o_project,0,1/p.tauG,
                       p.c,1/p.tauPhi,p.rhoL,p.rhoH,p.phiL,p.phiH,o_q,o_rhs);
    Vec beforeSource(initial.size());o_rhs.copyTo(beforeSource.data());
    launchSource(kernels.source,int(ids.size()),o_ids,p,o_q,o_gradient,o_rhs);
    o_rhs.copyTo(result.data());
    errors.full=std::max(errors.full,check(result,expected,"complete DG RHS"));
    unchanged(f,ids,result,beforeSource,true);unchanged(f,ids,result,initial,false);
    if(pattern==4 || (pattern==0 && p.gravity==0)) {
      Vec equilibrium=initial;
      for(int e:ids) for(int k=0;k<12;k++) for(int a=0;a<f.np;a++)equilibrium[(e*12+k)*f.np+a]=0;
      check(result,equilibrium,pattern==4?"hydrostatic gravity balance":"uniform rest equilibrium");
    }
    if(ids.size()==elements) {
      double mass=0;
      for(int e=0;e<elements;e++) for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++)
        mass+=result[e*12*f.np+j*f.n+i]*hx*hy*.25*f.cpu.weights[i]*f.cpu.weights[j];
      if(std::abs(mass)>2e-11)throw std::runtime_error("complete DG RHS does not conserve index mass");
    }
  }
  Vec afterQ(f.q.size()),afterGradient(gradient.size());
  o_q.copyTo(afterQ.data());o_gradient.copyTo(afterGradient.data());
  if(afterQ!=f.q || afterGradient!=gradient)throw std::runtime_error("DG assembly changed moments or prepared gradients");
  return errors;
}
} // namespace

int main(int argc,char** argv) {
  if(argc<2 || argc>3) {std::cerr<<"Usage: hczSourceQuad2D REPO_ROOT [Serial|OpenMP|CUDA]\n";return 2;}
  const std::string root=argv[1],mode=argc==3?argv[2]:"Serial";
  try {
    occa::device device;occa::json config;config["mode"]=mode;
    if(mode=="CUDA")config["device_id"]=0;
    device.setup(config);
    const std::vector<std::vector<int>> lists={{4,1,5,0,3,2},{5,0,3},{2}};
    for(int degree: {1,2,3}) {
      Vec r,w,qr,qw;quadrature(degree,r,w,qr,qw);
      auto props=sourceProperties(root,degree);
      auto build=[&](const std::string& name) {return device.buildKernel(root+"/solvers/bns/okl/"+name+".okl",name,props);};
      Kernels kernels{build("bnsHczSourceQuad2D"),build("bnsHczThermodynamicsQuad2D"),
                      build("bnsHczGradVolumeQuad2D"),build("bnsHczGradSurfaceQuad2D"),
                      build("bnsVolumeQuad2D"),build("bnsSurfaceQuad2D"),build("bnsRelaxationQuad2D")};
      double sourceError=0;Errors largest;
      for(int parameters=0;parameters<2;parameters++) for(double gravity: {0.0,.04}) {
        hcz::Parameters p;p.c=parameters?.7:1.3;p.gravity=gravity;p.kappa=0;
        p.tauPhi=parameters?.0006:.013;p.tauG=parameters?.0011:.027;
        p.rhoL=parameters?.8:1.2;p.rhoH=parameters?2.2:3.7;
        p.phiL=.03;p.phiH=.26;p.eosB=parameters?3.2:4;p.eosRatio=parameters?2.1:3;
        for(bool walls: {false,true}) for(int pattern=0;pattern<5;pattern++) {
          if(pattern==4 && !walls)continue;
          Fixture f(degree,12,walls,r,w,qr,qw);fill(f,p,pattern);
          for(const auto& ids:lists) {
            if(!walls && pattern<4)for(int gradientPattern=0;gradientPattern<4;gradientPattern++)
              sourceError=std::max(sourceError,isolated(device,kernels.source,f,ids,gradientPattern,pattern));
            Errors e=assembled(device,kernels,f,qw,ids,pattern);
            largest.preparedSource=std::max(largest.preparedSource,e.preparedSource);
            largest.full=std::max(largest.full,e.full);
          }
        }
      }
      std::cout<<mode<<" N="<<degree<<": max isolated source error="<<std::setprecision(4)<<sourceError
               <<", prepared source="<<largest.preparedSource<<", full DG="<<largest.full
               <<"; gravity/zero force, independent gradients, zero pressure, unequal rates, walls/periodic/halo traces, hydrostatic balance, subsets, partial blocks, additive RHS, preserved fields/input and index conservation passed\n";
    }
  } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
  return 0;
}
