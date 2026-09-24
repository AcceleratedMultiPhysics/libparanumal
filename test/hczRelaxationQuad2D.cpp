// Isolated cubature BGK and volume+surface+BGK assembly; no solver or MPI.
#include "hczQuad2DKernelTest.hpp"

using namespace hczKernelTest;

namespace {
occa::json relaxationProperties(const std::string& root,int degree,int fields) {
  auto props=kernelProperties(root,degree,fields);
  props["defines/p_NblockCub"]=4;
  props["defines/p_NblockV"]=4;
  props["defines/p_cubNq"]=degree+2;
  props["defines/p_cubNp"]=(degree+2)*(degree+2);
  props["defines/p_Nvgeo"]=6;
  props["defines/p_RXID"]=0;
  props["defines/p_RYID"]=1;
  props["defines/p_SXID"]=2;
  props["defines/p_SYID"]=3;
  props["defines/p_JWID"]=4;
  props["defines/p_IJWID"]=5;
  return props;
}

void fill(Fixture& f,const hcz::Parameters& p,int pattern) {
  f.fill(0);f.cpu.p=p;
  for(int e=0;e<elements;e++) for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++) {
    double variation=pattern>=2?std::sin(.71*(1+3*i+7*j+11*e)):0;
    State state{};
    state[0]=.12+.02*variation;
    state[6]=pattern==3?0:.6+.05*variation;
    state[7]=p.rho(state[0])*p.c*(.13+.04*variation);
    state[8]=p.rho(state[0])*p.c*(-.09+.03*variation);
    state=hcz::equilibrium(state,p);
    if(pattern==4) {
      // Change only hydrodynamic momentum after forming both equilibria:
      // collision must respond in the index distribution as well.
      state[7]+=.11;state[8]-=.07;
    } else if(pattern!=0) {
      for(int k: {1,2,3,4,5,9,10,11})state[k]+=.002*(k+1)*(1+.3*variation);
    }
    for(int k=0;k<12;k++)f.state[f.cpu.id(e,j,i,k,f.n)]=state[k];
  }
  for(int e=0;e<2*elements;e++) for(int a=0;a<f.np;a++) for(int k=0;k<f.fields;k++)
    f.q[(e*f.fields+k)*f.np+a]=f.state[((e%elements)*f.np+a)*12+k];
}

Vec pack(const Fixture& f,const Vec& nodal,const std::vector<int>& ids) {
  Vec result(f.seed.size(),0);
  for(int e:ids) for(int a=0;a<f.np;a++) for(int k=0;k<f.fields;k++)
    result[(e*f.fields+k)*f.np+a]=nodal[(e*f.np+a)*12+k];
  return result;
}

Vec collisionReference(const Fixture& f,const std::vector<int>& ids,int semiAnalytic) {
  const auto& cpu=f.cpu;
  // The reference uses full tensor contractions and a preweighted nodal
  // projection P. OCCA uses staged contractions, raw I^T, JW and invJW.
  Vec cub=cpu.project(f.state,f.n,cpu.nc,cpu.I);
  for(int e=0;e<elements;e++) for(int j=0;j<cpu.nc;j++) for(int i=0;i<cpu.nc;i++) {
    State q=cpu.get(cub,e,j,i,cpu.nc),co{};
    if(f.fields==12)co=hcz::collision(q,cpu.p);
    else {
      co[3]=(q[1]*q[2]/q[0]-(semiAnalytic?0:q[3]))/cpu.p.tauG;
      co[4]=(q[1]*q[1]/(hcz::sq2*q[0])-(semiAnalytic?0:q[4]))/cpu.p.tauG;
      co[5]=(q[2]*q[2]/(hcz::sq2*q[0])-(semiAnalytic?0:q[5]))/cpu.p.tauG;
    }
    for(int k=0;k<12;k++)cub[cpu.id(e,j,i,k,cpu.nc)]=co[k];
  }
  return pack(f,cpu.project(cub,cpu.nc,f.n,cpu.P),ids);
}

double check(const Vec& result,const Vec& expected,const char* label) {
  double error=0,scale=1;
  for(size_t a=0;a<result.size();a++) {
    if(!std::isfinite(result[a]) || !std::isfinite(expected[a]))
      throw std::runtime_error(std::string(label)+": nonfinite value");
    error=std::max(error,std::abs(result[a]-expected[a]));
    scale=std::max(scale,std::abs(expected[a]));
  }
  if(error>2e-12*scale) {
    std::cerr<<label<<": error="<<std::setprecision(17)<<error<<", scale="<<scale<<"\n";
    throw std::runtime_error("relaxation comparison failed");
  }
  return error;
}

struct Errors {double collision=0,assembled=0;};
Errors compare(occa::device& device,occa::kernel& relaxation,occa::kernel& volume,
               occa::kernel& surface,Fixture& f,const Vec& qw,const std::vector<int>& ids,
               int pattern,int semiAnalytic=0) {
  const auto& p=f.cpu.p;
  const int nc=f.cpu.nc, ncp=nc*nc;
  Vec vgeo(elements*6*f.np,0),cubvgeo(elements*6*ncp,0);
  Vec interp(f.n*nc),project(f.n*nc);
  for(int a=0;a<nc;a++) for(int i=0;i<f.n;i++) {
    // Device matrices follow mesh_t::CubatureSetupQuad2D, including its
    // column-major storage. Weights are supplied through the geometry.
    interp[a+i*nc]=f.cpu.I[a*f.n+i];
    project[i+a*f.n]=f.cpu.I[a*f.n+i];
  }
  for(int e=0;e<elements;e++) {
    // Different J across elements in relaxation-only tests checks element
    // indexing and cancellation. Assembly uses the transport fixture's J.
    double J=hx*hy*.25*(f.fields==6?1+.17*e:1);
    for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++) {
      int base=e*6*f.np+j*f.n+i;
      vgeo[base]=2/hx;vgeo[base+3*f.np]=2/hy;
      vgeo[base+4*f.np]=J*f.cpu.weights[i]*f.cpu.weights[j];
      vgeo[base+5*f.np]=1/vgeo[base+4*f.np];
    }
    for(int j=0;j<nc;j++) for(int i=0;i<nc;i++)
      cubvgeo[e*6*ncp+4*ncp+j*nc+i]=J*qw[i]*qw[j];
  }
  auto o_ids=device.malloc<int>(ids.size(),ids.data());
  auto o_vgeo=device.malloc<double>(vgeo.size(),vgeo.data());
  auto o_cubvgeo=device.malloc<double>(cubvgeo.size(),cubvgeo.data());
  auto o_interp=device.malloc<double>(interp.size(),interp.data());
  auto o_project=device.malloc<double>(project.size(),project.data());
  auto o_q=device.malloc<double>(f.q.size(),f.q.data());
  auto o_rhs=device.malloc<double>(f.seed.size(),f.seed.data());
  auto launch=[&]() {
    if(f.fields==12)
      relaxation(int(ids.size()),o_ids,o_vgeo,o_cubvgeo,o_interp,o_project,
                 0,1/p.tauG,p.c,1/p.tauPhi,p.rhoL,p.rhoH,p.phiL,p.phiH,o_q,o_rhs);
    else
      relaxation(int(ids.size()),o_ids,o_vgeo,o_cubvgeo,o_interp,o_project,
                 semiAnalytic,1/p.tauG,o_q,o_rhs);
  };
  Vec co=collisionReference(f,ids,semiAnalytic),result(f.seed.size());
  if(f.fields==12 && pattern==0)check(co,Vec(co.size(),0),"uniform equilibrium");
  if(f.fields==12 && pattern==1) {
    Vec constant(f.state.size(),0);
    State analytic=hcz::collision(f.cpu.get(f.state,0,0,0,f.n),p);
    for(int e=0;e<elements;e++) for(int a=0;a<f.np;a++) for(int k=0;k<12;k++)
      constant[(e*f.np+a)*12+k]=analytic[k];
    check(co,pack(f,constant,ids),"constant nonequilibrium projection");
  }
  Errors errors;
  auto checkUnchanged=[&](const Vec& before) {
    for(int e=0;e<elements;e++) for(int k=0;k<f.fields;k++) for(int a=0;a<f.np;a++) {
      bool selected=std::find(ids.begin(),ids.end(),e)!=ids.end();
      bool conserved=f.fields==12?(k==0 || k==6 || k==7 || k==8):k<3;
      size_t id=(e*f.fields+k)*f.np+a;
      if((!selected || conserved) && result[id]!=before[id])
        throw std::runtime_error("relaxation changed a conserved or unselected RHS entry");
    }
  };
  for(int repeat=1;repeat<=2;repeat++) {
    launch();o_rhs.copyTo(result.data());
    Vec expected=f.seed;
    for(size_t a=0;a<expected.size();a++)expected[a]+=repeat*co[a];
    errors.collision=std::max(errors.collision,check(result,expected,"additive collision"));
    checkUnchanged(f.seed);
  }
  if(f.fields==12) {
    auto o_D=device.malloc<double>(f.cpu.D.size(),f.cpu.D.data());
    auto o_sgeo=device.malloc<double>(f.geometry.size(),f.geometry.data());
    auto o_mapM=device.malloc<int>(f.mapM.size(),f.mapM.data());
    auto o_mapP=device.malloc<int>(f.mapP.size(),f.mapP.data());
    auto o_bc=device.malloc<int>(f.boundaries.size(),f.boundaries.data());
    auto o_xyz=device.malloc<double>(f.coordinates.size(),f.coordinates.data());
    auto o_lift=device.malloc<double>(1);
    o_rhs.copyFrom(f.seed.data());
    volume(int(ids.size()),o_ids,o_vgeo,o_D,o_xyz,o_xyz,o_xyz,0.0,p.c,.01,o_q,o_rhs);
    surface(int(ids.size()),o_ids,o_sgeo,o_lift,o_mapM,o_mapP,o_bc,
            o_xyz,o_xyz,o_xyz,0.0,p.c,.01,o_q,o_rhs);
    Vec before(result.size());o_rhs.copyTo(before.data());
    launch();o_rhs.copyTo(result.data());checkUnchanged(before);
    // Use the actual CPU DG operator and subtract its separate sources.
    // No HCZ gradient/source kernel is part of this increment.
    Vec expectedState=f.cpu.dg(f.state),sources(f.state.size(),0);
    f.cpu.addSources(f.state,sources,false);
    for(size_t a=0;a<sources.size();a++)expectedState[a]-=sources[a];
    Vec expected=pack(f,expectedState,ids);
    for(int e=0;e<elements;e++) if(std::find(ids.begin(),ids.end(),e)==ids.end())
      for(int k=0;k<f.fields;k++) for(int a=0;a<f.np;a++)expected[(e*f.fields+k)*f.np+a]=f.seed[(e*f.fields+k)*f.np+a];
    errors.assembled=check(result,expected,"volume+surface+collision");
    if(ids.size()==elements) {
      double mass=0;
      for(int e=0;e<elements;e++) for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++)
        mass+=result[e*12*f.np+j*f.n+i]*hx*hy*.25*f.cpu.weights[i]*f.cpu.weights[j];
      if(std::abs(mass)>2e-11)throw std::runtime_error("assembled index mass not conserved");
    }
  }
  Vec qAfter(f.q.size());o_q.copyTo(qAfter.data());
  if(qAfter!=f.q)throw std::runtime_error("relaxation/assembly changed input state");
  return errors;
}
} // namespace

int main(int argc,char** argv) {
  if(argc<2 || argc>3) {std::cerr<<"Usage: hczRelaxationQuad2D REPO_ROOT [Serial|OpenMP|CUDA]\n";return 2;}
  const std::string root=argv[1],mode=argc==3?argv[2]:"Serial";
  try {
    occa::device device;occa::json config;config["mode"]=mode;
    if(mode=="CUDA")config["device_id"]=0;
    device.setup(config);
    const std::vector<std::vector<int>> lists={{4,1,5,0,3,2},{5,0,3},{2}};
    for(int degree: {1,2,3}) {
      Vec r,w,qr,qw;quadrature(degree,r,w,qr,qw);
      auto props=relaxationProperties(root,degree,12);
      auto relaxation=device.buildKernel(root+"/solvers/bns/okl/bnsRelaxationQuad2D.okl","bnsRelaxationQuad2D",props);
      auto volume=device.buildKernel(root+"/solvers/bns/okl/bnsVolumeQuad2D.okl","bnsVolumeQuad2D",props);
      auto surface=device.buildKernel(root+"/solvers/bns/okl/bnsSurfaceQuad2D.okl","bnsSurfaceQuad2D",props);
      Errors largest;
      for(int parameters=0;parameters<2;parameters++) {
        hcz::Parameters p;p.c=parameters?.7:1.3;
        p.tauPhi=parameters?.0006:.013;p.tauG=parameters?.0011:.027;
        p.rhoL=parameters?.8:1.2;p.rhoH=parameters?2.2:3.7;
        p.phiL=parameters?.02:.03;p.phiH=parameters?.30:.26;
        p.gravity=0;
        for(bool walls: {false,true}) for(const auto& ids:lists) for(int pattern=0;pattern<5;pattern++) {
          Fixture f(degree,12,walls,r,w,qr,qw);fill(f,p,pattern);
          Errors error=compare(device,relaxation,volume,surface,f,qw,ids,pattern);
          largest.collision=std::max(largest.collision,error.collision);
          largest.assembled=std::max(largest.assembled,error.assembled);
        }
      }
      auto legacy=device.buildKernel(root+"/solvers/bns/okl/bnsRelaxationQuad2D.okl",
                                     "bnsRelaxationQuad2D",relaxationProperties(root,degree,6));
      for(const auto& ids:lists) for(int semiAnalytic: {0,1}) {
        Fixture f(degree,6,false,r,w,qr,qw);hcz::Parameters p;p.c=1.3;p.tauG=.027;
        fill(f,p,2);compare(device,legacy,volume,surface,f,qw,ids,2,semiAnalytic);
      }
      std::cout<<mode<<" N="<<degree<<": max HCZ collision error="<<std::setprecision(4)<<largest.collision
               <<", assembled error="<<largest.assembled
               <<"; equilibria, cubature, unequal rates, density maps, coupled velocity, conserved moments, additive RHS, subsets, partial blocks, mass and legacy explicit/semi-analytic BNS passed\n";
    }
  } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
  return 0;
}
