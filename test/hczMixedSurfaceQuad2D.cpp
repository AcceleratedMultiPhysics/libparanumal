// Mixed face corrections and an assembled spatial hybrid RHS versus hybrid().
#include "hczQuad2DKernelTest.hpp"
#include <cstring>
using namespace hczKernelTest;

namespace {
constexpr double sentinel=-987.25;
struct Check {
  occa::device& device;
  Fixture f;
  occa::kernel mixed,project,scale,weno,rhsKernel,thermo,gradient,source,reconstruct;
  std::vector<int> all={4,1,5,0,3,2},neighbors;
  occa::memory o_neighbors,o_R,o_weights,o_all;
  int cases=0;
  double maxCorrection=0,maxHybrid=0,maxMass=0;
  Check(occa::device& d,const std::string& root,int degree,bool walls,
        const Vec& r,const Vec& w,const Vec& qr,const Vec& qw):
    device(d),f(degree,12,walls,r,w,qr,qw),neighbors(elements*9) {
    f.cpu.total=2*elements;
    f.cpu.exchange=[](Vec& v,int stride){std::copy_n(v.begin(),elements*stride,v.begin()+elements*stride);};
    for(int e=0;e<elements;e++)for(int k=0;k<9;k++) {
      if(k!=4 && f.cpu.neighbor[e][k]>=0 && (e+k)%2)f.cpu.neighbor[e][k]+=elements;
      neighbors[9*e+k]=f.cpu.neighbor[e][k];
    }
    o_neighbors=device.malloc<int>(neighbors.size(),neighbors.data());o_all=ids(all);
    Vec R(f.n*f.cpu.ns);
    for(int a=0;a<f.cpu.ns;a++)for(int i=0;i<f.n;i++)R[i+a*f.n]=f.cpu.R[i*f.cpu.ns+a];
    o_R=device.malloc<double>(R.size(),R.data());o_weights=device.malloc<double>(f.n,f.cpu.weights.data());
    auto props=kernelProperties(root,degree,12);
    auto build=[&](const char* file,const char* name){return device.buildKernel(root+"/solvers/bns/okl/"+file+".okl",name,props);};
    mixed=build("bnsHczMixedSurfaceQuad2D","bnsHczMixedSurfaceQuad2D");
    project=build("bnsHczSubcellQuad2D","bnsHczProjectSubcellQuad2D");scale=build("bnsHczSubcellQuad2D","bnsHczScaleSubcellQuad2D");
    weno=build("bnsHczWenoFaceQuad2D","bnsHczWenoFaceQuad2D");rhsKernel=build("bnsHczSubcellRhsQuad2D","bnsHczSubcellRhsQuad2D");
    gradient=build("bnsHczSubcellGradientQuad2D","bnsHczSubcellGradientQuad2D");reconstruct=build("bnsHczReconstructQuad2D","bnsHczReconstructQuad2D");
    props["defines/p_Nq"]=f.cpu.ns;props["defines/p_Np"]=f.cpu.ns*f.cpu.ns;
    props["defines/p_NblockV"]=std::max(1,256/(f.cpu.ns*f.cpu.ns));
    thermo=build("bnsHczThermodynamicsQuad2D","bnsHczThermodynamicsQuad2D");source=build("bnsHczSourceQuad2D","bnsHczSourceQuad2D");
  }
  occa::memory ids(const std::vector<int>& values) {
    auto out=device.malloc<int>(std::max(size_t(1),values.size()));
    if(!values.empty())out.copyFrom(values.data(),values.size()*sizeof(int));
    return out;
  }
  Vec native(const Vec& q,int n) const {
    Vec out(q.size());
    for(int e=0;e<2*elements;e++)for(int node=0;node<n*n;node++)for(int k=0;k<12;k++)
      out[(e*12+k)*n*n+node]=q[(e*n*n+node)*12+k];
    return out;
  }
  size_t fid(int e,int axis,int k,int t,int a) const {
    const int ns=f.cpu.ns;return (((size_t(e)*2+axis)*12+k)*ns+t)*(ns+1)+a;
  }
  Vec state() const {
    Vec q(2*elements*12*f.np,0);
    for(int e=0;e<elements;e++)for(int j=0;j<f.n;j++)for(int i=0;i<f.n;i++) {
      State s{};s[0]=.15+.015*std::sin(.3*e+.7*i+.4*j);s[6]=0;
      s[7]=.015*std::sin(i+.3*e);s[8]=.013*std::cos(j+.4*e);
      for(int k: {1,2,3,4,5,9,10,11})s[k]=.009*(k-4)*std::sin(.2*k+i+.7*j+.4*e);
      for(int k=0;k<12;k++)q[f.cpu.id(e,j,i,k,f.n)]=s[k];
    }
    f.cpu.exchange(q,12*f.np);return q;
  }
  Vec mask(int pattern) const {
    Vec v(2*elements,0);
    for(int e=0;e<elements;e++)v[e]=pattern==1 || (pattern==2 && e==1) || (pattern==3 && e%2)
      || (pattern==4 && e%nxElements==0) || (pattern==5 && e>=nxElements);
    f.cpu.exchange(v,1);return v;
  }
  void compare(double value,double want,double& maximum,const char* label) {
    double error=std::abs(value-want);maximum=std::max(maximum,error);
    if(!std::isfinite(value) || error>8e-11*std::max(1.0,std::abs(want)))throw std::runtime_error(label);
  }
  void halo(occa::memory& buffer,int stride) {
    Vec values(2*elements*stride);buffer.copyTo(values.data());
    buffer.copyFrom(values.data(),elements*stride*sizeof(double),elements*stride*sizeof(double));
  }
  void test(const Vec& hostMask,const std::vector<int>& selected,bool full) {
    const auto& p=f.cpu.p;const int ns=f.cpu.ns,np=ns*ns;
    auto q=state(),base=f.cpu.dg(q),reference=f.cpu.hybrid(q,base,hostMask);
    Vec qNative=native(q,f.n),expected=native(reference,f.n),seed=native(base,f.n);
    std::vector<int> marked,flags(2*elements);
    for(int e=0;e<2*elements;e++) {flags[e]=int(hostMask[e]);if(e<elements && flags[e])marked.push_back(e);}
    for(size_t i=elements*12*f.np;i<seed.size();i++)seed[i]=sentinel;
    // The isolated correction starts with already replaced marked elements.
    if(!full)for(int e:marked)for(int a=0;a<12*f.np;a++)seed[e*12*f.np+a]=expected[e*12*f.np+a];
    auto o_q=device.malloc<double>(qNative.size(),qNative.data()),o_mask=device.malloc<int>(flags.size(),flags.data());
    auto o_ids=ids(selected),o_rhs=device.malloc<double>(seed.size(),seed.data());
    Vec faceBefore(elements*2*12*ns*(ns+1),std::numeric_limits<double>::quiet_NaN());
    auto o_faces=device.malloc<double>(faceBefore.size(),faceBefore.data());
    if(full) {
      Vec A(f.n*ns);for(int i=0;i<f.n;i++)for(int a=0;a<ns;a++)A[a+i*ns]=f.cpu.A[a*f.n+i];
      auto o_A=device.malloc<double>(A.size(),A.data()),o_sub=device.malloc<double>(2*elements*12*np);
      auto o_invalid=device.malloc<int>(elements);
      project(elements,o_all,o_A,o_q,o_sub);scale(elements,o_all,p.floor(),o_sub,o_invalid);
      std::vector<int> invalid(elements);o_invalid.copyTo(invalid.data());
      if(std::any_of(invalid.begin(),invalid.end(),[](int x){return x!=0;}))throw std::runtime_error("invalid mean in hybrid fixture");
      halo(o_sub,12*np);
      weno(elements,o_all,o_neighbors,p.c,p.alpha,p.wenoEps,o_sub,o_faces);
      auto o_fv=device.malloc<double>(2*elements*12*np),o_rho=device.malloc<double>(elements*np);
      auto o_p=device.malloc<double>(2*elements*2*np),o_g=device.malloc<double>(elements*4*np);
      rhsKernel(elements,o_all,p.c,p.tauPhi,p.tauG,p.rhoL,p.rhoH,p.phiL,p.phiH,f.cpu.hx,f.cpu.hy,o_sub,o_faces,o_fv);
      thermo(elements,o_all,p.c,p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.eosRatio,o_sub,o_rho,o_p);
      halo(o_p,2*np);gradient(elements,o_all,o_neighbors,f.cpu.hx,f.cpu.hy,o_p,o_g);
      source(elements,o_all,p.c,p.gravity,p.rhoL,p.rhoH,p.phiL,p.phiH,o_sub,o_g,o_fv);
      auto o_marked=ids(marked);
      if(!marked.empty())reconstruct(int(marked.size()),o_marked,o_R,o_weights,o_fv,o_rhs);
      o_faces.copyTo(faceBefore.data());
    }else{
      auto sub=f.cpu.project(q,f.n,ns,f.cpu.A);
      if(!f.cpu.scaleSubcellIndex(sub))throw std::runtime_error("invalid mean in face fixture");
      f.cpu.exchange(sub,12*np);
      // Poison all unused faces: walls, DG/DG faces, and faces of marked
      // elements must never be read by the mixed correction kernel.
      for(int e=0;e<elements;e++)if(!flags[e])for(int axis=0;axis<2;axis++)for(int sign: {-1,1}) {
        int nb=neighbors[9*e+4+sign*(axis?3:1)];if(nb<0 || !flags[nb])continue;
        int a=sign<0?0:ns;
        for(int t=0;t<ns;t++) {
          auto flux=f.cpu.wenoFace(sub,e,axis?a-1:t,axis?t:a-1,axis);
          for(int k=0;k<12;k++)faceBefore[fid(e,axis,k,t,a)]=flux[k];
        }
      }
      o_faces.copyFrom(faceBefore.data());
    }
    Vec before(seed.size());o_rhs.copyTo(before.data());
    const int repeats=full?1:2;
    for(int repeat=1;repeat<=repeats;repeat++) {
      mixed(int(selected.size()),o_ids,o_neighbors,o_mask,o_R,o_weights,p.c,f.cpu.hx,f.cpu.hy,o_q,o_faces,o_rhs);
      Vec actual(seed.size());o_rhs.copyTo(actual.data());
      for(int e=0;e<2*elements;e++)for(int k=0;k<12;k++)for(int node=0;node<f.np;node++) {
        size_t pos=(e*12+k)*f.np+node;
        const bool chosen=std::find(selected.begin(),selected.end(),e)!=selected.end();
        double want=before[pos];
        if(full && e<elements)want=expected[pos];
        else if(chosen && !flags[e])want=seed[pos]+repeat*(expected[pos]-seed[pos]);
        compare(actual[pos],want,full?maxHybrid:maxCorrection,"mixed DG/subcell residual differs from host hybrid");
        // Marked/unselected elements, interiors and nodes without an active
        // mixed face must remain bitwise unchanged by the correction itself.
        bool active=false;
        if(e<elements && chosen && !flags[e])for(int axis=0;axis<2;axis++)for(int sign: {-1,1}) {
          int nb=neighbors[9*e+4+sign*(axis?3:1)],a=sign<0?0:f.n-1;
          active |= nb>=0 && flags[nb] && (axis?node/f.n:node%f.n)==a;
        }
        if(!active && actual[pos]!=before[pos])throw std::runtime_error("mixed correction changed an inactive entry");
      }
      if(full) {
        double integral=0;
        for(int e=0;e<elements;e++)for(int j=0;j<f.n;j++)for(int i=0;i<f.n;i++)
          integral+=.25*f.cpu.hx*f.cpu.hy*f.cpu.weights[j]*f.cpu.weights[i]*actual[e*12*f.np+j*f.n+i];
        maxMass=std::max(maxMass,std::abs(integral));
        if(std::abs(integral)>2e-11)throw std::runtime_error("hybrid spatial RHS lost index conservation");
      }
    }
    Vec after(qNative.size());o_q.copyTo(after.data());if(after!=qNative)throw std::runtime_error("mixed correction changed DG state");
    after.resize(faceBefore.size());o_faces.copyTo(after.data());
    if(std::memcmp(after.data(),faceBefore.data(),after.size()*sizeof(double)))throw std::runtime_error("mixed correction changed WENO faces");
    std::vector<int> afterMask(flags.size());o_mask.copyTo(afterMask.data());if(afterMask!=flags)throw std::runtime_error("mixed correction changed mask");
    cases++;
  }
  void tests() {
    auto& p=f.cpu.p;p.kappa=0;
    for(int params=0;params<2;params++) {
      p.c=params?.7:1.3;p.rhoL=params?.7:1;p.rhoH=params?1.2:3;
      p.tauPhi=.0025;p.tauG=.0017;p.gravity=f.walls?.04:0;
      f.cpu.hx=params?.19:.31;f.cpu.hy=params?.73:.47;
      for(int pattern=0;pattern<6;pattern++){test(mask(pattern),all,false);test(mask(pattern),all,true);}
      for(const auto& list:std::vector<std::vector<int>>{{5,0,3},{2},{}})test(mask(3),list,false);
    }
    // Make the face mean correction appreciable. The production
    // pseudoinverse nearly preserves means even without the explicit fix.
    f.cpu.R[0]+=.07;
    Vec biased(f.n*f.cpu.ns);
    for(int a=0;a<f.cpu.ns;a++)for(int i=0;i<f.n;i++)biased[i+a*f.n]=f.cpu.R[i*f.cpu.ns+a];
    o_R.copyFrom(biased.data());
    test(mask(2),all,false);test(mask(3),all,true);
  }
};
}
int main(int argc,char** argv) {
  if(argc<2 || argc>3)return 2;
  const std::string root=argv[1],mode=argc==3?argv[2]:"Serial";
  try {
    occa::device device;occa::json config;config["mode"]=mode;if(mode=="CUDA")config["device_id"]=0;
    device.setup(config);
    for(int degree: {1,2,3,4})for(bool walls: {false,true}) {
      Vec r,w,qr,qw;quadrature(degree,r,w,qr,qw);
      if(degree==4){r={-1,-std::sqrt(3./7),0,std::sqrt(3./7),1};w={.1,49./90,32./45,49./90,.1};}
      Check check(device,root,degree,walls,r,w,qr,qw);check.tests();
      std::cout<<mode<<" N="<<degree<<(walls?" walls":" periodic")<<": "<<check.cases
        <<" cases passed; correction="<<check.maxCorrection<<", hybrid="<<check.maxHybrid<<", index balance="<<check.maxMass<<"\n";
    }
  } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
  return 0;
}
