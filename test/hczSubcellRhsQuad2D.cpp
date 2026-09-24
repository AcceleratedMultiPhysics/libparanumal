// Isolated pointwise BGK and subcell flux divergence; no force sources yet.
#include "hczQuad2DKernelTest.hpp"
using namespace hczKernelTest;

namespace {
constexpr double sentinel=-987.25;
struct Check {
  occa::device& device;
  Fixture f;
  occa::kernel rhsKernel,faceKernel;
  std::vector<int> all={4,1,5,0,3,2};
  double maxError=0,maxBalance=0;
  int cases=0;
  Check(occa::device& d,const std::string& root,int degree,bool walls,
        const Vec& r,const Vec& w,const Vec& qr,const Vec& qw):
    device(d),f(degree,12,walls,r,w,qr,qw) {
    f.cpu.total=2*elements;
    for(int e=0;e<elements;e++)for(int k=0;k<9;k++)
      if(k!=4 && f.cpu.neighbor[e][k]>=0 && (e+k)%2)f.cpu.neighbor[e][k]+=elements;
    auto props=kernelProperties(root,degree,12);
    rhsKernel=device.buildKernel(root+"/solvers/bns/okl/bnsHczSubcellRhsQuad2D.okl","bnsHczSubcellRhsQuad2D",props);
    faceKernel=device.buildKernel(root+"/solvers/bns/okl/bnsHczWenoFaceQuad2D.okl","bnsHczWenoFaceQuad2D",props);
  }
  size_t fid(int e,int axis,int k,int t,int a) const {
    const int ns=f.cpu.ns;
    return (((size_t(e)*2+axis)*12+k)*ns+t)*(ns+1)+a;
  }
  Vec native(const Vec& q) const {
    const int np=f.cpu.ns*f.cpu.ns;Vec out(q.size());
    for(int e=0;e<2*elements;e++)for(int node=0;node<np;node++)for(int k=0;k<12;k++)
      out[(e*12+k)*np+node]=q[(e*np+node)*12+k];
    return out;
  }
  Vec state(bool equilibrium=false,bool moving=false) const {
    const int ns=f.cpu.ns;const auto& p=f.cpu.p;
    Vec sub(2*elements*ns*ns*12);
    for(int e=0;e<elements;e++)for(int j=0;j<ns;j++)for(int i=0;i<ns;i++) {
      State q{};
      q[0]=equilibrium?.15:.15+.035*std::sin(.7*i+.5*j+e);
      q[6]=0; // Independent pressure can be zero; never divide by it.
      q[7]=equilibrium?(moving?.07*p.rho(q[0])*p.c:0):.03*std::sin(i+.3*e);
      q[8]=equilibrium?(moving?-.035*p.rho(q[0])*p.c:0):.025*std::cos(j+.5*e);
      if(equilibrium)q=hcz::equilibrium(q,p);
      else for(int k: {1,2,3,4,5,9,10,11})q[k]=.015*(k-4)*std::sin(.2*k+i+.7*j+.4*e);
      for(int k=0;k<12;k++)sub[f.cpu.id(e,j,i,k,ns)]=q[k];
    }
    std::copy_n(sub.begin(),elements*ns*ns*12,sub.begin()+elements*ns*ns*12);
    return sub;
  }
  // mode 0: constant faces (BGK only), 1: linear x, 2: linear y,
  // 3: both. Physical normal coordinates make the expected divergence
  // independent of ns and distinguish hx from hy.
  Vec manufactured(int mode) const {
    const int ns=f.cpu.ns;Vec faces(elements*2*12*ns*(ns+1));
    for(int e=0;e<elements;e++)for(int axis=0;axis<2;axis++)for(int k=0;k<12;k++)
      for(int t=0;t<ns;t++)for(int a=0;a<=ns;a++) {
        const double slope=(mode&(axis?2:1))?(axis?-.031:.047)*(k+1):0;
        faces[fid(e,axis,k,t,a)]=.2*(e+1)+.007*t+slope*a*(axis?f.cpu.hy:f.cpu.hx)/ns;
      }
    return faces;
  }
  Vec run(const Vec& sub,const Vec& faces,const std::vector<int>& ids,
          int manufacturedMode=-1,bool equilibrium=false,occa::memory* supplied=nullptr) {
    const int ns=f.cpu.ns,np=ns*ns;const auto& p=f.cpu.p;
    Vec input=native(sub),seed((elements+2)*12*np,sentinel),expected=seed,actual(seed.size());
    for(int e:ids)for(int j=0;j<ns;j++)for(int i=0;i<ns;i++) {
      State co=hcz::collision(f.cpu.get(sub,e,j,i,ns),p);
      for(int k=0;k<12;k++)expected[(e*12+k)*np+j*ns+i]=co[k]
        -(faces[fid(e,0,k,j,i+1)]-faces[fid(e,0,k,j,i)])/(f.cpu.hx/ns)
        -(faces[fid(e,1,k,i,j+1)]-faces[fid(e,1,k,i,j)])/(f.cpu.hy/ns);
    }
    auto o_ids=device.malloc<int>(std::max(size_t(1),ids.size()));
    if(!ids.empty())o_ids.copyFrom(ids.data(),ids.size()*sizeof(int));
    auto o_sub=device.malloc<double>(input.size(),input.data());
    auto o_faces=supplied?*supplied:device.malloc<double>(faces.size(),faces.data());
    Vec faceBefore(faces.size());o_faces.copyTo(faceBefore.data());
    auto o_rhs=device.malloc<double>(seed.size(),seed.data());
    for(int repeat=0;repeat<2;repeat++) {
      rhsKernel(int(ids.size()),o_ids,p.c,p.tauPhi,p.tauG,p.rhoL,p.rhoH,p.phiL,p.phiH,
                f.cpu.hx,f.cpu.hy,o_sub,o_faces,o_rhs);
      o_rhs.copyTo(actual.data());
      for(size_t i=0;i<actual.size();i++) {
        const double error=std::abs(actual[i]-expected[i]);maxError=std::max(maxError,error);
        if(!std::isfinite(actual[i]) || error>3e-11*std::max(1.0,std::abs(expected[i])))
          throw std::runtime_error("subcell BGK/divergence differs from host");
      }
      for(int e=0;e<elements+2;e++) {
        bool selected=std::find(ids.begin(),ids.end(),e)!=ids.end();
        for(int node=0;node<np;node++)for(int k=0;k<12;k++) {
          double value=actual[(e*12+k)*np+node];
          if(!selected && value!=sentinel)throw std::runtime_error("residual overwrote unselected/halo sentinel");
          if(!selected)continue;
          if(equilibrium && std::abs(value)>2e-11)throw std::runtime_error("uniform equilibrium has nonzero residual");
          if(manufacturedMode>=0) {
            const auto co=hcz::collision(f.cpu.get(sub,e,node/ns,node%ns,ns),p);
            double analytic=co[k]-(manufacturedMode&1?.047*(k+1):0)-(manufacturedMode&2?-.031*(k+1):0);
            if(std::abs(value-analytic)>3e-11)throw std::runtime_error("analytic flux divergence mismatch");
            if(manufacturedMode==0 && (k==0 || k==6 || k==7 || k==8) && value!=0)
              throw std::runtime_error("BGK changed a conserved moment");
          }
        }
      }
    }
    Vec after(input.size());o_sub.copyTo(after.data());
    if(after!=input)throw std::runtime_error("residual kernel modified prepared subcells");
    after.resize(faces.size());o_faces.copyTo(after.data());
    if(after!=faceBefore)throw std::runtime_error("residual kernel modified stored WENO faces");
    if(ids.size()==elements)for(int k: {0,6,7,8}) {
      double balance=0;
      for(int e=0;e<elements;e++) {
        for(int node=0;node<np;node++)balance+=actual[(e*12+k)*np+node]*f.cpu.hx*f.cpu.hy/np;
        for(int t=0;t<ns;t++) {
          balance+=(faceBefore[fid(e,0,k,t,ns)]-faceBefore[fid(e,0,k,t,0)])*f.cpu.hy/ns;
          balance+=(faceBefore[fid(e,1,k,t,ns)]-faceBefore[fid(e,1,k,t,0)])*f.cpu.hx/ns;
        }
      }
      maxBalance=std::max(maxBalance,std::abs(balance));
      if(std::abs(balance)>2e-12)throw std::runtime_error("conserved-moment residual does not telescope to boundary flux");
    }
    cases++;return actual;
  }
  void pipeline(const Vec& sub,bool equilibrium=false) {
    const int ns=f.cpu.ns;const auto& p=f.cpu.p;
    Vec faces(elements*2*12*ns*(ns+1));
    for(int e=0;e<elements;e++)for(int axis=0;axis<2;axis++)for(int t=0;t<ns;t++)for(int a=0;a<=ns;a++) {
      auto flux=f.cpu.wenoFace(sub,e,axis?a-1:t,axis?t:a-1,axis);
      for(int k=0;k<12;k++)faces[fid(e,axis,k,t,a)]=flux[k];
    }
    Vec input=native(sub);std::vector<int> neighbors(elements*9);
    for(int e=0;e<elements;e++)std::copy(f.cpu.neighbor[e].begin(),f.cpu.neighbor[e].end(),neighbors.begin()+9*e);
    auto o_neighbors=device.malloc<int>(neighbors.size(),neighbors.data()),o_ids=device.malloc<int>(all.size(),all.data());
    auto o_sub=device.malloc<double>(input.size(),input.data()),o_faces=device.malloc<double>(faces.size());
    faceKernel(elements,o_ids,o_neighbors,p.c,p.alpha,p.wenoEps,o_sub,o_faces);
    auto rhs=run(sub,faces,all,-1,equilibrium,&o_faces);
    for(int k: {0,6}) {
      double sum=0;for(int e=0;e<elements;e++)for(int node=0;node<ns*ns;node++)sum+=rhs[(e*12+k)*ns*ns+node]*f.cpu.hx*f.cpu.hy/(ns*ns);
      if(std::abs(sum)>2e-12)throw std::runtime_error("closed WENO/BGK sequence lost index or pressure-family integral");
    }
  }
  void tests() {
    for(int params=0;params<2;params++) {
      auto& p=f.cpu.p;
      p.c=params?.7:1.3;p.tauPhi=params?.0031:.0025;p.tauG=params?.0013:.0017;
      p.rhoL=params?.7:1;p.rhoH=params?1.2:3;
      p.phiL=params?.02:.02381;p.phiH=params?.3:.2508;
      f.cpu.hx=params?.19:.31;f.cpu.hy=params?.73:.47;
      const Vec sub=state();
      for(int mode=0;mode<4;mode++)run(sub,manufactured(mode),all,mode);
      for(const auto& ids:std::vector<std::vector<int>>{{5,0,3},{2},{}})run(sub,manufactured(3),ids,3);
      run(state(true),manufactured(0),all,0,true);
      pipeline(sub);pipeline(state(true),true);
      if(!f.walls)pipeline(state(true,true),true);
      // Independent relaxation-time changes must not affect conserved BGK
      // entries; the all-moment host comparison checks the other eight rates.
      p.tauPhi*=2;run(sub,manufactured(0),all,0);
      p.tauG*=3;run(sub,manufactured(0),all,0);
    }
  }
};
}

int main(int argc,char** argv) {
  if(argc<2 || argc>3)return 2;
  const std::string root=argv[1],mode=argc==3?argv[2]:"Serial";
  try {
    occa::device device;occa::json config;config["mode"]=mode;
    if(mode=="CUDA")config["device_id"]=0;
    device.setup(config);
    for(int degree: {1,2,3,4})for(bool walls: {false,true}) {
      Vec r,w,qr,qw;quadrature(degree,r,w,qr,qw);
      if(degree==4){r={-1,-std::sqrt(3./7),0,std::sqrt(3./7),1};w={.1,49./90,32./45,49./90,.1};}
      Check check(device,root,degree,walls,r,w,qr,qw);check.tests();
      std::cout<<mode<<" N="<<degree<<(walls?" walls":" periodic")<<": "<<check.cases
               <<" cases passed; max residual error="<<check.maxError<<", balance="<<check.maxBalance<<"\n";
    }
  } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
  return 0;
}
