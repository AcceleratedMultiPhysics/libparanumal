// Isolated WENO3-Z faces versus the host routine used by hybrid().
#include "hczQuad2DKernelTest.hpp"
#include <cstring>
using namespace hczKernelTest;

namespace {
constexpr double sentinel=-987.25;
struct Check {
  occa::device& device;
  Fixture f;
  occa::kernel kernel,project,scale;
  std::vector<int> all={4,1,5,0,3,2};
  int cases=0;
  double maxError=0,maxBalance=0,maxInterfaceError=0;

  Check(occa::device& d,const std::string& root,int degree,bool walls,
        const Vec& r,const Vec& w,const Vec& qr,const Vec& qw,bool characteristic):
    device(d),f(degree,12,walls,r,w,qr,qw) {
    f.cpu.total=2*elements;
    // Supply explicit halo slots for selected neighbors, including periodic
    // wraps. Wall entries remain -1; the host and kernel must mirror locally.
    for(int e=0;e<elements;e++)for(int k=0;k<9;k++)
      if(k!=4 && f.cpu.neighbor[e][k]>=0 && (e+k)%2)
        f.cpu.neighbor[e][k]+=elements;
    f.cpu.p.characteristicWeno=characteristic;
    auto props=kernelProperties(root,degree,12);
    props["defines/p_HCZ_CHARACTERISTIC_WENO"]=characteristic?1:0;
    kernel=device.buildKernel(root+"/solvers/bns/okl/bnsHczWenoFaceQuad2D.okl","bnsHczWenoFaceQuad2D",props);
    const auto file=root+"/solvers/bns/okl/bnsHczSubcellQuad2D.okl";
    project=device.buildKernel(file,"bnsHczProjectSubcellQuad2D",props);
    scale=device.buildKernel(file,"bnsHczScaleSubcellQuad2D",props);
  }

  size_t faceId(int e,int axis,int k,int tangent,int normal) const {
    const int ns=f.cpu.ns;
    return (((size_t(e)*2+axis)*12+k)*ns+tangent)*(ns+1)+normal;
  }
  void halos(Vec& sub) const {
    const int count=elements*f.cpu.ns*f.cpu.ns*12;
    std::copy_n(sub.begin(),count,sub.begin()+count);
  }
  Vec native(const Vec& q,int n) const {
    Vec out(q.size());
    for(int e=0;e<2*elements;e++)for(int node=0;node<n*n;node++)for(int k=0;k<12;k++)
      out[(e*12+k)*n*n+node]=q[(e*n*n+node)*12+k];
    return out;
  }
  Vec make(int pattern) const {
    const int ns=f.cpu.ns;Vec sub(2*elements*ns*ns*12);
    for(int e=0;e<elements;e++)for(int j=0;j<ns;j++)for(int i=0;i<ns;i++)for(int k=0;k<12;k++) {
      double v=.025*(k+1);
      if(pattern==1)v+=.13*std::sin(.27*(k+1)+.71*i+.53*j+1.13*e);
      if(pattern==2)v=(i<ns/2?.21:-.08)*(k%3+1)+(e%2?.12:-.07);
      if(pattern==3)v=((i+j+e+k)%2?.23:-.17);
      if(pattern==4)v=1e-8*std::sin(.3*(i+4*j+7*e+2*k));
      if(pattern==0 && f.walls && (k==1 || k==2 || k==7 || k==8))v=0;
      sub[f.cpu.id(e,j,i,k,ns)]=v;
    }
    halos(sub);return sub;
  }

  Vec run(const Vec& sub,const std::vector<int>& ids,const char* label,
          bool uniform=false,occa::memory* prepared=nullptr) {
    const int ns=f.cpu.ns;const auto& p=f.cpu.p;
    Vec input=native(sub,ns);
    std::vector<int> neighbors(elements*9);
    for(int e=0;e<elements;e++)std::copy(f.cpu.neighbor[e].begin(),f.cpu.neighbor[e].end(),neighbors.begin()+9*e);
    auto o_ids=device.malloc<int>(std::max(size_t(1),ids.size()));
    if(!ids.empty())o_ids.copyFrom(ids.data(),ids.size()*sizeof(int));
    auto o_neighbors=device.malloc<int>(neighbors.size(),neighbors.data());
    auto o_sub=prepared?*prepared:device.malloc<double>(input.size(),input.data());
    Vec seed((elements+2)*2*12*ns*(ns+1),sentinel),expected=seed,actual(seed.size());
    for(int e:ids)for(int axis=0;axis<2;axis++)for(int t=0;t<ns;t++)for(int a=0;a<=ns;a++) {
      const State flux=f.cpu.wenoFace(sub,e,axis?a-1:t,axis?t:a-1,axis);
      for(int k=0;k<12;k++)expected[faceId(e,axis,k,t,a)]=flux[k];
    }
    auto o_faces=device.malloc<double>(seed.size(),seed.data());
    for(int repeat=0;repeat<2;repeat++) {
      kernel(int(ids.size()),o_ids,o_neighbors,p.c,p.alpha,p.wenoEps,o_sub,o_faces);
      o_faces.copyTo(actual.data());
      for(size_t i=0;i<actual.size();i++) {
        const double error=std::abs(actual[i]-expected[i]);maxError=std::max(maxError,error);
        if(!std::isfinite(actual[i]) || error>2e-11*std::max(1.0,std::abs(expected[i]))) {
          std::cerr<<label<<": face entry "<<i<<", device="<<actual[i]<<", host="<<expected[i]<<"\n";
          throw std::runtime_error("WENO face differs from host");
        }
        if(expected[i]==sentinel && actual[i]!=sentinel)throw std::runtime_error("face kernel overwrote sentinel");
      }
    }
    Vec after(input.size());o_sub.copyTo(after.data());
    // Prepared data can differ from the host projection by roundoff; its
    // exact bitwise preservation is checked separately in pipeline().
    if(!prepared && std::memcmp(after.data(),input.data(),input.size()*sizeof(double)))
      throw std::runtime_error("face kernel modified subcells");
    std::vector<int> afterNb(neighbors.size());o_neighbors.copyTo(afterNb.data());
    if(afterNb!=neighbors)throw std::runtime_error("face kernel modified neighbors");
    if(ids.size()==elements) {
      for(int k: {0,6}) {
        double balance=0;
        for(int e=0;e<elements;e++)for(int t=0;t<ns;t++) {
          balance+=hy/ns*(actual[faceId(e,0,k,t,ns)]-actual[faceId(e,0,k,t,0)]);
          balance+=hx/ns*(actual[faceId(e,1,k,t,ns)]-actual[faceId(e,1,k,t,0)]);
          if(f.walls)for(int a: {0,ns})if(f.cpu.neighbor[e][a?7:1]<0)
            if(std::abs(actual[faceId(e,1,k,t,a)])>2e-12)throw std::runtime_error("wall has nonzero index/pressure-family flux");
        }
        maxBalance=std::max(maxBalance,std::abs(balance));
        if(std::abs(balance)>2e-12)throw std::runtime_error("closed/periodic flux balance failed");
      }
      // Each side computes the same physical face flux, oriented in +x/+y.
      for(int e=0;e<elements;e++)for(int axis=0;axis<2;axis++) {
        const int nb=f.cpu.neighbor[e][axis?7:5];if(nb<0)continue;
        for(int t=0;t<ns;t++)for(int k=0;k<12;k++) {
          const double error=std::abs(actual[faceId(e,axis,k,t,ns)]-actual[faceId(nb%elements,axis,k,t,0)]);
          maxInterfaceError=std::max(maxInterfaceError,error);
          if(error>2e-12)throw std::runtime_error("two sides disagree on shared WENO face");
        }
      }
      if(uniform)for(int e=0;e<elements;e++)for(int axis=0;axis<2;axis++) {
        const auto physical=hcz::flux(f.cpu.get(sub,e,0,0,ns),axis,p.c);
        for(int t=0;t<ns;t++)for(int a=0;a<=ns;a++)for(int k=0;k<12;k++)
          if(std::abs(actual[faceId(e,axis,k,t,a)]-physical[k])>2e-12)
            throw std::runtime_error("uniform WENO flux is not the physical H2 flux");
      }
    }
    cases++;return actual;
  }

  void pipeline() {
    const int ns=f.cpu.ns,np=ns*ns;
    Vec q(2*elements*f.np*12,0);
    int column=-1;
    if(f.n>=3)for(int s=0;s<ns;s++)for(int i=0;i<f.n;i++)if(f.cpu.A[s*f.n+i]<-.02)column=i;
    for(int e=0;e<elements;e++)for(int j=0;j<f.n;j++)for(int i=0;i<f.n;i++)for(int k=0;k<12;k++)
      q[f.cpu.id(e,j,i,k,f.n)]=k==0?(column<0?.15:.01+(i==column?.8:0)):.02*(k-4)*std::sin(i+2*j+.3*e);
    Vec reference=f.cpu.project(q,f.n,ns,f.cpu.A),unscaled=reference;
    if(!f.cpu.scaleSubcellIndex(reference))throw std::runtime_error("invalid pipeline fixture mean");
    if(column>=0 && reference==unscaled)throw std::runtime_error("pipeline fixture must exercise conservative scaling");
    halos(reference);
    Vec qNative=native(q,f.n),mat(ns*f.n);
    for(int i=0;i<f.n;i++)for(int s=0;s<ns;s++)mat[i*ns+s]=f.cpu.A[s*f.n+i];
    auto o_q=device.malloc<double>(qNative.size(),qNative.data()),o_A=device.malloc<double>(mat.size(),mat.data());
    auto o_ids=device.malloc<int>(all.size(),all.data());
    auto o_sub=device.malloc<double>(2*elements*np*12),o_invalid=device.malloc<int>(elements);
    project(elements,o_ids,o_A,o_q,o_sub);
    scale(elements,o_ids,f.cpu.p.floor(),o_sub,o_invalid);
    std::vector<int> invalid(elements);o_invalid.copyTo(invalid.data());
    if(std::any_of(invalid.begin(),invalid.end(),[](int x){return x!=0;}))throw std::runtime_error("device pipeline rejected mean");
    Vec deviceInput(2*elements*np*12);o_sub.copyTo(deviceInput.data());
    // Emulate a halo exchange by filling only the ghost slots; preserve
    // the actual owned device projection/scaling output for the flux call.
    std::copy_n(deviceInput.begin(),elements*np*12,deviceInput.begin()+elements*np*12);
    o_sub.copyFrom(deviceInput.data()+elements*np*12,elements*np*12*sizeof(double),elements*np*12*sizeof(double));
    run(reference,all,"project-scale-face pipeline",false,&o_sub);
    Vec after(deviceInput.size());o_sub.copyTo(after.data());
    if(std::memcmp(after.data(),deviceInput.data(),after.size()*sizeof(double)))throw std::runtime_error("pipeline faces modified prepared state");
  }

  void characteristicTests() {
    const int ns=f.cpu.ns;
    // Independent directional oracle: scalar WENO on a single eigenmode,
    // including arbitrary variation of both zero-speed modes.
    const double eigenvalues[6]={std::sqrt(3.0),-std::sqrt(3.0),1,-1,0,0};
    for(int axis=0;axis<2;axis++)for(int family: {0,6})for(int wave=0;wave<6;wave++) {
      State eigen{};
      if(wave<2) {
        eigen[family]=1/std::sqrt(6.0);
        eigen[family+1+axis]=(wave==0?1:-1)/std::sqrt(2.0);
        eigen[family+4+axis]=1/std::sqrt(3.0);
      } else if(wave<4) {
        eigen[family+2-axis]=1/std::sqrt(2.0);
        eigen[family+3]=(wave==2?1:-1)/std::sqrt(2.0);
      } else if(wave==4) {
        eigen[family]=-std::sqrt(2.0/3.0);eigen[family+4+axis]=1/std::sqrt(3.0);
      } else eigen[family+5-axis]=1;
      Vec sub(2*elements*ns*ns*12),scalar(2*elements*ns*ns*12);
      for(int e=0;e<elements;e++)for(int j=0;j<ns;j++)for(int i=0;i<ns;i++) {
        const double value=.2*std::sin(.7*i+.3*j+e)+(i<ns/2?.4:-.1);
        scalar[f.cpu.id(e,j,i,0,ns)]=value;
        for(int k=0;k<12;k++)sub[f.cpu.id(e,j,i,k,ns)]=value*eigen[k];
      }
      halos(sub);halos(scalar);
      const Vec result=run(sub,all,"single characteristic");
      // Interior faces avoid reflected boundary mixing of eigenvectors.
      for(int e=0;e<elements;e++)for(int t=0;t<ns;t++)for(int a=1;a<ns;a++) {
        // Only stencil-contained faces for walls; periodic cases cover halos.
        if(f.walls && (a<2 || a>ns-2))continue;
        double v[4];
        for(int z=0;z<4;z++)v[z]=f.cpu.sample(scalar,e,axis?a+z-2:t,axis?t:a+z-2,ns)[0];
        const double d0=v[1]-v[0],d1=v[2]-v[1],d2=v[3]-v[2];
        const double recon=wave%2==0?
          f.cpu.blend(1.5*v[1]-.5*v[0],.5*(v[1]+v[2]),d0*d0,d1*d1,1./3):
          f.cpu.blend(.5*(v[1]+v[2]),1.5*v[2]-.5*v[3],d1*d1,d2*d2,2./3);
        for(int k=0;k<12;k++)if(std::abs(result[faceId(e,axis,k,t,a)]-
                f.cpu.p.c*eigenvalues[wave]*recon*eigen[k])>2e-12)
          throw std::runtime_error("single-characteristic upwind oracle failed");
      }
    }
    // Smooth finite-volume face reconstruction must retain third order.
    // Use exact sine cell averages, with the tested face at x=0.
    if(ns>=5)for(int axis=0;axis<2;axis++)for(int direction: {1,-1}) {
      double previous=0;
      for(double h: {.1,.05,.025}) {
        Vec sub(2*elements*ns*ns*12,0);
        for(int e=0;e<elements;e++)for(int j=0;j<ns;j++)for(int i=0;i<ns;i++) {
          const double x=((axis?j:i)-1.5)*h;
          const double value=std::sin(.7+x)*2*std::sin(h/2)/h;
          sub[f.cpu.id(e,j,i,0,ns)]=value/std::sqrt(6.0);
          sub[f.cpu.id(e,j,i,1+axis,ns)]=direction*value/std::sqrt(2.0);
          sub[f.cpu.id(e,j,i,4+axis,ns)]=value/std::sqrt(3.0);
        }
        halos(sub);
        const Vec result=run(sub,all,"smooth characteristic convergence");
        const double exact=direction*f.cpu.p.c*std::sin(.7)/std::sqrt(2.0);
        const double error=std::abs(result[faceId(0,axis,0,0,2)]-exact);
        if(previous && (previous/error<6.5 || previous/error>9.5))
          throw std::runtime_error("smooth WENO face reconstruction lost third order");
        previous=error;
      }
    }
    const Vec sub=make(1),before=run(sub,all,"alpha independence baseline");
    f.cpu.p.alpha*=2;
    const Vec after=run(sub,all,"alpha independence");
    if(before!=after)throw std::runtime_error("LF alpha changed characteristic flux");
    f.cpu.p.alpha/=2;
  }

  void tests() {
    for(int params=0;params<2;params++) {
      f.cpu.p.c=params?.7:1.3;f.cpu.p.alpha=params?2.4:1.75;f.cpu.p.wenoEps=params?1e-4:1e-12;
      run(make(0),all,"uniform",true);
      for(int pattern=1;pattern<=4;pattern++)run(make(pattern),all,"varying/jump/alternating/small state");
      const Vec sub=make(1);
      for(const auto& ids:std::vector<std::vector<int>>{{5,0,3},{2},{}})run(sub,ids,"selected faces");
      Vec positive=run(sub,all,"sign symmetry baseline"),negativeSub=sub;
      for(double& v:negativeSub)v=-v;
      Vec negative=run(negativeSub,all,"sign symmetry");
      for(size_t i=0;i<size_t(elements)*2*12*f.cpu.ns*(f.cpu.ns+1);i++)
        if(std::abs(positive[i]+negative[i])>2e-12)throw std::runtime_error("WENO sign symmetry failed");
      for(int component=0;component<12;component++) {
        Vec one(sub.size(),0);
        for(size_t node=0;node<sub.size()/12;node++)one[node*12+component]=sub[node*12+component];
        run(one,all,"individual moment");
      }
    }
    pipeline();
    if(f.cpu.p.characteristicWeno)characteristicTests();
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
    // Check the transform against the physical matrix and DG |A| flux,
    // independently of the WENO host/device comparison.
    for(int axis=0;axis<2;axis++)for(int sample=0;sample<16;sample++) {
      State l{},r{};
      for(int k=0;k<12;k++){l[k]=std::sin(.7*k+sample);r[k]=std::cos(.3*k-sample);}
      const State wl=hcz::characteristics(l,axis),wr=hcz::characteristics(r,axis);
      const State back=hcz::moments(wl,axis);
      const State af=hcz::characteristics(hcz::flux(l,axis,1.3),axis);
      State f{};const double speeds[6]={std::sqrt(3.0),-std::sqrt(3.0),1,-1,0,0};
      for(int k=0;k<12;k++) {
        const double speed=1.3*speeds[k%6];
        if(std::abs(back[k]-l[k])>2e-14 || std::abs(af[k]-speed*wl[k])>2e-14)
          throw std::runtime_error("characteristic diagonalization failed");
        f[k]=speed*(speed>0?wl[k]:wr[k]);
      }
      const State expected=hcz::upwind(l,r,axis,1.3),actual=hcz::moments(f,axis);
      for(int k=0;k<12;k++)if(std::abs(expected[k]-actual[k])>2e-14)
        throw std::runtime_error("characteristic flux differs from DG upwind");
    }
    for(bool characteristic: {true,false})
    for(int degree: {1,2,3,4})for(bool walls: {false,true}) {
      Vec r,w,qr,qw;quadrature(degree,r,w,qr,qw);
      if(degree==4){r={-1,-std::sqrt(3./7),0,std::sqrt(3./7),1};w={.1,49./90,32./45,49./90,.1};}
      Check check(device,root,degree,walls,r,w,qr,qw,characteristic);check.tests();
      std::cout<<mode<<(characteristic?" CHARACTERISTIC":" LF")<<" N="<<degree<<(walls?" walls":" periodic")<<": "<<check.cases
               <<" cases passed; max flux error="<<check.maxError<<", shared-face error="
               <<check.maxInterfaceError<<", balance="<<check.maxBalance<<"\n";
    }
  } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
  return 0;
}
