// Conservative residual reconstruction versus project(fv,ns,n,R,true).
#include "hczQuad2DKernelTest.hpp"
#include <cstring>
using namespace hczKernelTest;

namespace {
constexpr double sentinel=-987.25;
struct Check {
  occa::device& device;
  Fixture f;
  occa::kernel reconstruct,maskedReconstruct,project;
  std::vector<int> all={4,1,5,0,3,2};
  double maxError=0,maxMean=0,maxShift=0,maxRoundtrip=0;
  int cases=0;
  Check(occa::device& d,const std::string& root,int degree,
        const Vec& r,const Vec& w,const Vec& qr,const Vec& qw):
    device(d),f(degree,12,false,r,w,qr,qw) {
    auto props=kernelProperties(root,degree,12);
    reconstruct=device.buildKernel(root+"/solvers/bns/okl/bnsHczReconstructQuad2D.okl","bnsHczReconstructQuad2D",props);
    project=device.buildKernel(root+"/solvers/bns/okl/bnsHczSubcellQuad2D.okl","bnsHczProjectSubcellQuad2D",props);
    props["defines/p_HCZ_MASKED_RECONSTRUCT"]=1;
    maskedReconstruct=device.buildKernel(root+"/solvers/bns/okl/bnsHczReconstructQuad2D.okl","bnsHczReconstructQuad2D",props);
  }
  Vec native(const Vec& q,int n) const {
    Vec out(q.size());
    for(int e=0;e<elements;e++)for(int node=0;node<n*n;node++)for(int k=0;k<12;k++)
      out[(e*12+k)*n*n+node]=q[(e*n*n+node)*12+k];
    return out;
  }
  Vec run(const Vec& sub,const std::vector<int>& ids,const Vec& R,occa::memory* supplied=nullptr) {
    const int ns=f.cpu.ns,np=ns*ns,n=f.n;
    Vec input=native(sub,ns),mat(n*ns);
    for(int s=0;s<ns;s++)for(int i=0;i<n;i++)mat[i+s*n]=R[i*ns+s];
    const auto expected=native(f.cpu.project(sub,ns,n,R,true),n);
    const auto raw=native(f.cpu.project(sub,ns,n,R),n);
    auto o_ids=device.malloc<int>(std::max(size_t(1),ids.size()));
    if(!ids.empty())o_ids.copyFrom(ids.data(),ids.size()*sizeof(int));
    auto o_R=device.malloc<double>(mat.size(),mat.data());
    auto o_w=device.malloc<double>(f.cpu.weights.size(),f.cpu.weights.data());
    auto o_sub=supplied?*supplied:device.malloc<double>(input.size(),input.data());
    Vec before(input.size());o_sub.copyTo(before.data());
    Vec seed((elements+2)*n*n*12,sentinel),actual(seed.size());
    auto o_dg=device.malloc<double>(seed.size(),seed.data());
    for(int repeat=0;repeat<2;repeat++) {
      reconstruct(int(ids.size()),o_ids,o_R,o_w,o_sub,o_dg);o_dg.copyTo(actual.data());
      for(int e=0;e<elements+2;e++)for(int k=0;k<12;k++) {
        const bool selected=std::find(ids.begin(),ids.end(),e)!=ids.end();
        double avg=0,dgMean=0,scale=1;
        if(selected)for(int node=0;node<np;node++) {
          avg+=input[(e*12+k)*np+node]/np;
          scale=std::max(scale,std::abs(input[(e*12+k)*np+node]));
        }
        for(int j=0;j<n;j++)for(int i=0;i<n;i++) {
          const size_t pos=(e*12+k)*n*n+j*n+i;
          const double want=selected?expected[pos]:sentinel;
          const double error=std::abs(actual[pos]-want);maxError=std::max(maxError,error);
          if(!std::isfinite(actual[pos]) || error>3e-12*scale)
            throw std::runtime_error("reconstructed residual differs from host");
          if(!selected && actual[pos]!=sentinel)throw std::runtime_error("reconstruction overwrote unselected/halo entry");
          if(selected) {
            dgMean+=.25*f.cpu.weights[j]*f.cpu.weights[i]*actual[pos];
            const double shift=expected[pos]-raw[pos];maxShift=std::max(maxShift,std::abs(shift));
            if(std::abs((actual[pos]-raw[pos])-shift)>3e-12*scale)
              throw std::runtime_error("mean correction mismatch");
          }
        }
        if(selected) {
          const double error=std::abs(avg-dgMean);maxMean=std::max(maxMean,error);
          if(error>3e-12*scale)throw std::runtime_error("reconstruction lost a subcell moment mean");
        }
      }
    }
    Vec after(before.size());o_sub.copyTo(after.data());
    if(std::memcmp(after.data(),before.data(),after.size()*sizeof(double)))throw std::runtime_error("reconstruction changed subcell residual");
    after.resize(mat.size());o_R.copyTo(after.data());if(after!=mat)throw std::runtime_error("reconstruction changed R");
    after.resize(f.cpu.weights.size());o_w.copyTo(after.data());if(after!=f.cpu.weights)throw std::runtime_error("reconstruction changed weights");
    // Device stage mode uses the same element list with a mask instead of
    // uploading a compacted marked-element list. Check exact preservation of
    // all inactive outputs and halos for all/none/alternating masks.
    for(int pattern=0;pattern<3;pattern++) {
      std::vector<int> mask(elements+2,0);
      for(int e=0;e<elements;e++)mask[e]=pattern==1 || (pattern==2 && e%2);
      auto o_mask=device.malloc<int>(mask.size(),mask.data());
      o_dg.copyFrom(seed.data());
      maskedReconstruct(int(ids.size()),o_ids,o_R,o_w,o_sub,o_dg,o_mask);
      Vec result(seed.size());o_dg.copyTo(result.data());
      for(int e=0;e<elements+2;e++)for(int k=0;k<12*n*n;k++) {
        const size_t pos=e*12*n*n+k;
        const bool selected=mask[e] && std::find(ids.begin(),ids.end(),e)!=ids.end();
        // The masked path must be bitwise equal to the unmasked result on
        // selected elements, and retain the seeded output everywhere else.
        if(result[pos]!=(selected?actual[pos]:sentinel))
          throw std::runtime_error("masked reconstruction selection mismatch");
      }
      std::vector<int> afterMask(mask.size());o_mask.copyTo(afterMask.data());
      if(afterMask!=mask)throw std::runtime_error("masked reconstruction changed mask");
    }
    cases++;return actual;
  }
  Vec data(int pattern) const {
    const int ns=f.cpu.ns;Vec sub(elements*ns*ns*12,0);
    for(int e=0;e<elements;e++)for(int j=0;j<ns;j++)for(int i=0;i<ns;i++)for(int k=0;k<12;k++) {
      double value=0;
      if(pattern==1)value=.13*(k-5);
      if(pattern==2)value=.27*std::sin(.7*i+.4*j+1.1*e+.3*k);
      if(pattern==3)value=((i+j+k+e)%2?1:-1)*(.3+.02*k)+(i<ns/2?.5:-.1);
      if(pattern==4)value=1000*std::cos(.3*(i+2*j+e+k));
      if(pattern==5)value=1e-8*std::sin(.3*(i+2*j+e+k));
      sub[f.cpu.id(e,j,i,k,ns)]=value;
    }
    return sub;
  }
  void roundtrip() {
    const int n=f.n,ns=f.cpu.ns;Vec q(elements*n*n*12);
    for(int e=0;e<elements;e++)for(int j=0;j<n;j++)for(int i=0;i<n;i++)for(int k=0;k<12;k++) {
      const double x=f.cpu.nodes[i],y=f.cpu.nodes[j];
      q[f.cpu.id(e,j,i,k,n)]=.1*(k-5)+.03*(e+1)*x+.017*x*y
        +(n>=3?.02*x*x-.01*y*y:0)+(n>=4?.003*x*x*x*y:0)+(n>=5?.007*x*x*x*x:0);
    }
    Vec input=native(q,n),A(n*ns);
    for(int i=0;i<n;i++)for(int s=0;s<ns;s++)A[s+i*ns]=f.cpu.A[s*n+i];
    auto o_q=device.malloc<double>(input.size(),input.data()),o_A=device.malloc<double>(A.size(),A.data());
    auto o_ids=device.malloc<int>(all.size(),all.data()),o_sub=device.malloc<double>(elements*ns*ns*12);
    project(elements,o_ids,o_A,o_q,o_sub);
    auto expectedSub=f.cpu.project(q,n,ns,f.cpu.A);
    auto actual=run(expectedSub,all,f.cpu.R,&o_sub);
    for(size_t i=0;i<input.size();i++) {
      double error=std::abs(actual[i]-input[i]);maxRoundtrip=std::max(maxRoundtrip,error);
      if(error>3e-12)throw std::runtime_error("DG polynomial projection/reconstruction round trip failed");
    }
  }
  void tests() {
    for(int pattern=0;pattern<=5;pattern++)run(data(pattern),all,f.cpu.R);
    for(const auto& ids:std::vector<std::vector<int>>{{5,0,3},{2},{}})run(data(3),ids,f.cpu.R);
    const int ns=f.cpu.ns;
    for(int k=0;k<12;k++) {
      Vec sub(elements*ns*ns*12,0);sub[f.cpu.id(4,0,ns-1,k,ns)]=-1;
      run(sub,all,f.cpu.R);
    }
    // The exact pseudoinverse almost preserves the mean already. Perturb a
    // test matrix to make omission of the explicit correction observable.
    Vec biased=f.cpu.R;biased[0]+=.07;
    maxShift=0;run(data(2),all,biased);
    if(maxShift<1e-5)throw std::runtime_error("test matrix did not exercise mean correction");
    roundtrip();
  }
};

void maskControlTests(occa::device& device,const std::string& root) {
  auto props=kernelProperties(root,1,12);
  const auto file=root+"/solvers/bns/okl/bnsHczMaskControlQuad2D.okl";
  auto difference=device.buildKernel(file,"bnsHczMaskDifferenceQuad2D",props);
  auto merge=device.buildKernel(file,"bnsHczMaskUnionQuad2D",props);
  auto count=device.buildKernel(file,"bnsHczMaskCountQuad2D",props);
  auto clear=device.buildKernel(file,"bnsHczMaskClearQuad2D",props);
  auto o_result=device.malloc<int>(1);
  for(int n: {0,1,255,256,257,1031})for(int pattern=0;pattern<4;pattern++) {
    std::vector<int> current(n+5,7),next(n+5,9);
    for(int e=0;e<n;e++) {
      current[e]=pattern==1 || (pattern>=2 && e%3==0);
      next[e]=pattern==1 || (pattern>=2 && e%5==0);
      if(pattern==3)next[e]=!current[e];
    }
    auto o_current=device.malloc<int>(current.size(),current.data());
    auto o_next=device.malloc<int>(next.size(),next.data());
    if(n)difference(n,o_current,o_next);
    auto want=next;
    for(int e=0;e<n;e++)want[e]=next[e] && !current[e];
    std::vector<int> result(want.size());o_next.copyTo(result.data());
    if(result!=want)throw std::runtime_error("newly troubled mask or halo mismatch");
    count(n,o_next,o_result);int actualCount=-1;o_result.copyTo(&actualCount);
    if(actualCount!=std::count(want.begin(),want.begin()+n,1))
      throw std::runtime_error("newly troubled reduction mismatch");
    if(n)merge(n,o_next,o_current);
    for(int e=0;e<n;e++)current[e]=current[e] || want[e];
    o_current.copyTo(result.data());
    if(result!=current)throw std::runtime_error("merged mask or halo mismatch");
    count(n,o_current,o_result);o_result.copyTo(&actualCount);
    if(actualCount!=std::count(current.begin(),current.begin()+n,1))
      throw std::runtime_error("merged mask count mismatch");
    o_next.copyTo(result.data());
    if(result!=want)throw std::runtime_error("merge changed input mask");
    if(n)clear(n,o_current);
    o_current.copyTo(result.data());
    for(int e=0;e<n;e++)current[e]=0;
    if(result!=current)throw std::runtime_error("mask reset changed halo or missed owned entries");
  }
  std::cout<<"Mask control: 24 cases passed (including empty and strided reductions)\n";
}
}
int main(int argc,char** argv) {
  if(argc<2 || argc>3)return 2;
  const std::string root=argv[1],mode=argc==3?argv[2]:"Serial";
  try {
    occa::device device;occa::json config;config["mode"]=mode;if(mode=="CUDA")config["device_id"]=0;
    device.setup(config);
    maskControlTests(device,root);
    for(int degree: {1,2,3,4}) {
      Vec r,w,qr,qw;quadrature(degree,r,w,qr,qw);
      if(degree==4){r={-1,-std::sqrt(3./7),0,std::sqrt(3./7),1};w={.1,49./90,32./45,49./90,.1};}
      Check check(device,root,degree,r,w,qr,qw);check.tests();
      std::cout<<mode<<" N="<<degree<<": "<<check.cases<<" cases passed; residual="<<check.maxError
               <<", mean="<<check.maxMean<<", roundtrip="<<check.maxRoundtrip<<"\n";
    }
  } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
  return 0;
}
