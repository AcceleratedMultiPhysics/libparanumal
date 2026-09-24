// Device input/candidate decisions versus the HCZ host criteria.
#include "hczQuad2DKernelTest.hpp"
#include <cstring>
using namespace hczKernelTest;

namespace {
occa::json checkProperties(const std::string& root,int degree) {
  auto props=kernelProperties(root,degree,12);
  props["defines/p_cubNq"]=degree+2;
  return props;
}

void compare(occa::device& device,occa::kernel& kernel,occa::kernel& reduce,
             Fixture& f,const std::vector<int>& ids,const Vec& q,const Vec& residual,
             const Vec& rhs,bool candidate,double dt,double a,double b,
             int required,const char* label) {
  Vec trial(elements*f.np*12);
  for(int e=0;e<elements;e++) for(int node=0;node<f.np;node++) for(int k=0;k<12;k++) {
    size_t native=(e*12+k)*f.np+node;
    double value=q[native];
    if(candidate)value+=b*(dt*rhs[native]+(a==0?0:a*residual[native]));
    trial[(e*f.np+node)*12+k]=value;
  }
  Vec cub=f.cpu.project(trial,f.n,f.cpu.nc,f.cpu.I);
  std::vector<int> expected(ids.size()+1,91);
  int combined=0;
  for(size_t t=0;t<ids.size();t++) {
    int e=ids[t],bad=0;
    for(int j=0;j<f.n;j++) for(int i=0;i<f.n;i++) {
      State state=f.cpu.get(trial,e,j,i,f.n);
      for(double value:state)if(!std::isfinite(value))bad|=1;
      if(!f.cpu.p.physical(state[0]))bad|=1;
    }
    for(int j=0;j<f.cpu.nc;j++) for(int i=0;i<f.cpu.nc;i++) {
      State state=f.cpu.get(cub,e,j,i,f.cpu.nc);
      for(double value:state)if(!std::isfinite(value))bad|=2;
      if(!(f.cpu.p.rho(state[0])>0))bad|=2;
    }
    expected[t]=bad;combined|=bad;
  }
  if(required>=0 && combined!=required)throw std::runtime_error(std::string(label)+": reference fixture has wrong flags");
  // Also compare the public host decision, replacing unselected elements by
  // a valid uniform state so the host's all-element scan has the same scope.
  Vec selectedTrial=trial;
  for(int e=0;e<elements;e++)if(std::find(ids.begin(),ids.end(),e)==ids.end())
    for(int node=0;node<f.np;node++)for(int k=0;k<12;k++)selectedTrial[(e*f.np+node)*12+k]=k==0?.2:0;
  if(f.cpu.admissible(selectedTrial)!=(combined==0))throw std::runtime_error("host decision disagrees with expected flags");
  Vec interp(f.n*f.cpu.nc);
  for(int i=0;i<f.n;i++)for(int j=0;j<f.cpu.nc;j++)interp[i*f.cpu.nc+j]=f.cpu.I[j*f.n+i];
  auto o_ids=device.malloc<int>(std::max(size_t(1),ids.size()));
  if(!ids.empty())o_ids.copyFrom(ids.data(),ids.size()*sizeof(int));
  auto o_I=device.malloc<double>(interp.size(),interp.data());
  auto o_q=device.malloc<double>(q.size(),q.data());
  auto o_res=device.malloc<double>(residual.size(),residual.data());
  auto o_rhs=device.malloc<double>(rhs.size(),rhs.data());
  std::vector<int> flags(expected.size(),91);
  auto o_flags=device.malloc<int>(flags.size(),flags.data());
  auto o_result=device.malloc<int>(1);
  const auto& p=f.cpu.p;
  for(int repeat=0;repeat<2;repeat++) {
    kernel(int(ids.size()),o_ids,o_I,int(candidate),dt,a,b,p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.floor(),o_q,o_res,o_rhs,o_flags);
    reduce(int(ids.size()),o_flags,o_result);
    int result=91;o_result.copyTo(&result);o_flags.copyTo(flags.data());
    if(result!=combined || flags!=expected) {
      std::cerr<<label<<": result="<<result<<", expected="<<combined<<"\n";
      throw std::runtime_error("device admissibility differs from host");
    }
  }
  for(const auto& pair: {std::make_pair(o_q,&q),std::make_pair(o_res,&residual),std::make_pair(o_rhs,&rhs)}) {
    Vec after(pair.second->size());pair.first.copyTo(after.data());
    if(std::memcmp(after.data(),pair.second->data(),after.size()*sizeof(double)))
      throw std::runtime_error("check modified state, residual or RHS");
  }
}

void reduction(occa::device& device,occa::kernel& kernel) {
  for(int count: {0,1,255,256,257,1031})for(bool invalid: {false,true}) {
    std::vector<int> flags(std::max(count,1),0);
    if(invalid && count) {flags[0]=1;flags[count-1]|=2;}
    auto o_flags=device.malloc<int>(flags.size(),flags.data());auto o_result=device.malloc<int>(1);
    kernel(count,o_flags,o_result);int result=-1;o_result.copyTo(&result);
    if(result!=(invalid && count?3:0))throw std::runtime_error("flag reduction failed");
  }
}
} // namespace

int main(int argc,char** argv) {
  if(argc<2 || argc>3)return 2;
  const std::string root=argv[1],mode=argc==3?argv[2]:"Serial";
  try {
    occa::device device;occa::json config;config["mode"]=mode;
    if(mode=="CUDA")config["device_id"]=0;
    device.setup(config);
    for(int degree: {1,2,3}) {
      Vec r,w,qr,qw;quadrature(degree,r,w,qr,qw);
      Fixture f(degree,12,false,r,w,qr,qw);
      const std::string file=root+"/solvers/bns/okl/bnsHczAdmissibilityQuad2D.okl";
      auto props=checkProperties(root,degree);
      auto kernel=device.buildKernel(file,"bnsHczAdmissibilityQuad2D",props);
      auto reduce=device.buildKernel(file,"bnsHczAdmissibilityReduce",props);
      reduction(device,reduce);
      Vec base(elements*12*f.np,0);
      for(int e=0;e<elements;e++)for(int node=0;node<f.np;node++)base[e*12*f.np+node]=.2;
      const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
      const std::vector<int> all={4,1,5,0,3,2};
      Vec zero(base.size(),0),poison(base.size(),nan);
      for(const auto& ids: std::vector<std::vector<int>>{all,{5,0,3},{2},{}}) {
        compare(device,kernel,reduce,f,ids,base,poison,poison,false,1,1,1,0,"valid input ignores RHS/residual");
        compare(device,kernel,reduce,f,ids,base,poison,zero,true,.1,0,.2,0,"first stage ignores residual");
        Vec invalid=base;invalid[4*12*f.np]=nan;
        compare(device,kernel,reduce,f,ids,invalid,zero,zero,false,0,0,0,-1,"unselected/selected invalid element");
      }
      for(int k: {0,6,11})for(double bad: {nan,inf,-inf}) {
        Vec q=base;q[(4*12+k)*f.np]=bad;
        compare(device,kernel,reduce,f,all,q,zero,zero,false,0,0,0,-1,"nonfinite nodal moment");
      }
      for(double phi: {f.cpu.p.floor(),std::nextafter(f.cpu.p.floor(),inf),3.999/4,std::nextafter(3.999/4,0.0)}) {
        Vec q=base;
        for(int node=0;node<f.np;node++)q[4*12*f.np+node]=phi;
        bool valid=f.cpu.p.physical(phi);
        compare(device,kernel,reduce,f,all,q,zero,zero,false,0,0,0,valid?0:-1,"strict nodal threshold");
      }
      Vec rhs=zero,res=zero;
      for(int node=0;node<f.np;node++)rhs[4*12*f.np+node]=-4;
      compare(device,kernel,reduce,f,all,base,poison,rhs,true,.5,0,.5,-1,"invalid first-stage candidate");
      for(int node=0;node<f.np;node++)res[4*12*f.np+node]=4;
      compare(device,kernel,reduce,f,all,base,res,zero,true,.1,-.5,.5,-1,"residual invalidates candidate");
      for(int node=0;node<f.np;node++)res[4*12*f.np+node]=-4;
      compare(device,kernel,reduce,f,all,base,res,rhs,true,.5,-.5,.5,0,"RHS and residual cancel in a valid candidate");
      compare(device,kernel,reduce,f,all,base,poison,zero,true,.1,-.5,.5,3,"later residual NaN is rejected");
      rhs=zero;rhs[4*12*f.np+11*f.np]=inf;
      compare(device,kernel,reduce,f,all,base,zero,rhs,true,.1,0,.5,-1,"nonfinite candidate moment");
      if(degree>=2) {
        int column=-1;
        for(int j=0;j<f.cpu.nc;j++)for(int i=0;i<f.n;i++)if(f.cpu.I[j*f.n+i]<-.02)column=i;
        if(column<0)throw std::runtime_error("missing overshoot fixture");
        auto saved=f.cpu.p;
        // Positive nodal densities, but negative mapped density at cubature.
        f.cpu.p.phiL=.2;f.cpu.p.phiH=.4;f.cpu.p.rhoL=1;f.cpu.p.rhoH=3;
        Vec q=base;
        for(int j=0;j<f.n;j++)for(int i=0;i<f.n;i++)q[4*12*f.np+j*f.n+i]=.11+(i==column?.8:0);
        compare(device,kernel,reduce,f,all,q,zero,zero,false,0,0,0,2,"cubature-only density failure");
        rhs=zero;
        for(size_t i=0;i<q.size();i++)rhs[i]=(q[i]-base[i])/.25;
        compare(device,kernel,reduce,f,all,base,zero,rhs,true,.5,0,.5,2,"cubature-only candidate failure");
        f.cpu.p=saved;f.cpu.p.rhoH=1.2;
        for(int j=0;j<f.n;j++)for(int i=0;i<f.n;i++)q[4*12*f.np+j*f.n+i]=.01+(i==column?.8:0);
        compare(device,kernel,reduce,f,all,q,zero,zero,false,0,0,0,0,"negative cubature index with positive density");
        for(int j=0;j<f.n;j++)for(int i=0;i<f.n;i++)q[4*12*f.np+j*f.n+i]=.98-(i==column?.8:0);
        compare(device,kernel,reduce,f,all,q,zero,zero,false,0,0,0,0,"cubature has no EOS bound");
        f.cpu.p=saved;
        // All nodal moments are finite, but a high-order interpolant can
        // overflow. Cubature must check every moment, not only the index.
        q=base;int row=0;
        while(f.cpu.I[row*f.n+column]>=0)row++;
        for(int j=0;j<f.n;j++)for(int i=0;i<f.n;i++) {
          double weight=f.cpu.I[row*f.n+j]*f.cpu.I[row*f.n+i];
          q[(4*12+11)*f.np+j*f.n+i]=std::copysign(.9*std::numeric_limits<double>::max(),weight);
        }
        compare(device,kernel,reduce,f,all,q,zero,zero,false,0,0,0,2,"cubature-only nonfinite stress");
      }
      std::cout<<mode<<" N="<<degree<<": input/candidate flags agree with host; strict bounds, nonfinite states, cubature-only failures, first-stage residual, subsets, empty lists, reductions and input preservation passed\n";
    }
  } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
  return 0;
}
