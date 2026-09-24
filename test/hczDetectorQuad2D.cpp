// Compare raw OCCA detector masks with QuadOperator::detect, independently
// of buffering and subcell recomputation. Halo values are supplied explicitly.
#include "hczQuad2DKernelTest.hpp"
#include <cstring>
using namespace hczKernelTest;

namespace {
constexpr int side=4, owned=side*side, total=2*owned;
constexpr int sentinel=73;
const double nan=std::numeric_limits<double>::quiet_NaN();
const double inf=std::numeric_limits<double>::infinity();

Vec native(const Vec& host,int np) {
  Vec out(host.size());
  for(int e=0;e<total;e++)for(int node=0;node<np;node++)for(int k=0;k<12;k++)
    out[(e*12+k)*np+node]=host[(e*np+node)*12+k];
  return out;
}

struct Check {
  occa::device& device;
  occa::kernel boundsKernel,detector,stateDetector,interfaceKernel;
  hcz::QuadOperator cpu;
  std::vector<int> all;
  int comparisons=0;

  Check(occa::device& d,const std::string& root,int degree,bool walls,
        const Vec& r,const Vec& w,const Vec& qr,const Vec& qw):
    device(d),cpu(degree,owned,total,hx,hy,r,w,qr,qw),all(owned) {
    std::iota(all.begin(),all.end(),0);
    std::reverse(all.begin(),all.end());
    for(int e=0;e<owned;e++)for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
      int x=(e%side+dx+side)%side,y=e/side+dy,k=(dy+1)*3+dx+1;
      if(!walls)y=(y+side)%side;
      int nb=y<0 || y>=side?-1:x+side*y;
      if(nb>=0 && k!=4 && (e+k)%2)nb+=owned;
      cpu.neighbor[e][k]=nb;
    }
    cpu.exchange=[](Vec& q,int stride) {
      std::copy_n(q.begin(),owned*stride,q.begin()+owned*stride);
    };
    auto props=kernelProperties(root,degree,12);
    props["defines/p_cubNq"]=degree+2;
    const auto file=root+"/solvers/bns/okl/bnsHczDetectorQuad2D.okl";
    boundsKernel=device.buildKernel(file,"bnsHczIndexBoundsQuad2D",props);
    interfaceKernel=device.buildKernel(file,"bnsHczInterfaceMaskQuad2D",props);
    detector=device.buildKernel(file,"bnsHczDetectorQuad2D",props);
    props["defines/p_HCZ_DETECT_STATE"]=1;
    stateDetector=device.buildKernel(file,"bnsHczDetectorQuad2D",props);
  }

  Vec state(double phi=.15) const {
    Vec q(total*cpu.n*cpu.n*12,0);
    for(size_t i=0;i<q.size();i+=12)q[i]=phi;
    return q;
  }
  void phi(Vec& q,int e,double value) const {
    for(int node=0;node<cpu.n*cpu.n;node++)q[(e*cpu.n*cpu.n+node)*12]=value;
  }

  void interfaceTests() {
    const auto defaults=cpu.p;
    cpu.p.interfaceSubcells=true;
    const int np=cpu.n*cpu.n;
    for(double threshold: {.01,.1,.49})for(int shift: {0,7}) {
      cpu.p.interfaceThreshold=threshold;
      const double low=cpu.p.phiL+threshold*(cpu.p.phiH-cpu.p.phiL);
      const double high=cpu.p.phiH-threshold*(cpu.p.phiH-cpu.p.phiL);
      Vec q=state(cpu.p.phiL);
      std::vector<int> expected(owned,0);
      auto cell=[&](int k){return (k+shift)%owned;};
      phi(q,cell(0),cpu.p.phiH); // pure heavy
      phi(q,cell(1),low);phi(q,cell(2),high); // exact open-band edges
      phi(q,cell(3),std::nextafter(low,high));expected[cell(3)]=1;
      phi(q,cell(4),std::nextafter(high,low));expected[cell(4)]=1;
      // Neither endpoint is in the band, but the range crosses the interface.
      q[cell(5)*np*12]=cpu.p.phiH;expected[cell(5)]=1;
      phi(q,cell(6),.5*(cpu.p.phiL+cpu.p.phiH));expected[cell(6)]=1;
      const Vec host=cpu.interfaceMask(q);
      for(int e=0;e<owned;e++)if(host[e]!=expected[e])throw std::runtime_error("wrong host interface seed");
      auto values=native(q,np);
      auto o_q=device.malloc<double>(values.size(),values.data());
      auto o_bounds=device.malloc<double>(2*total);
      auto o_ids=device.malloc<int>(all.size(),all.data());
      auto o_mask=device.malloc<int>(total);
      boundsKernel(owned,o_ids,o_q,o_bounds);
      // Reuse poisoned output, then clear/repopulate it with a shifted seed.
      // Include subset and empty element lists; halos must remain untouched.
      for(int count: {owned,3,0}) {
        std::vector<int> actual(total,sentinel),want=actual;
        for(int i=0;i<count;i++)want[all[i]]=expected[all[i]];
        o_mask.copyFrom(actual.data());
        interfaceKernel(count,o_ids,o_bounds,low,high,o_mask);
        o_mask.copyTo(actual.data());
        if(actual!=want)throw std::runtime_error("interface kernel mismatch, stale flag, or modified halo");
      }
    }
    cpu.p=defaults;
    const auto disabled=cpu.interfaceMask(state());
    if(std::any_of(disabled.begin(),disabled.end(),[](double v){return v!=0;}))
      throw std::runtime_error("disabled interface policy marked cells");
  }

  // required applies to the full host mask; -1 means compare without a
  // prescribed count. Device calls additionally check permuted/subset lists.
  void run(const char* label,Vec reference,const Vec& q,const Vec& residual,
           const Vec& rhs,double dt,double a,double b,int required=-1,
           const std::vector<int>* selected=nullptr) {
    const auto& ids=selected?*selected:all;
    const int np=cpu.n*cpu.n;
    Vec candidate=q;
    for(int i=0;i<owned*np*12;i++)candidate[i]+=b*(dt*rhs[i]+(a==0?0:a*residual[i]));
    const Vec expected=cpu.detect(reference,candidate);
    int count=std::accumulate(expected.begin(),expected.begin()+owned,0.0);
    if(required>=0 && count!=required)
      throw std::runtime_error(std::string(label)+": unexpected host mask count "+std::to_string(count));
    cpu.exchange(reference,np*12);
    Vec ref=native(reference,np),base=native(q,np),res=native(residual,np),force=native(rhs,np);
    Vec interp(cpu.n*cpu.nc),expectedBounds(total*2);
    for(int i=0;i<cpu.n;i++)for(int j=0;j<cpu.nc;j++)interp[i*cpu.nc+j]=cpu.I[j*cpu.n+i];
    for(int e=0;e<total;e++) {
      double lo=inf,hi=-inf;
      for(int node=0;node<np;node++) {
        const double value=reference[(e*np+node)*12];
        lo=std::min(lo,value);hi=std::max(hi,value);
      }
      expectedBounds[2*e]=lo;expectedBounds[2*e+1]=hi;
    }
    std::vector<int> nb(owned*9),boundIds(total);
    for(int e=0;e<owned;e++)std::copy(cpu.neighbor[e].begin(),cpu.neighbor[e].end(),nb.begin()+9*e);
    std::iota(boundIds.begin(),boundIds.end(),0);std::reverse(boundIds.begin(),boundIds.end());
    auto o_nb=device.malloc<int>(nb.size(),nb.data());
    auto o_ids=device.malloc<int>(std::max(size_t(1),ids.size()));
    if(!ids.empty())o_ids.copyFrom(ids.data(),ids.size()*sizeof(int));
    auto o_boundIds=device.malloc<int>(total,boundIds.data());
    auto o_ref=device.malloc<double>(ref.size(),ref.data());
    auto o_q=device.malloc<double>(base.size(),base.data());
    auto o_res=device.malloc<double>(res.size(),res.data());
    auto o_rhs=device.malloc<double>(force.size(),force.data());
    Vec trial=native(candidate,np);
    auto o_trial=device.malloc<double>(trial.size(),trial.data());
    auto o_I=device.malloc<double>(interp.size(),interp.data());
    Vec bounds(total*2+2,sentinel);
    auto o_bounds=device.malloc<double>(bounds.size(),bounds.data());
    std::vector<int> mask(total,sentinel),want=mask;
    for(int e:ids)want[e]=int(expected[e]);
    auto o_mask=device.malloc<int>(mask.size(),mask.data());
    const auto& p=cpu.p;
    for(int repeat=0;repeat<2;repeat++) {
      // Prepare supplied halo bounds with the same kernel as owned bounds.
      // Actual MPI exchange is intentionally outside this isolated increment.
      boundsKernel(total,o_boundIds,o_ref,o_bounds);
      o_bounds.copyTo(bounds.data());
      for(int e=0;e<total*2;e++)if(bounds[e]!=expectedBounds[e])
        throw std::runtime_error(std::string(label)+": index bounds mismatch");
      if(bounds[total*2]!=sentinel || bounds[total*2+1]!=sentinel)
        throw std::runtime_error("bounds kernel overwrote sentinel");
      detector(int(ids.size()),o_ids,o_nb,o_bounds,o_I,dt,a,b,
               p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.floor(),p.dmpDelta,p.dmpEps,p.boundTol,
               o_q,o_res,o_rhs,o_mask);
      o_mask.copyTo(mask.data());
      if(mask!=want) {
        for(int e:ids)if(mask[e]!=want[e])std::cerr<<label<<": element "<<e<<", device="<<mask[e]<<", host="<<want[e]<<"\n";
        throw std::runtime_error("raw detector mask differs from host");
      }
      std::fill(mask.begin(),mask.end(),sentinel);o_mask.copyFrom(mask.data());
      // SARK4 supplies an already assembled candidate. Poison the RK
      // coefficients to ensure this specialization ignores them completely.
      stateDetector(int(ids.size()),o_ids,o_nb,o_bounds,o_I,nan,nan,nan,
                    p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.floor(),p.dmpDelta,p.dmpEps,p.boundTol,
                    o_q,o_res,o_trial,o_mask);
      o_mask.copyTo(mask.data());
      if(mask!=want)throw std::runtime_error("assembled-candidate detector differs from host");
    }
    for(const auto& pair: {std::make_pair(o_ref,&ref),std::make_pair(o_q,&base),
                           std::make_pair(o_res,&res),std::make_pair(o_rhs,&force),
                           std::make_pair(o_trial,&trial),std::make_pair(o_I,&interp),std::make_pair(o_bounds,&bounds)}) {
      Vec after(pair.second->size());pair.first.copyTo(after.data());
      if(std::memcmp(after.data(),pair.second->data(),after.size()*sizeof(double)))
        throw std::runtime_error("detector modified a read-only buffer");
    }
    std::vector<int> after(nb.size());o_nb.copyTo(after.data());
    if(after!=nb)throw std::runtime_error("detector modified neighbor map");
    // Bounds preparation itself must honor subsets and empty lists.
    std::fill(bounds.begin(),bounds.end(),sentinel);o_bounds.copyFrom(bounds.data());
    boundsKernel(int(ids.size()),o_ids,o_ref,o_bounds);o_bounds.copyTo(bounds.data());
    for(int e=0;e<total;e++)for(int k=0;k<2;k++) {
      const double wantBound=std::find(ids.begin(),ids.end(),e)==ids.end()?sentinel:expectedBounds[2*e+k];
      if(bounds[2*e+k]!=wantBound)throw std::runtime_error("bounds subset mismatch");
    }
    comparisons++;
  }

  void direct(const char* label,const Vec& reference,const Vec& candidate,int count=-1) {
    Vec zero(candidate.size(),0),poison(candidate.size(),nan);
    run(label,reference,candidate,poison,zero,1,0,1,count);
  }

  void tests() {
    const auto defaults=cpu.p;
    Vec base=state(),zero(base.size(),0),poison(base.size(),nan);
    direct("uniform",base,base,0);
    Vec trial=base;phi(trial,5,.18);
    direct("absolute DMP violation",base,trial,1);
    const std::vector<std::vector<int>> lists={{5,0,15},{1},{}};
    for(const auto& ids:lists)run("selected mask",base,trial,poison,zero,1,0,1,1,&ids);

    // Equality is accepted, one representable value beyond is rejected.
    for(bool upper: {false,true}) {
      const double edge=.15+(upper?1:-1)*cpu.p.dmpDelta*(cpu.p.phiH-cpu.p.phiL);
      trial=base;phi(trial,5,edge);direct("DMP equality",base,trial,0);
      phi(trial,5,std::nextafter(edge,upper?inf:-inf));direct("DMP strict excursion",base,trial,1);
    }
    Vec reference=base;
    // A diagonal halo extends the envelope; a distant element does not.
    phi(reference,10,.22);trial=base;phi(trial,5,.21);
    direct("diagonal neighbor envelope",reference,trial,0);
    reference=base;phi(reference,15,.22);
    direct("outside neighborhood",reference,trial,1);
    // Same test at the periodic x boundary and, when applicable, y boundary.
    reference=base;phi(reference,3,.22);trial=base;phi(trial,0,.21);
    direct("periodic x neighbor",reference,trial,0);
    reference=base;phi(reference,12,.22);
    direct("y wall or periodic neighbor",reference,trial,cpu.neighbor[0][1]<0?1:0);

    reference=base;phi(reference,10,.22);trial=base;
    double edge=.22+cpu.p.dmpEps*(.22-.15);
    phi(trial,5,edge);direct("relative DMP equality",reference,trial,0);
    phi(trial,5,edge+1e-10);direct("relative DMP violation",reference,trial,1);

    cpu.p.dmpDelta=10; // Isolate phase and physical criteria from DMP.
    for(bool upper: {false,true}) {
      double edge=(upper?cpu.p.phiH:cpu.p.phiL)+(upper?1:-1)*cpu.p.boundTol*(cpu.p.phiH-cpu.p.phiL);
      trial=base;phi(trial,5,edge);direct("phase bound equality",base,trial,0);
      phi(trial,5,std::nextafter(edge,upper?inf:-inf));direct("phase bound violation",base,trial,1);
    }
    cpu.p.boundTol=10;
    for(double value: {cpu.p.floor(),std::nextafter(cpu.p.floor(),inf),3.999/4,std::nextafter(3.999/4,0.0)}) {
      trial=base;phi(trial,5,value);
      direct("physical nodal threshold",base,trial,cpu.p.physical(value)?0:1);
    }
    for(int k=0;k<12;k++)for(double bad: {nan,inf,-inf}) {
      trial=base;trial[(5*cpu.n*cpu.n)*12+k]=bad;
      direct("nonfinite candidate moment",base,trial,1);
    }
    cpu.p=defaults;
    Vec rhs=zero,res=zero;phi(rhs,5,.16);phi(res,5,.08);
    run("later-stage cancellation",base,base,res,rhs,.25,-.5,.5,0);
    run("first-stage residual ignored",base,base,poison,zero,.1,0,.2,0);
    res=zero;res[(5*cpu.n*cpu.n)*12+11]=nan;
    run("later-stage poisoned residual",base,base,res,zero,.1,-.5,.2,1);
    run("RHS triggers rejection",base,base,poison,rhs,.25,0,.5,1);

    // Spatially varying reference and nonzero histories exercise complete
    // candidate formation rather than only synthetic zero-increment inputs.
    for(int pattern=0;pattern<4;pattern++) {
      reference=base;res=zero;rhs=zero;
      for(int e=0;e<owned;e++)for(int node=0;node<cpu.n*cpu.n;node++)for(int k=0;k<12;k++) {
        int i=(e*cpu.n*cpu.n+node)*12+k;
        if(k==0)reference[i]+=.035*std::sin(.2*i);
        else reference[i]=.01*std::cos(.17*i);
        res[i]=.02*std::sin(.37*i);rhs[i]=(pattern+.1)*std::cos(.13*i);
      }
      run("varying stage candidate",reference,reference,res,rhs,.08,-.4,.6);
    }

    if(cpu.n>=3) {
      int column=-1,row=-1;
      for(int j=0;j<cpu.nc;j++)for(int i=0;i<cpu.n;i++)if(cpu.I[j*cpu.n+i]<-.02){row=j;column=i;}
      if(column<0)throw std::runtime_error("missing cubature overshoot fixture");
      cpu.p.dmpDelta=10;cpu.p.boundTol=10;
      cpu.p.phiL=.2;cpu.p.phiH=.4;cpu.p.rhoL=1;cpu.p.rhoH=3;
      reference=state(.3);trial=reference;
      for(int j=0;j<cpu.n;j++)for(int i=0;i<cpu.n;i++)trial[cpu.id(5,j,i,0,cpu.n)]=.11+(i==column?.8:0);
      direct("cubature-only negative density",reference,trial,1);
      cpu.p=defaults;cpu.p.dmpDelta=10;cpu.p.boundTol=10;
      cpu.p.rhoH=std::numeric_limits<double>::max();
      trial=base;
      for(int j=0;j<cpu.n;j++)for(int i=0;i<cpu.n;i++)trial[cpu.id(5,j,i,0,cpu.n)]=.24-(i==column?.215:0);
      for(int j=0;j<cpu.n;j++)for(int i=0;i<cpu.n;i++)
        if(!std::isfinite(cpu.p.rho(trial[cpu.id(5,j,i,0,cpu.n)])))
          throw std::runtime_error("nonfinite-density fixture must have finite nodal densities");
      direct("cubature-only nonfinite density",base,trial,1);
      cpu.p=defaults;cpu.p.dmpDelta=10;cpu.p.boundTol=10;cpu.p.rhoH=1.2;
      for(int j=0;j<cpu.n;j++)for(int i=0;i<cpu.n;i++)trial[cpu.id(5,j,i,0,cpu.n)]=.01+(i==column?.8:0);
      direct("negative cubature index accepted",base,trial,0);
      for(int j=0;j<cpu.n;j++)for(int i=0;i<cpu.n;i++)trial[cpu.id(5,j,i,0,cpu.n)]=.98-(i==column?.8:0);
      direct("cubature EOS overshoot accepted",base,trial,0);
      trial=base;
      for(int j=0;j<cpu.n;j++)for(int i=0;i<cpu.n;i++) {
        double weight=cpu.I[row*cpu.n+j]*cpu.I[row*cpu.n+i];
        trial[cpu.id(5,j,i,11,cpu.n)]=std::copysign(.9*std::numeric_limits<double>::max(),weight);
      }
      // detect() accepts this; the separate hard admissibility check rejects
      // its overflowing cubature stress. Do not silently strengthen detect().
      if(cpu.admissible(trial))throw std::runtime_error("missing cubature stress overflow");
      direct("non-index cubature overflow ignored by detector",base,trial,0);
      cpu.p=defaults;
    }
  }
};
} // namespace

int main(int argc,char** argv) {
  if(argc<2 || argc>3)return 2;
  const std::string root=argv[1],mode=argc==3?argv[2]:"Serial";
  try {
    occa::device device;occa::json config;config["mode"]=mode;
    if(mode=="CUDA")config["device_id"]=0;
    device.setup(config);
    for(int degree: {1,2,3})for(bool walls: {false,true}) {
      Vec r,w,qr,qw;quadrature(degree,r,w,qr,qw);
      Check check(device,root,degree,walls,r,w,qr,qw);check.tests();check.interfaceTests();
      std::cout<<mode<<" N="<<degree<<(walls?" walls":" periodic")<<": "<<check.comparisons
               <<" raw masks agree with host; bounds, halos, thresholds and interface seed checks passed\n";
    }
  } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
  return 0;
}
