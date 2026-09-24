// Isolated subcell preparation versus the host routines used by hybrid().
#include "hczQuad2DKernelTest.hpp"
#include <cstring>
using namespace hczKernelTest;

namespace {
constexpr double sentinel=-987.25;
Vec native(const Vec& q,int m) {
  Vec out(q.size());
  for(int e=0;e<elements;e++)for(int node=0;node<m*m;node++)for(int k=0;k<12;k++)
    out[(e*12+k)*m*m+node]=q[(e*m*m+node)*12+k];
  return out;
}

struct Check {
  occa::device& device;
  Fixture f;
  occa::kernel project,scale;
  std::vector<int> all={4,1,5,0,3,2};
  double maxError=0,maxMeanError=0;
  int repairs=0,failures=0,cases=0;

  Check(occa::device& d,const std::string& root,int degree,
        const Vec& r,const Vec& w,const Vec& qr,const Vec& qw):
    device(d),f(degree,12,false,r,w,qr,qw) {
    auto props=kernelProperties(root,degree,12);
    const auto file=root+"/solvers/bns/okl/bnsHczSubcellQuad2D.okl";
    project=device.buildKernel(file,"bnsHczProjectSubcellQuad2D",props);
    scale=device.buildKernel(file,"bnsHczScaleSubcellQuad2D",props);
  }

  Vec subcell(double phi) const {
    const int ns=f.cpu.ns;
    Vec v(elements*ns*ns*12);
    for(int e=0;e<elements;e++)for(int node=0;node<ns*ns;node++)for(int k=0;k<12;k++)
      v[(e*ns*ns+node)*12+k]=k==0?phi:.03*(k-4)*std::sin(.3*(node+e+1));
    return v;
  }

  // Pass host-ordered projected cells to the real host scaling helper.
  // Individual decisions are obtained from a one-element reference operator.
  void checkScale(const Vec& sub,const std::vector<int>& ids,const char* label,
                  occa::memory* existing=nullptr) {
    const int ns=f.cpu.ns,np=ns*ns;
    auto one=f.cpu;one.local=one.total=1;
    Vec before=native(sub,ns),expected=before;
    std::vector<int> flags(elements+2,91),want=flags;
    for(int e:ids) {
      Vec cell(sub.begin()+e*np*12,sub.begin()+(e+1)*np*12),saved=cell;
      want[e]=!one.scaleSubcellIndex(cell);
      if(want[e])failures++;
      else if(cell!=saved)repairs++;
      for(int node=0;node<np;node++)for(int k=0;k<12;k++)
        expected[(e*12+k)*np+node]=cell[node*12+k];
    }
    // Also exercise the collective host helper on the whole input.
    Vec whole=sub;bool valid=f.cpu.scaleSubcellIndex(whole);
    if(ids.size()==elements && valid!=(std::count(want.begin(),want.begin()+elements,1)==0))
      throw std::runtime_error("host collective scaling decision mismatch");
    Vec padded=before;padded.resize(before.size()+2*np*12,sentinel);
    auto o_sub=existing?*existing:device.malloc<double>(padded.size(),padded.data());
    // A projection may differ by roundoff: use its actual output for the
    // scale-only comparison and conservation checks in the chained case.
    auto o_ids=device.malloc<int>(std::max(size_t(1),ids.size()));
    if(!ids.empty())o_ids.copyFrom(ids.data(),ids.size()*sizeof(int));
    auto o_flags=device.malloc<int>(flags.size(),flags.data());
    scale(int(ids.size()),o_ids,f.cpu.p.floor(),o_sub,o_flags);
    Vec actual(padded.size());o_sub.copyTo(actual.data());o_flags.copyTo(flags.data());
    if(flags!=want)throw std::runtime_error(std::string(label)+": invalid-mean flags differ from host");
    for(int e=0;e<elements;e++) {
      const bool selected=std::find(ids.begin(),ids.end(),e)!=ids.end();
      for(int k=0;k<12;k++) {
        double oldMean=0,newMean=0;
        for(int node=0;node<np;node++) {
          size_t i=(e*12+k)*np+node;
          double err=std::abs(actual[i]-expected[i]);maxError=std::max(maxError,err);
          if(!std::isfinite(actual[i]) || err>2e-12*std::max(1.0,std::abs(expected[i])))
            throw std::runtime_error(std::string(label)+": scaled moment differs from host");
          if((!selected || k>=6 || want[e]==1) && actual[i]!=before[i])
            throw std::runtime_error("scaling changed an unselected, hydro or invalid-mean entry");
          oldMean+=before[i]/np;newMean+=actual[i]/np;
          // At a mean within roundoff of the floor, even the host formula
          // can land on the floor. Preserve that behavior; demand strict
          // clearance when the host result has a resolved positive margin.
          if(selected && want[e]==0 && k==0 && expected[i]-f.cpu.p.floor()>1e-14
             && !(actual[i]>f.cpu.p.floor()))
            throw std::runtime_error(std::string(label)+": repaired subcell did not clear physical floor");
        }
        const double error=std::abs(oldMean-newMean);maxMeanError=std::max(maxMeanError,error);
        if(error>2e-13)throw std::runtime_error("scaling changed a conserved element mean");
      }
    }
    for(size_t i=before.size();i<actual.size();i++)if(actual[i]!=sentinel)
      throw std::runtime_error("scaling wrote halo sentinel");
    cases++;
  }

  void checkProject(const Vec& q,const std::vector<int>& ids,bool analytic=false) {
    const auto& cpu=f.cpu;const int ns=cpu.ns,np=ns*ns;
    Vec input=native(q,f.n),expected=native(cpu.project(q,f.n,ns,cpu.A),ns);
    Vec mat(f.n*ns);
    for(int i=0;i<f.n;i++)for(int s=0;s<ns;s++)mat[i*ns+s]=cpu.A[s*f.n+i];
    auto o_q=device.malloc<double>(input.size(),input.data()),o_A=device.malloc<double>(mat.size(),mat.data());
    auto o_ids=device.malloc<int>(std::max(size_t(1),ids.size()));
    if(!ids.empty())o_ids.copyFrom(ids.data(),ids.size()*sizeof(int));
    Vec seed((elements+2)*np*12,sentinel),actual(seed.size());
    auto o_sub=device.malloc<double>(seed.size(),seed.data());
    for(int repeat=0;repeat<2;repeat++) {
      project(int(ids.size()),o_ids,o_A,o_q,o_sub);o_sub.copyTo(actual.data());
      for(int e=0;e<elements+2;e++)for(int k=0;k<12;k++) {
        const bool selected=std::find(ids.begin(),ids.end(),e)!=ids.end();
        double mean=0,dgMean=0;
        for(int node=0;node<np;node++) {
          size_t i=(e*12+k)*np+node;
          const double want=selected?expected[i]:sentinel;
          double err=std::abs(actual[i]-want);maxError=std::max(maxError,err);
          if(!std::isfinite(actual[i]) || err>2e-12 || (!selected && actual[i]!=sentinel))
            throw std::runtime_error("DG-to-subcell projection differs from host or overwrote sentinel");
          mean+=actual[i]/np;
          if(selected && analytic) {
            double x=-1+(2*(node%ns)+1.0)/ns,y=-1+(2*(node/ns)+1.0)/ns;
            double wantPoly=.2+.003*(k+1)*(x+.5*y)+.002*x*y;
            if(f.n>=3)wantPoly+=.01*(x*x+1.0/(3*ns*ns));
            if(std::abs(actual[i]-wantPoly)>2e-13)throw std::runtime_error("analytic subcell average mismatch");
          }
        }
        if(selected) {
          for(int j=0;j<f.n;j++)for(int i=0;i<f.n;i++)
            dgMean+=.25*cpu.weights[j]*cpu.weights[i]*q[cpu.id(e,j,i,k,f.n)];
          const double err=std::abs(mean-dgMean);maxMeanError=std::max(maxMeanError,err);
          if(err>2e-13)throw std::runtime_error("projection lost the DG element mean");
        }
      }
    }
    Vec after(input.size());o_q.copyTo(after.data());
    if(after!=input)throw std::runtime_error("projection changed DG input");
    after.resize(mat.size());o_A.copyTo(after.data());
    if(after!=mat)throw std::runtime_error("projection changed interpolation matrix");
    // Chain the actual projected device output into scaling. Populate host
    // data from those same values to isolate scaling from projection roundoff.
    Vec projected(elements*np*12);
    for(int e=0;e<elements;e++)for(int node=0;node<np;node++)for(int k=0;k<12;k++)
      projected[(e*np+node)*12+k]=actual[(e*12+k)*np+node];
    checkScale(projected,ids,"projection-to-scaling",&o_sub);
    cases++;
  }

  void tests() {
    Vec q(elements*f.np*12);
    for(int e=0;e<elements;e++)for(int j=0;j<f.n;j++)for(int i=0;i<f.n;i++)for(int k=0;k<12;k++) {
      double x=f.cpu.nodes[i],y=f.cpu.nodes[j];
      q[f.cpu.id(e,j,i,k,f.n)]=.2+.003*(k+1)*(x+.5*y)+.002*x*y+(f.n>=3?.01*x*x:0);
    }
    for(const auto& ids:std::vector<std::vector<int>>{all,{5,0,3},{2},{}})checkProject(q,ids,true);
    for(int pattern=0;pattern<3;pattern++) {
      f.fill(pattern);q=f.state;
      checkProject(q,all);
    }
    const auto defaults=f.cpu.p;
    for(bool densityFloor: {false,true}) {
      if(densityFloor){f.cpu.p.phiL=.2;f.cpu.p.phiH=.4;f.cpu.p.rhoL=1;f.cpu.p.rhoH=3;}
      const int np=f.cpu.ns*f.cpu.ns;const double floor=f.cpu.p.floor();
      Vec sub=subcell(.3);
      checkScale(sub,all,"inactive scaling");
      sub[(4*np)*12]=floor-.04;
      for(const auto& ids:std::vector<std::vector<int>>{all,{4,0,3},{1},{}})
        checkScale(sub,ids,"repairable minimum");
      sub[(4*np)*12]=floor;
      checkScale(sub,all,"minimum equals floor");
      sub[(4*np)*12]=std::nextafter(floor,std::numeric_limits<double>::infinity());
      checkScale(sub,all,"minimum just above floor");
      // Mixed repairable, irreparable and untouched elements in one call.
      sub[(4*np)*12]=floor-.04;
      for(int node=0;node<np;node++)sub[(2*np+node)*12]=.5*floor;
      checkScale(sub,all,"invalid mean and repairable neighbor");
      // Exact average-at-floor test: all index mass in one subcell.
      for(int node=0;node<np;node++)sub[(2*np+node)*12]=node==0?floor*np:0;
      checkScale(sub,all,"mean at floor");
      f.cpu.p=defaults;
    }
    // Valid nodal states can project below the floor for N>=2. Arrange a
    // large endpoint value where an averaging coefficient is negative.
    if(f.n>=3) {
      int column=-1;
      for(int s=0;s<f.cpu.ns;s++)for(int i=0;i<f.n;i++)if(f.cpu.A[s*f.n+i]<-.02)column=i;
      if(column<0)throw std::runtime_error("missing projection undershoot fixture");
      f.cpu.p.rhoH=1.2;
      for(int e=0;e<elements;e++)for(int j=0;j<f.n;j++)for(int i=0;i<f.n;i++)for(int k=0;k<12;k++)
        q[f.cpu.id(e,j,i,k,f.n)]=k==0?.01+(i==column?.8:0):.02*(k-4)*std::sin(i+2*j+.3*e);
      if(!f.cpu.admissible(q))throw std::runtime_error("projection undershoot fixture must be admissible DG");
      int before=repairs;checkProject(q,all);
      if(repairs==before)throw std::runtime_error("projection undershoot did not trigger scaling");
      f.cpu.p=defaults;
    }
    if(!repairs || !failures)throw std::runtime_error("missing scaling branch coverage");
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
    for(int degree: {1,2,3}) {
      Vec r,w,qr,qw;quadrature(degree,r,w,qr,qw);
      Check check(device,root,degree,r,w,qr,qw);check.tests();
      std::cout<<mode<<" N="<<degree<<": "<<check.cases<<" cases passed; repairs="<<check.repairs
               <<", failures="<<check.failures<<", max error="<<check.maxError
               <<", max mean error="<<check.maxMeanError<<"\n";
    }
  } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
  return 0;
}
