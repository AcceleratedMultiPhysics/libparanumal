// Subcell source preparation with reused pointwise HCZ kernels.
#include "hczQuad2DKernelTest.hpp"
using namespace hczKernelTest;

namespace {
constexpr double sentinel=-987.25;
struct Check {
  occa::device& device;
  Fixture f;
  occa::kernel thermo,gradient,source,facesKernel,rhsKernel,reconstruct;
  std::vector<int> all={4,1,5,0,3,2},neighbors;
  occa::memory o_neighbors;
  double maxGradient=0,maxSource=0,maxRhs=0,maxReconstruction=0;
  int cases=0;
  Check(occa::device& d,const std::string& root,int degree,bool walls,
        const Vec& r,const Vec& w,const Vec& qr,const Vec& qw):
    device(d),f(degree,12,walls,r,w,qr,qw),neighbors(elements*9) {
    f.cpu.total=2*elements;
    f.cpu.exchange=[](Vec& v,int stride){std::copy_n(v.begin(),elements*stride,v.begin()+elements*stride);};
    for(int e=0;e<elements;e++)for(int k=0;k<9;k++) {
      if(k!=4 && f.cpu.neighbor[e][k]>=0 && (e+k)%2)f.cpu.neighbor[e][k]+=elements;
      neighbors[e*9+k]=f.cpu.neighbor[e][k];
    }
    o_neighbors=device.malloc<int>(neighbors.size(),neighbors.data());
    auto props=kernelProperties(root,degree,12);
    gradient=device.buildKernel(root+"/solvers/bns/okl/bnsHczSubcellGradientQuad2D.okl","bnsHczSubcellGradientQuad2D",props);
    facesKernel=device.buildKernel(root+"/solvers/bns/okl/bnsHczWenoFaceQuad2D.okl","bnsHczWenoFaceQuad2D",props);
    rhsKernel=device.buildKernel(root+"/solvers/bns/okl/bnsHczSubcellRhsQuad2D.okl","bnsHczSubcellRhsQuad2D",props);
    reconstruct=device.buildKernel(root+"/solvers/bns/okl/bnsHczReconstructQuad2D.okl","bnsHczReconstructQuad2D",props);
    // The existing pointwise kernels need only a different grid shape.
    const int ns=f.cpu.ns;
    props["defines/p_Nq"]=ns;props["defines/p_Np"]=ns*ns;
    props["defines/p_NblockV"]=std::max(1,256/(ns*ns));
    thermo=device.buildKernel(root+"/solvers/bns/okl/bnsHczThermodynamicsQuad2D.okl","bnsHczThermodynamicsQuad2D",props);
    source=device.buildKernel(root+"/solvers/bns/okl/bnsHczSourceQuad2D.okl","bnsHczSourceQuad2D",props);
  }
  occa::memory idsBuffer(const std::vector<int>& ids) {
    auto out=device.malloc<int>(std::max(size_t(1),ids.size()));
    if(!ids.empty())out.copyFrom(ids.data(),ids.size()*sizeof(int));
    return out;
  }
  Vec native(const Vec& q) const {
    const int np=f.cpu.ns*f.cpu.ns;Vec out(q.size());
    for(int e=0;e<2*elements;e++)for(int node=0;node<np;node++)for(int k=0;k<12;k++)
      out[(e*12+k)*np+node]=q[(e*np+node)*12+k];
    return out;
  }
  double x(int e,int i) const {return (e%nxElements+(i+.5)/f.cpu.ns)*f.cpu.hx;}
  double y(int e,int j) const {return (e/nxElements+(j+.5)/f.cpu.ns)*f.cpu.hy;}
  Vec referenceGradient(const Vec& potentials) const {
    const int np=f.cpu.ns*f.cpu.ns;Vec out(2*elements*4*np,0);
    for(int p=0;p<2;p++) {
      Vec values(2*elements*np);
      for(int e=0;e<2*elements;e++)for(int node=0;node<np;node++)values[e*np+node]=potentials[(e*2+p)*np+node];
      for(int axis=0;axis<2;axis++) {
        auto deriv=f.cpu.derivative(values,axis,true,p==1);
        for(int e=0;e<elements;e++)for(int node=0;node<np;node++)out[(e*4+2*p+axis)*np+node]=deriv[e*np+node];
      }
    }
    return out;
  }
  void compare(double value,double expected,double& maximum,const char* label) {
    double error=std::abs(value-expected);maximum=std::max(maximum,error);
    if(!std::isfinite(value) || error>3e-11*std::max(1.0,std::abs(expected)))throw std::runtime_error(label);
  }
  void prescribed(int pattern,const std::vector<int>& ids) {
    const int ns=f.cpu.ns,np=ns*ns;
    Vec potentials(2*elements*2*np);
    for(int e=0;e<elements;e++)for(int j=0;j<ns;j++)for(int i=0;i<ns;i++)for(int p=0;p<2;p++) {
      double value=0;
      if(pattern==1)value=.3*(p+1);
      if(pattern==2)value=.2+.04*x(e,i)+.07*y(e,j)+.03*y(e,j)*y(e,j);
      if(pattern==3)value=p==0?.13*std::sin(.7*i+.3*j+e):0;
      if(pattern==4)value=p==1?.13*std::cos(.3*i+.7*j+e):0;
      potentials[(e*2+p)*np+j*ns+i]=value;
    }
    f.cpu.exchange(potentials,2*np);
    const auto expected=referenceGradient(potentials);
    auto o_ids=idsBuffer(ids),o_p=device.malloc<double>(potentials.size(),potentials.data());
    Vec initial(2*elements*4*np,sentinel),actual(initial.size());
    auto o_g=device.malloc<double>(initial.size(),initial.data());
    for(int repeat=0;repeat<2;repeat++) {
      gradient(int(ids.size()),o_ids,o_neighbors,f.cpu.hx,f.cpu.hy,o_p,o_g);o_g.copyTo(actual.data());
      for(int e=0;e<2*elements;e++)for(int k=0;k<4;k++)for(int node=0;node<np;node++) {
        const int pos=(e*4+k)*np+node;
        const bool selected=std::find(ids.begin(),ids.end(),e)!=ids.end();
        compare(actual[pos],selected?expected[pos]:sentinel,maxGradient,"subcell derivative differs from host");
        if(!selected && actual[pos]!=sentinel)throw std::runtime_error("gradient overwrote unselected/halo entry");
        if(selected && pattern==2) {
          const int j=node/ns,i=node%ns;
          if(k%2==0 && (e%nxElements*ns+i)>0 && (e%nxElements*ns+i)<nxElements*ns-1)
            compare(actual[pos],.04,maxGradient,"analytic x derivative mismatch");
          if(k%2 && f.walls) {
            double slope=.07+.06*y(e,j);
            if(k==1 && j==0 && neighbors[9*e+1]<0)slope=.5*(slope+.03*f.cpu.hy/ns);
            if(k==1 && j==ns-1 && neighbors[9*e+7]<0)slope=.5*(slope-.03*f.cpu.hy/ns);
            compare(actual[pos],slope,maxGradient,"copied/one-sided wall derivative mismatch");
          }
        }
      }
    }
    Vec unchanged(potentials.size());o_p.copyTo(unchanged.data());
    if(unchanged!=potentials)throw std::runtime_error("gradient changed potentials");
    cases++;
  }
  Vec state(int pattern) const {
    const int ns=f.cpu.ns;const auto& p=f.cpu.p;
    Vec q(2*elements*12*ns*ns,0);
    for(int e=0;e<elements;e++)for(int j=0;j<ns;j++)for(int i=0;i<ns;i++) {
      State s{};s[0]=pattern==1?.15+.015*std::sin(.7*i+.3*j+e):.15;
      s[6]=pattern==2?.2-p.rho(s[0])*p.gravity*y(e,j):0;
      if(pattern==1) {
        s[7]=.03*std::sin(i+.3*e);s[8]=.025*std::cos(j+.5*e);
        for(int k: {1,2,3,4,5,9,10,11})s[k]=.015*(k-4)*std::sin(.2*k+i+.7*j+.4*e);
      }
      for(int k=0;k<12;k++)q[f.cpu.id(e,j,i,k,ns)]=s[k];
    }
    f.cpu.exchange(q,12*ns*ns);return q;
  }
  void chain(int pattern,const std::vector<int>& ids,bool full) {
    const int ns=f.cpu.ns,np=ns*ns;const auto& p=f.cpu.p;
    Vec host=state(pattern),q=native(host),seed(q.size(),.013);
    auto o_q=device.malloc<double>(q.size(),q.data()),o_ids=idsBuffer(ids),o_all=idsBuffer(all);
    Vec density(2*elements*np,sentinel),potentials(2*elements*2*np,sentinel),grads(2*elements*4*np,sentinel);
    auto o_rho=device.malloc<double>(density.size(),density.data()),o_p=device.malloc<double>(potentials.size(),potentials.data());
    auto o_g=device.malloc<double>(grads.size(),grads.data()),o_rhs=device.malloc<double>(seed.size(),seed.data());
    // Thermodynamics runs for all owned elements needed by selected stencils.
    thermo(elements,o_all,p.c,p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.eosRatio,o_q,o_rho,o_p);
    o_rho.copyTo(density.data());o_p.copyTo(potentials.data());
    double thermoError=0;
    for(int e=0;e<elements;e++)for(int node=0;node<np;node++) {
      double phi=q[e*12*np+node];
      compare(density[e*np+node],p.rho(phi),thermoError,"subcell density mismatch");
      compare(potentials[e*2*np+node],p.cs(phi)-p.c*p.c*phi,thermoError,"index potential mismatch");
      compare(potentials[(e*2+1)*np+node],q[(e*12+6)*np+node]-p.c*p.c*p.rho(phi),thermoError,"pressure correction mismatch");
    }
    for(size_t i=elements*2*np;i<potentials.size();i++)if(potentials[i]!=sentinel)throw std::runtime_error("thermodynamics wrote halo");
    // Supply only the potential halo, preserving the owned device results.
    f.cpu.exchange(potentials,2*np);
    o_p.copyFrom(potentials.data()+elements*2*np,elements*2*np*sizeof(double),elements*2*np*sizeof(double));
    gradient(int(ids.size()),o_ids,o_neighbors,f.cpu.hx,f.cpu.hy,o_p,o_g);
    Vec expected=seed;
    if(full) {
      auto o_nb=device.malloc<int>(neighbors.size(),neighbors.data());
      auto o_faces=device.malloc<double>(elements*2*12*ns*(ns+1));
      facesKernel(elements,o_all,o_nb,p.c,p.alpha,p.wenoEps,o_q,o_faces);
      rhsKernel(elements,o_all,p.c,p.tauPhi,p.tauG,p.rhoL,p.rhoH,p.phiL,p.phiH,f.cpu.hx,f.cpu.hy,o_q,o_faces,o_rhs);
      for(int e=0;e<elements;e++)for(int j=0;j<ns;j++)for(int i=0;i<ns;i++) {
        auto co=hcz::collision(f.cpu.get(host,e,j,i,ns),p);
        auto xp=f.cpu.wenoFace(host,e,j,i,0),xm=f.cpu.wenoFace(host,e,j,i-1,0);
        auto yp=f.cpu.wenoFace(host,e,j,i,1),ym=f.cpu.wenoFace(host,e,j-1,i,1);
        for(int k=0;k<12;k++)expected[(e*12+k)*np+j*ns+i]=co[k]-(xp[k]-xm[k])/(f.cpu.hx/ns)-(yp[k]-ym[k])/(f.cpu.hy/ns);
      }
    }
    Vec hostSource(host.size(),0);f.cpu.addSources(host,hostSource,true);
    Vec addition=native(hostSource);
    for(int repeat=1;repeat<=2;repeat++) {
      source(int(ids.size()),o_ids,p.c,p.gravity,p.rhoL,p.rhoH,p.phiL,p.phiH,o_q,o_g,o_rhs);
      Vec actual(seed.size());o_rhs.copyTo(actual.data());
      for(int e=0;e<2*elements;e++)for(int k=0;k<12;k++)for(int node=0;node<np;node++) {
        const bool selected=std::find(ids.begin(),ids.end(),e)!=ids.end();
        size_t pos=(e*12+k)*np+node;
        compare(actual[pos],expected[pos]+(selected?repeat*addition[pos]:0),full?maxRhs:maxSource,"subcell source/residual differs from host");
        if((!selected || k==0 || k==7) && actual[pos]!=expected[pos] && !full)
          throw std::runtime_error("source changed unselected or zero-source entry");
        if(selected && pattern==2 && !full) {
          double analytic=k==8?-p.c*p.rho(.15)*p.gravity:0;
          compare(actual[pos],seed[pos]+repeat*analytic,maxSource,"hydrostatic subcell source normalization mismatch");
        }
      }
      if(full) {
        double massChange=0;
        for(int e=0;e<elements;e++)for(int node=0;node<np;node++)massChange+=actual[e*12*np+node]*f.cpu.hx*f.cpu.hy/np;
        if(std::abs(massChange)>2e-12)throw std::runtime_error("complete subcell residual lost index conservation");
        // Feed the actual full device residual into conservative DG
        // reconstruction and compare the entire sequence with the host.
        Vec fv(host.size(),0),R(f.n*ns);
        for(int e=0;e<elements;e++)for(int node=0;node<np;node++)for(int k=0;k<12;k++)
          fv[(e*np+node)*12+k]=expected[(e*12+k)*np+node]+repeat*addition[(e*12+k)*np+node];
        for(int a=0;a<ns;a++)for(int i=0;i<f.n;i++)R[i+a*f.n]=f.cpu.R[i*ns+a];
        const Vec reference=f.cpu.project(fv,ns,f.n,f.cpu.R,true);
        Vec dg(2*elements*12*f.np,sentinel);
        auto o_R=device.malloc<double>(R.size(),R.data());
        auto o_w=device.malloc<double>(f.cpu.weights.size(),f.cpu.weights.data());
        auto o_dg=device.malloc<double>(dg.size(),dg.data());
        reconstruct(elements,o_all,o_R,o_w,o_rhs,o_dg);o_dg.copyTo(dg.data());
        for(int e=0;e<elements;e++)for(int k=0;k<12;k++) {
          double subMean=0,dgMean=0;
          for(int node=0;node<np;node++)subMean+=actual[(e*12+k)*np+node]/np;
          for(int j=0;j<f.n;j++)for(int i=0;i<f.n;i++) {
            const double value=dg[(e*12+k)*f.np+j*f.n+i];
            compare(value,reference[f.cpu.id(e,j,i,k,f.n)],maxReconstruction,"full subcell-to-DG sequence differs from host");
            dgMean+=.25*f.cpu.weights[j]*f.cpu.weights[i]*value;
          }
          if(std::abs(subMean-dgMean)>3e-12*std::max(1.0,std::abs(subMean)))
            throw std::runtime_error("full reconstructed residual changed a moment mean");
        }
        for(size_t i=elements*12*f.np;i<dg.size();i++)if(dg[i]!=sentinel)
          throw std::runtime_error("reconstruction wrote halo output");
      }
    }
    Vec after(q.size());o_q.copyTo(after.data());if(after!=q)throw std::runtime_error("source chain changed subcells");
    Vec afterP(potentials.size());o_p.copyTo(afterP.data());if(afterP!=potentials)throw std::runtime_error("source chain changed potentials");
    cases++;
  }
  void tests() {
    f.cpu.p.kappa=0;
    for(int pattern=0;pattern<5;pattern++)prescribed(pattern,all);
    for(const auto& ids:std::vector<std::vector<int>>{{5,0,3},{2},{}})prescribed(2,ids);
    for(int params=0;params<2;params++) {
      auto& p=f.cpu.p;p.c=params?.7:1.3;p.rhoL=params?.7:1;p.rhoH=params?1.2:3;
      p.eosB=params?3.2:4;p.eosRatio=params?2.1:3;p.tauPhi=.0025;p.tauG=.0017;
      p.gravity=0;chain(0,all,false);chain(0,all,true);
      p.gravity=f.walls?.04:0;chain(1,all,false);chain(1,all,true);
      for(const auto& ids:std::vector<std::vector<int>>{{5,0,3},{2},{}})chain(1,ids,false);
      if(f.walls){chain(2,all,false);chain(2,all,true);}
    }
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
               <<" cases passed; gradient="<<check.maxGradient<<", source="<<check.maxSource<<", full residual="<<check.maxRhs
               <<", reconstructed="<<check.maxReconstruction<<"\n";
    }
  } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
  return 0;
}
