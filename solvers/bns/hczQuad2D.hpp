// HCZ independent-index H2 DG / WENO3-Z operators on Cartesian quadrilaterals.
// Numerical formulas ported from the two HCZ Python reference scripts.
// Layout: [element][y node][x node][moment]. Halos contain complete elements.
#ifndef BNS_HCZ_QUAD_2D_HPP
#define BNS_HCZ_QUAD_2D_HPP
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hcz {
struct SubcellMeanError: std::runtime_error {
  SubcellMeanError():std::runtime_error("HCZ subcell index mean cannot be repaired conservatively") {}
};
using Vec = std::vector<double>;
using State = std::array<double,12>;
constexpr double sq2 = 1.4142135623730950488;
struct Parameters {
  double c=1, tauPhi=0.00078125, tauG=0.00078125;
  double rhoL=1, rhoH=3, phiL=.02381, phiH=.2508;
  double eosB=4, eosRatio=3, kappa=0, gravity=.04;
  double alpha=1.75, wenoEps=1e-12;
  double dmpDelta=.005, dmpEps=.05, boundTol=.01;
  int buffer=1, passes=2;
  bool subcell=true;
  bool characteristicWeno=true;
  bool interfaceSubcells=false;
  double interfaceThreshold=.01;
  double rho(double phi) const { return rhoL+(phi-phiL)*(rhoH-rhoL)/(phiH-phiL); }
  double cs(double phi) const {
    const double eta=eosB*phi/4;
    return c*c*phi*(1+eta+eta*eta-eta*eta*eta)/std::pow(1-eta,3)
           -eosRatio*eosB*c*c*phi*phi;
  }
  double floor() const {
    // Protect both index and mapped physical density.
    return std::max(1e-10, phiL-rhoL*(phiH-phiL)/(rhoH-rhoL)+1e-10);
  }
  bool physical(double phi) const {
    return std::isfinite(phi) && phi>floor() && rho(phi)>0 && eosB*phi<3.999;
  }
};

inline State flux(const State& q, int axis, double c) {
  State f{};
  for(int s: {0,6}) {
    if(axis==0) {
      f[s]=c*q[s+1]; f[s+1]=c*(q[s]+sq2*q[s+4]);
      f[s+2]=c*q[s+3]; f[s+3]=c*q[s+2]; f[s+4]=sq2*c*q[s+1];
    } else {
      f[s]=c*q[s+2]; f[s+1]=c*q[s+3];
      f[s+2]=c*(q[s]+sq2*q[s+5]); f[s+3]=c*q[s+1]; f[s+5]=sq2*c*q[s+2];
    }
  }
  return f;
}
inline State upwind(const State& l, const State& r, int axis, double c) {
  // Exact |A| on the 3x3 acoustic block and 2x2 shear block.
  State d{}, z{};
  for(int k=0;k<12;k++) d[k]=r[k]-l[k];
  for(int s: {0,6}) {
    int u=s+1+axis, stress=s+4+axis, v=s+2-axis;
    z[s]=(d[s]+sq2*d[stress])/std::sqrt(3.0);
    z[u]=std::sqrt(3.0)*d[u];
    z[stress]=(sq2*d[s]+2*d[stress])/std::sqrt(3.0);
    z[v]=d[v]; z[s+3]=d[s+3];
  }
  State fl=flux(l,axis,c), fr=flux(r,axis,c), out{};
  for(int k=0;k<12;k++) out[k]=.5*(fl[k]+fr[k]-c*z[k]);
  return out;
}
// Orthonormal characteristics of A_axis/c, per six-moment family:
// acoustic +/-sqrt(3), shear +/-1, acoustic zero, transverse-stress zero.
inline State characteristics(const State& q, int axis) {
  State w{};
  const double i2=1/std::sqrt(2.0),i3=1/std::sqrt(3.0),i6=1/std::sqrt(6.0);
  for(int s: {0,6}) {
    const int u=s+1+axis,v=s+2-axis,stress=s+4+axis;
    w[s]=i6*q[s]+i2*q[u]+i3*q[stress];
    w[s+1]=i6*q[s]-i2*q[u]+i3*q[stress];
    w[s+2]=i2*(q[v]+q[s+3]);
    w[s+3]=i2*(q[v]-q[s+3]);
    w[s+4]=i3*(-sq2*q[s]+q[stress]);
    w[s+5]=q[s+5-axis];
  }
  return w;
}
inline State moments(const State& w, int axis) {
  State q{};
  const double i2=1/std::sqrt(2.0),i3=1/std::sqrt(3.0),i6=1/std::sqrt(6.0);
  for(int s: {0,6}) {
    q[s]=i6*(w[s]+w[s+1])-sq2*i3*w[s+4];
    q[s+1+axis]=i2*(w[s]-w[s+1]);
    q[s+4+axis]=i3*(w[s]+w[s+1]+w[s+4]);
    q[s+2-axis]=i2*(w[s+2]+w[s+3]);
    q[s+3]=i2*(w[s+2]-w[s+3]);
    q[s+5-axis]=w[s+5];
  }
  return q;
}
inline State equilibrium(const State& q, const Parameters& p) {
  const double phi=q[0], rho=p.rho(phi), u=q[7]/(rho*p.c), v=q[8]/(rho*p.c);
  State eq{};
  eq[0]=phi; eq[1]=phi*u/p.c; eq[2]=phi*v/p.c;
  eq[3]=phi*u*v/(p.c*p.c); eq[4]=phi*u*u/(sq2*p.c*p.c); eq[5]=phi*v*v/(sq2*p.c*p.c);
  eq[6]=q[6]; eq[7]=q[7]; eq[8]=q[8];
  eq[9]=rho*u*v; eq[10]=rho*u*u/sq2; eq[11]=rho*v*v/sq2;
  return eq;
}
inline State collision(const State& q, const Parameters& p) {
  State out=equilibrium(q,p);
  for(int k=0;k<12;k++) out[k]=(out[k]-q[k])/(k<6?p.tauPhi:p.tauG);
  out[0]=out[6]=out[7]=out[8]=0;
  return out;
}
inline State source(const State& q, const Parameters& p, double px, double py,
                    double rx, double ry, double capx, double capy) {
  const double rho=p.rho(q[0]), u=q[7]/(rho*p.c), v=q[8]/(rho*p.c);
  const double fx=capx, fy=capy-rho*p.gravity, c2=p.c*p.c;
  State s{};
  s[1]=-px/p.c; s[2]=-py/p.c; s[3]=-(u*py+v*px)/c2;
  s[4]=-sq2*u*px/c2; s[5]=-sq2*v*py/c2;
  s[6]=-(u*rx+v*ry); s[7]=p.c*fx; s[8]=p.c*fy;
  s[9]=u*(fy-ry)+v*(fx-rx); s[10]=sq2*u*(fx-rx); s[11]=sq2*v*(fy-ry);
  return s;
}

class QuadOperator {
public:
  Parameters p;
  int n, ns, nc, local, total;
  double hx, hy;
  // Neighbor ordering: (dy+1)*3+dx+1; -1 marks a physical y wall.
  std::vector<std::array<int,9>> neighbor;
  Vec nodes, weights, D, I, P, A, R;
  std::function<void(Vec&,int)> exchange=[](Vec&,int){};
  std::function<bool(bool)> any=[](bool b){return b;};
  long long candidateChecks=0, recomputations=0;
  int lastTroubled=0, maxRecomputePasses=0;
  Vec lastSubcellMask; // HOST LSERK4: final buffered mask of the latest stage.
  Vec lastInterfaceMask; // Unbuffered interface seed for the reported update.

  QuadOperator(int degree, int nlocal, int ntotal, double dx, double dy,
               const Vec& r, const Vec& w, const Vec& qr, const Vec& qw):
    n(degree+1), ns(2*degree+1), nc(degree+2), local(nlocal), total(ntotal),
    hx(dx), hy(dy), neighbor(nlocal), nodes(r), weights(w) {
    D.assign(n*n,0); I.resize(nc*n); P.resize(n*nc); A.assign(ns*n,0);
    Vec bary(n,1);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) if(i!=j) bary[i]/=nodes[i]-nodes[j];
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) if(i!=j) {
      D[i*n+j]=bary[j]/(bary[i]*(nodes[i]-nodes[j])); D[i*n+i]-=D[i*n+j];
    }
    auto lagrange=[&](int i,double x) {
      double a=1; for(int j=0;j<n;j++) if(j!=i) a*=(x-nodes[j])/(nodes[i]-nodes[j]); return a;
    };
    for(int a=0;a<nc;a++) for(int i=0;i<n;i++) {
      I[a*n+i]=lagrange(i,qr[a]); P[i*nc+a]=I[a*n+i]*qw[a]/weights[i];
    }
    for(int s=0;s<ns;s++) for(int a=0;a<nc;a++) for(int i=0;i<n;i++)
      A[s*n+i]+=.5*qw[a]*lagrange(i,-1+(2*s+1.0+qr[a])/ns);
    // R=(A^T A)^-1 A^T. Tensor-product reconstruction followed by an
    // explicit mean correction below, as in the Python pseudoinverse port.
    Vec aug(n*2*n,0);
    for(int i=0;i<n;i++) {
      for(int j=0;j<n;j++) for(int s=0;s<ns;s++) aug[i*2*n+j]+=A[s*n+i]*A[s*n+j];
      aug[i*2*n+n+i]=1;
    }
    for(int k=0;k<n;k++) {
      int pivot=k; for(int j=k+1;j<n;j++) if(std::abs(aug[j*2*n+k])>std::abs(aug[pivot*2*n+k])) pivot=j;
      for(int j=0;j<2*n;j++) std::swap(aug[k*2*n+j],aug[pivot*2*n+j]);
      double d=aug[k*2*n+k]; if(std::abs(d)<1e-14) throw std::runtime_error("HCZ singular subcell reconstruction");
      for(int j=0;j<2*n;j++) aug[k*2*n+j]/=d;
      for(int i=0;i<n;i++) if(i!=k) {
        d=aug[i*2*n+k]; for(int j=0;j<2*n;j++) aug[i*2*n+j]-=d*aug[k*2*n+j];
      }
    }
    R.assign(n*ns,0);
    for(int i=0;i<n;i++) for(int s=0;s<ns;s++) for(int j=0;j<n;j++) R[i*ns+s]+=aug[i*2*n+n+j]*A[s*n+j];
  }
  size_t id(int e,int j,int i,int k,int m,int fields=12) const {
    return ((size_t(e)*m+j)*m+i)*fields+k;
  }
  State get(const Vec& q,int e,int j,int i,int m) const {
    State a{}; for(int k=0;k<12;k++) a[k]=q[id(e,j,i,k,m)]; return a;
  }
  State sample(const Vec& q,int e,int j,int i,int m,bool kinetic=true) const {
    int dx=0,dy=0;
    if(i<0) {dx=-1;i+=m;} else if(i>=m) {dx=1;i-=m;}
    if(j<0) dy=-1; else if(j>=m) dy=1;
    int other=neighbor[e][(dy+1)*3+dx+1]; bool wall=other<0;
    if(wall) {other=neighbor[e][3+dx+1]; j=(j<0)?-j-1:2*m-1-j;}
    else if(dy) j-=dy*m;
    State a=get(q,other,j,i,m);
    if(wall && kinetic) for(int k: {1,2,7,8}) a[k]=-a[k];
    return a;
  }
  double scalarSample(const Vec& q,int e,int j,int i,int m) const {
    int dx=0,dy=0;
    if(i<0) {dx=-1;i+=m;} else if(i>=m) {dx=1;i-=m;}
    if(j<0) dy=-1; else if(j>=m) dy=1;
    int other=neighbor[e][(dy+1)*3+dx+1];
    if(other<0) {other=neighbor[e][3+dx+1]; j=std::max(0,std::min(m-1,j));}
    else if(dy) j-=dy*m;
    return q[id(other,j,i,0,m,1)];
  }
  Vec derivative(Vec value,int axis,bool fv=false,bool oneSided=false) const {
    int m=fv?ns:n; double h=axis?hy:hx;
    exchange(value,m*m);
    Vec out(size_t(total)*m*m,0);
    for(int e=0;e<local;e++) for(int j=0;j<m;j++) for(int i=0;i<m;i++) {
      auto at=[&](int offset) {return scalarSample(value,e,j+(axis?offset:0),i+(axis?0:offset),m);};
      double d=0;
      if(fv) {
        d=(at(1)-at(-1))/(2*h/m);
        if(oneSided && axis && j==0 && neighbor[e][1]<0) d=(-3*at(0)+4*at(1)-at(2))/(2*h/m);
        if(oneSided && axis && j==m-1 && neighbor[e][7]<0) d=(3*at(0)-4*at(-1)+at(-2))/(2*h/m);
      } else {
        int t=axis?j:i;
        for(int a=0;a<n;a++) d+=D[t*n+a]*value[id(e,axis?a:j,axis?i:a,0,n,1)];
        if(t==0 || t==n-1) {
          int sign=t==0?-1:1;
          d+=sign*.5*(at(sign)-at(0))/weights[t];
        }
        d*=2/h;
      }
      out[id(e,j,i,0,m,1)]=d;
    }
    return out;
  }
  Vec project(const Vec& q,int from,int to,const Vec& mat,bool meanFix=false) const {
    Vec out(size_t(total)*to*to*12,0);
    for(int e=0;e<local;e++) for(int j=0;j<to;j++) for(int i=0;i<to;i++)
      for(int a=0;a<from;a++) for(int b=0;b<from;b++) for(int k=0;k<12;k++)
        out[id(e,j,i,k,to)]+=mat[j*from+a]*mat[i*from+b]*q[id(e,a,b,k,from)];
    if(meanFix) for(int e=0;e<local;e++) for(int k=0;k<12;k++) {
      double avg=0, dg=0;
      for(int j=0;j<from;j++) for(int i=0;i<from;i++) avg+=q[id(e,j,i,k,from)]/(from*from);
      for(int j=0;j<to;j++) for(int i=0;i<to;i++) dg+=.25*weights[j]*weights[i]*out[id(e,j,i,k,to)];
      for(int j=0;j<to;j++) for(int i=0;i<to;i++) out[id(e,j,i,k,to)]+=avg-dg;
    }
    return out;
  }
  bool admissible(const Vec& q,bool cubature=true) const {
    bool bad=false;
    for(int e=0;e<local;e++) for(int j=0;j<n;j++) for(int i=0;i<n;i++) {
      State s=get(q,e,j,i,n);
      for(double a:s) bad|=!std::isfinite(a);
      bad|=!p.physical(s[0]);
    }
    if(cubature) {
      Vec v=project(q,n,nc,I);
      // Cubature evaluates BGK, not the EOS: the Python core requires a
      // finite positive mapped density here, not a positive cubature index.
      for(int e=0;e<local;e++) for(int j=0;j<nc;j++) for(int i=0;i<nc;i++) {
        State s=get(v,e,j,i,nc);
        for(double a:s) bad|=!std::isfinite(a);
        bad|=!(p.rho(s[0])>0);
      }
    }
    return !any(bad);
  }
  Vec dg(Vec q) const {
    if(!admissible(q)) throw std::runtime_error("HCZ nonphysical nodal/cubature state entering DG RHS");
    exchange(q,n*n*12);
    Vec rhs(q.size(),0);
    for(int e=0;e<local;e++) for(int j=0;j<n;j++) for(int i=0;i<n;i++) {
      State out{};
      for(int axis=0;axis<2;axis++) {
        int t=axis?j:i; double h=axis?hy:hx;
        for(int a=0;a<n;a++) {
          State f=flux(get(q,e,axis?a:j,axis?i:a,n),axis,p.c);
          for(int k=0;k<12;k++) out[k]-=2/h*D[t*n+a]*f[k];
        }
        if(t==0 || t==n-1) {
          int sign=t==0?-1:1;
          State own=get(q,e,j,i,n), other=sample(q,e,j+(axis?sign:0),i+(axis?0:sign),n);
          State star=sign>0?upwind(own,other,axis,p.c):upwind(other,own,axis,p.c);
          State f=flux(own,axis,p.c);
          for(int k=0;k<12;k++) out[k]-=2/h*sign*(star[k]-f[k])/weights[t];
        }
      }
      for(int k=0;k<12;k++) rhs[id(e,j,i,k,n)]=out[k];
    }
    Vec cq=project(q,n,nc,I);
    for(int e=0;e<local;e++) for(int j=0;j<nc;j++) for(int i=0;i<nc;i++) {
      State co=collision(get(cq,e,j,i,nc),p);
      for(int k=0;k<12;k++) cq[id(e,j,i,k,nc)]=co[k];
    }
    Vec co=project(cq,nc,n,P);
    for(size_t i=0;i<rhs.size();i++) rhs[i]+=co[i];
    addSources(q,rhs,false);
    return rhs;
  }
  void addSources(const Vec& q,Vec& rhs,bool fv) const {
    int m=fv?ns:n;
    Vec phi(size_t(total)*m*m,0), psi(phi.size(),0), rhoPsi(phi.size(),0);
    for(int e=0;e<local;e++) for(int j=0;j<m;j++) for(int i=0;i<m;i++) {
      size_t a=id(e,j,i,0,m,1); phi[a]=q[id(e,j,i,0,m)];
      psi[a]=p.cs(phi[a])-p.c*p.c*phi[a]; rhoPsi[a]=q[id(e,j,i,6,m)]-p.c*p.c*p.rho(phi[a]);
    }
    Vec px=derivative(psi,0,fv), py=derivative(psi,1,fv);
    Vec rx=derivative(rhoPsi,0,fv), ry=derivative(rhoPsi,1,fv,true);
    Vec lx(phi.size(),0), ly(phi.size(),0);
    if(p.kappa!=0) {
      Vec lap(phi.size(),0);
      if(fv) {
        exchange(phi,m*m);
        for(int e=0;e<local;e++) for(int j=0;j<m;j++) for(int i=0;i<m;i++) {
          size_t a=id(e,j,i,0,m,1);
          lap[a]=(scalarSample(phi,e,j,i+1,m)-2*phi[a]+scalarSample(phi,e,j,i-1,m))/std::pow(hx/m,2)
                +(scalarSample(phi,e,j+1,i,m)-2*phi[a]+scalarSample(phi,e,j-1,i,m))/std::pow(hy/m,2);
        }
      } else {
        lap=derivative(derivative(phi,0),0);
        Vec yy=derivative(derivative(phi,1),1);
        for(size_t a=0;a<lap.size();a++) lap[a]+=yy[a];
      }
      lx=derivative(lap,0,fv); ly=derivative(lap,1,fv);
    }
    for(int e=0;e<local;e++) for(int j=0;j<m;j++) for(int i=0;i<m;i++) {
      size_t a=id(e,j,i,0,m,1);
      State src=source(get(q,e,j,i,m),p,px[a],py[a],rx[a],ry[a],p.kappa*phi[a]*lx[a],p.kappa*phi[a]*ly[a]);
      for(int k=0;k<12;k++) rhs[id(e,j,i,k,m)]+=src[k];
    }
  }
  double blend(double v0,double v1,double b0,double b1,double d0) const {
    double tau=std::abs(b0-b1);
    double a0=d0*(1+std::pow(tau/(b0+p.wenoEps),2));
    double a1=(1-d0)*(1+std::pow(tau/(b1+p.wenoEps),2));
    return (a0*v0+a1*v1)/(a0+a1);
  }
  State wenoFace(const Vec& sub,int e,int j,int i,int axis) const {
    // Face immediately after (j,i); reconstruct incoming characteristic
    // states from subcell averages, then apply A+ q_L + A- q_R.
    if(p.characteristicWeno) {
      State w[4],f{};
      for(int t=0;t<4;t++)
        w[t]=characteristics(sample(sub,e,j+(axis?t-1:0),i+(axis?0:t-1),ns),axis);
      for(int s: {0,6}) for(int a=0;a<4;a++) {
        const int k=s+a;
        const double speed=p.c*(a<2?std::sqrt(3.0):1.0);
        if(a%2==0)
          f[k]=speed*blend(1.5*w[1][k]-.5*w[0][k],.5*(w[1][k]+w[2][k]),
                          std::pow(w[1][k]-w[0][k],2),std::pow(w[2][k]-w[1][k],2),1.0/3);
        else
          f[k]=-speed*blend(.5*(w[1][k]+w[2][k]),1.5*w[2][k]-.5*w[3][k],
                           std::pow(w[2][k]-w[1][k],2),std::pow(w[3][k]-w[2][k],2),2.0/3);
      }
      return moments(f,axis);
    }
    // Original component-wise LF flux splitting, retained for comparisons.
    State fp[4],fm[4];
    for(int t=0;t<4;t++) {
      State q=sample(sub,e,j+(axis?t-1:0),i+(axis?0:t-1),ns);
      State f=flux(q,axis,p.c);
      for(int k=0;k<12;k++) {fp[t][k]=.5*(f[k]+p.alpha*p.c*q[k]);fm[t][k]=.5*(f[k]-p.alpha*p.c*q[k]);}
    }
    State f{};
    for(int k=0;k<12;k++) {
      f[k]=blend(1.5*fp[1][k]-.5*fp[0][k],.5*(fp[1][k]+fp[2][k]),
                 std::pow(fp[1][k]-fp[0][k],2),std::pow(fp[2][k]-fp[1][k],2),1.0/3)
          +blend(.5*(fm[1][k]+fm[2][k]),1.5*fm[2][k]-.5*fm[3][k],
                 std::pow(fm[2][k]-fm[1][k],2),std::pow(fm[3][k]-fm[2][k],2),2.0/3);
    }
    return f;
  }
  // Shared host reference for conservative subcell index preparation.
  // Returns true only if all ranks succeed; leaves invalid-mean cells unchanged.
  bool scaleSubcellIndex(Vec& sub) const {
    bool bad=false;
    for(int e=0;e<local;e++) {
      State avg{}; double minPhi=std::numeric_limits<double>::max();
      for(int j=0;j<ns;j++) for(int i=0;i<ns;i++) {
        minPhi=std::min(minPhi,sub[id(e,j,i,0,ns)]);
        for(int k=0;k<6;k++) avg[k]+=sub[id(e,j,i,k,ns)]/(ns*ns);
      }
      if(minPhi<=p.floor()) {
        if(avg[0]<=p.floor()) {bad=true;continue;}
        double theta=(avg[0]-p.floor())/(avg[0]-minPhi)*(1-1e-12);
        for(int j=0;j<ns;j++) for(int i=0;i<ns;i++) for(int k=0;k<6;k++)
          sub[id(e,j,i,k,ns)]=avg[k]+theta*(sub[id(e,j,i,k,ns)]-avg[k]);
      }
    }
    return !any(bad);
  }
  Vec hybrid(Vec q,const Vec& dgRhs,Vec mask) const {
    exchange(q,n*n*12); exchange(mask,1);
    Vec sub=project(q,n,ns,A);
    if(!scaleSubcellIndex(sub)) throw SubcellMeanError();
    exchange(sub,ns*ns*12);
    Vec fv(sub.size(),0);
    // Face layout per element: x faces then y faces, each ns*(ns+1).
    std::vector<State> faces(size_t(local)*2*ns*(ns+1));
    auto face=[&](int e,int axis,int tangent,int normal)->State& {
      return faces[(size_t(e)*2+axis)*ns*(ns+1)+tangent*(ns+1)+normal];
    };
    for(int e=0;e<local;e++) {
      for(int axis=0;axis<2;axis++) for(int t=0;t<ns;t++) for(int a=0;a<=ns;a++)
        face(e,axis,t,a)=wenoFace(sub,e,axis?a-1:t,axis?t:a-1,axis);
      for(int j=0;j<ns;j++) for(int i=0;i<ns;i++) {
        State co=collision(get(sub,e,j,i,ns),p);
        for(int k=0;k<12;k++) fv[id(e,j,i,k,ns)]=co[k]
          -(face(e,0,j,i+1)[k]-face(e,0,j,i)[k])/(hx/ns)
          -(face(e,1,i,j+1)[k]-face(e,1,i,j)[k])/(hy/ns);
      }
    }
    addSources(sub,fv,true);
    Vec reconstructed=project(fv,ns,n,R,true), out=dgRhs;
    for(int e=0;e<local;e++) {
      if(mask[e]) {
        for(int j=0;j<n;j++) for(int i=0;i<n;i++) for(int k=0;k<12;k++) out[id(e,j,i,k,n)]=reconstructed[id(e,j,i,k,n)];
        continue;
      }
      for(int axis=0;axis<2;axis++) for(int sign: {-1,1}) {
        int nb=neighbor[e][4+sign*(axis?3:1)];
        if(nb<0 || !mask[nb]) continue;
        std::vector<State> nf(n); State avg{}, dgAvg{};
        for(int t=0;t<n;t++) for(int s=0;s<ns;s++) for(int k=0;k<12;k++)
          nf[t][k]+=R[t*ns+s]*face(e,axis,s,sign<0?0:ns)[k];
        for(int s=0;s<ns;s++) for(int k=0;k<12;k++) avg[k]+=face(e,axis,s,sign<0?0:ns)[k]/ns;
        for(int t=0;t<n;t++) for(int k=0;k<12;k++) dgAvg[k]+=.5*weights[t]*nf[t][k];
        int a=sign<0?0:n-1;
        for(int t=0;t<n;t++) {
          int j=axis?a:t, i=axis?t:a;
          State own=get(q,e,j,i,n), other=sample(q,e,j+(axis?sign:0),i+(axis?0:sign),n);
          State old=sign>0?upwind(own,other,axis,p.c):upwind(other,own,axis,p.c);
          for(int k=0;k<12;k++) out[id(e,j,i,k,n)]-=2/(axis?hy:hx)*sign/weights[a]*(nf[t][k]+avg[k]-dgAvg[k]-old[k]);
        }
      }
    }
    return out;
  }
  Vec interfaceMask(const Vec& q) const {
    Vec mask(total,0);
    if(!p.interfaceSubcells)return mask;
    const double low=p.phiL+p.interfaceThreshold*(p.phiH-p.phiL);
    const double high=p.phiH-p.interfaceThreshold*(p.phiH-p.phiL);
    for(int e=0;e<local;e++) {
      double lo=std::numeric_limits<double>::max(),hi=-lo;
      for(int a=0;a<n*n;a++) {
        const double phi=q[(size_t(e)*n*n+a)*12];
        lo=std::min(lo,phi);hi=std::max(hi,phi);
      }
      // Range overlap also catches endpoints on opposite sides of the band.
      mask[e]=(hi>low && lo<high);
    }
    return mask;
  }
  Vec detect(Vec reference,const Vec& candidate) const {
    exchange(reference,n*n*12);
    Vec mask(total,0), cq=project(candidate,n,nc,I);
    for(int e=0;e<local;e++) {
      double lo=std::numeric_limits<double>::max(),hi=-lo;
      for(int nb:neighbor[e]) if(nb>=0) for(int j=0;j<n;j++) for(int i=0;i<n;i++) {
        double phi=reference[id(nb,j,i,0,n)]; lo=std::min(lo,phi);hi=std::max(hi,phi);
      }
      double delta=std::max(p.dmpDelta*(p.phiH-p.phiL),p.dmpEps*(hi-lo));
      for(int j=0;j<n;j++) for(int i=0;i<n;i++) {
        State q=get(candidate,e,j,i,n); double phi=q[0];
        bool bad=!p.physical(phi) || phi<lo-delta || phi>hi+delta
          || phi<p.phiL-p.boundTol*(p.phiH-p.phiL) || phi>p.phiH+p.boundTol*(p.phiH-p.phiL);
        for(double a:q) bad|=!std::isfinite(a);
        if(bad) mask[e]=1;
      }
      for(int j=0;j<nc;j++) for(int i=0;i<nc;i++) {
        double rho=p.rho(cq[id(e,j,i,0,nc)]);
        if(!std::isfinite(rho) || rho<=0) mask[e]=1;
      }
    }
    return mask;
  }
  Vec expand(Vec mask) const {
    for(int layer=0;layer<p.buffer;layer++) {
      exchange(mask,1); Vec old=mask;
      for(int e=0;e<local;e++) for(int nb:neighbor[e]) if(nb>=0 && old[nb]) mask[e]=1;
    }
    return mask;
  }
  // Constant nodal diagonal split for SARK4. Keep the complete projected
  // collision in dg(); cancelling nodal Lambda*q here preserves that ODE.
  double sarkLambda(int k) const {
    return k>=1 && k<=5?-1/p.tauPhi:(k>=9 && k<=11?-1/p.tauG:0);
  }
  Vec sarkRhs(const Vec& q) const {
    Vec rhs=dg(q);
    for(int e=0;e<local;e++)for(int a=0;a<n*n;a++)for(int k=0;k<12;k++)
      if(sarkLambda(k)!=0)rhs[(size_t(e)*n*n+a)*12+k]-=sarkLambda(k)*q[(size_t(e)*n*n+a)*12+k];
    return rhs;
  }
  Vec stage(const Vec& q,const Vec& res,double dt,double a,double b) {
    const Vec base=dg(q);
    return stageWithRhs(q,res,dt,a,b,base,[&](const Vec& mask) {return hybrid(q,base,mask);});
  }
  // Share the existing detector/recompute policy with alternative spatial
  // backends. The callback must always start from the original DG RHS.
  Vec stageWithRhs(const Vec& q,const Vec& res,double dt,double a,double b,
                   const Vec& base,const std::function<Vec(const Vec&)>& hybridRhs,
                   const std::function<void(const Vec&)>& checkDetector={}) {
    Vec rhs=base;
    auto candidate=[&](const Vec& f) {
      Vec out=q;
      for(size_t i=0;i<size_t(local)*n*n*12;i++) out[i]+=b*(dt*f[i]+(a==0?0:a*res[i]));
      return out;
    };
    Vec trial=candidate(rhs);
    // Observe each raw mask before buffering or clearing already marked
    // elements. Verification must not alter the reference stage policy.
    auto detectCandidate=[&]() {
      Vec mask=detect(q,trial);
      if(checkDetector)checkDetector(mask);
      return mask;
    };
    lastTroubled=0;
    lastSubcellMask.assign(local,0);
    if(p.subcell) {
      candidateChecks++;
      Vec mask=detectCandidate();
      lastInterfaceMask=interfaceMask(q);
      for(int e=0;e<local;e++)mask[e]=std::max(mask[e],lastInterfaceMask[e]);
      mask=expand(mask);
      auto marked=[&](const Vec& v) {return std::any_of(v.begin(),v.begin()+local,[](double x){return x!=0;});};
      if(any(marked(mask))) {
        bool unresolved=false;
        for(int pass=0;pass<p.passes;pass++) {
          rhs=hybridRhs(mask); trial=candidate(rhs); recomputations++;
          maxRecomputePasses=std::max(maxRecomputePasses,pass+1);
          Vec next=detectCandidate();
          for(int e=0;e<local;e++) if(mask[e]) next[e]=0;
          unresolved=any(marked(next));
          if(!unresolved) break;
          if(pass+1<p.passes) {
            next=expand(next); for(int e=0;e<local;e++) mask[e]=std::max(mask[e],next[e]);
          }
        }
        if(unresolved) throw std::runtime_error("HCZ LSERK4 detector needs more recompute passes or a smaller CFL");
        for(int e=0;e<local;e++) if(mask[e]) lastTroubled++;
      }
      std::copy_n(mask.begin(),local,lastSubcellMask.begin());
    }
    if(!admissible(trial)) throw std::runtime_error("HCZ LSERK4 candidate remains nonphysical; reduce CFL");
    return rhs;
  }
};
} // namespace hcz
#endif
