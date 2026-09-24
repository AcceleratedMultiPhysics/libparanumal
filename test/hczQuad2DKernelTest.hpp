// Shared fixtures for isolated HCZ Quad2D OCCA transport tests.
#ifndef HCZ_QUAD_2D_KERNEL_TEST_HPP
#define HCZ_QUAD_2D_KERNEL_TEST_HPP
#include <occa.hpp>
#include "../solvers/bns/hczQuad2D.hpp"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <string>

namespace hczKernelTest {
using hcz::Vec;
using hcz::State;
constexpr int nxElements=3, nyElements=2, elements=nxElements*nyElements;
constexpr double hx=.31, hy=.47, soundSpeed=1.3;

void quadrature(int degree, Vec& r, Vec& w, Vec& qr, Vec& qw) {
  if(degree==1) {r={-1,1};w={1,1};}
  else if(degree==2) {r={-1,0,1};w={1./3,4./3,1./3};}
  else {r={-1,-1/std::sqrt(5.),1/std::sqrt(5.),1};w={1./6,5./6,5./6,1./6};}
  const int count=degree+2;
  qr.resize(count);qw.resize(count);
  for(int i=0;i<count;i++) {
    double x=std::cos(std::acos(-1.)*(i+.75)/(count+.5)), derivative=0;
    for(int it=0;it<50;it++) {
      double previous=1, current=x;
      for(int j=2;j<=count;j++) {
        double next=((2*j-1)*x*current-(j-1)*previous)/j;
        previous=current;current=next;
      }
      derivative=count*(x*current-previous)/(x*x-1);
      double dx=current/derivative;x-=dx;
      if(std::abs(dx)<1e-15)break;
    }
    qr[i]=x;qw[i]=2/((1-x*x)*derivative*derivative);
  }
}

struct Fixture {
  int n, np, fields;
  bool walls;
  hcz::QuadOperator cpu;
  Vec state, q, seed, geometry, coordinates;
  std::vector<int> mapM, mapP, boundaries;

  Fixture(int degree, int nfields, bool wallY,
          const Vec& r,const Vec& w,const Vec& qr,const Vec& qw):
    n(degree+1), np(n*n), fields(nfields), walls(wallY),
    cpu(degree,elements,elements,hx,hy,r,w,qr,qw),
    state(elements*np*12,0), q(2*elements*np*fields,0),
    seed(elements*np*fields,0), geometry(elements*4*n*4,0),
    coordinates(2*elements*np,0), mapM(elements*4*n),mapP(mapM.size()),boundaries(elements*4,0) {
    for(int e=0;e<elements;e++) {
      for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
        int x=(e%nxElements+dx+nxElements)%nxElements, y=e/nxElements+dy;
        if(!walls)y=(y+nyElements)%nyElements;
        cpu.neighbor[e][(dy+1)*3+dx+1]=(y<0 || y>=nyElements)?-1:x+nxElements*y;
      }
      for(int face=0;face<4;face++) for(int t=0;t<n;t++) {
        int i=face==1?n-1:face==3?0:t, j=face==0?0:face==2?n-1:t;
        int axis=(face==1 || face==3)?0:1, sign=(face==0 || face==3)?-1:1;
        int nb=cpu.neighbor[e][4+sign*(axis?3:1)], sk=(e*4+face)*n+t;
        mapM[sk]=e*np+j*n+i;
        if(nb<0) {
          boundaries[e*4+face]=1;
          // An unrelated valid exterior trace checks that the wall ghost is
          // constructed entirely from the interior six-moment distribution.
          mapP[sk]=((e+1)%elements)*np+j*n+i;
        } else {
          int pi=axis?i:(sign<0?n-1:0), pj=axis?(sign<0?n-1:0):j;
          // Some traces reside in a halo region beyond the owned elements.
          // Values are supplied here; this is not an MPI exchange test.
          int storedElement=nb+((face+e)%2?elements:0);
          mapP[sk]=storedElement*np+pj*n+pi;
        }
        geometry[4*sk+0]=axis?0:sign;
        geometry[4*sk+1]=axis?sign:0;
        geometry[4*sk+2]=(axis?hx:hy)/2;
        geometry[4*sk+3]=1/(hx*hy*.25*w[axis?j:i]);
      }
    }
  }

  void fill(int pattern) {
    for(int e=0;e<elements;e++) for(int j=0;j<n;j++) for(int i=0;i<n;i++) for(int k=0;k<12;k++) {
      double value=.1*(k+1);
      if(pattern!=0)value+=.027*std::sin(.73*(1+k+3*i+7*j+11*e));
      // HCZ pressure can be zero: dividing q_g by its first moment in the
      // legacy primitive boundary routine would corrupt the wall result.
      if(k==6)value=0;
      if(pattern==2 && k>=6)value=0; // index distribution only
      if(pattern==3 && k<6)value=0;  // hydrodynamic distribution only
      state[cpu.id(e,j,i,k,n)]=value;
    }
    for(int e=0;e<2*elements;e++) for(int a=0;a<np;a++) for(int k=0;k<fields;k++)
      q[(e*fields+k)*np+a]=state[((e%elements)*np+a)*12+k];
    for(size_t a=0;a<seed.size();a++)seed[a]=.01*std::cos(.13*(a+1));
  }

  Vec reference(const std::vector<int>& ids) const {
    Vec result(seed.size(),0);
    for(int e:ids) for(int j=0;j<n;j++) for(int i=0;i<n;i++) {
      State correction{};
      for(int axis=0;axis<2;axis++) {
        int t=axis?j:i;
        if(t!=0 && t!=n-1)continue;
        int sign=t==0?-1:1;
        State own=cpu.get(state,e,j,i,n);
        State other=cpu.sample(state,e,j+(axis?sign:0),i+(axis?0:sign),n);
        State numerical=sign>0?hcz::upwind(own,other,axis,soundSpeed):hcz::upwind(other,own,axis,soundSpeed);
        State physical=hcz::flux(own,axis,soundSpeed);
        for(int k=0;k<12;k++)correction[k]-=2/(axis?hy:hx)*sign*(numerical[k]-physical[k])/cpu.weights[t];
      }
      for(int k=0;k<fields;k++)result[(e*fields+k)*np+j*n+i]=correction[k];
    }
    return result;
  }

  void checkTransportConservation(const Vec& surface) const {
    // The strong-form surface correction by itself need not integrate to
    // zero. Add the CPU volume transport and check index mass conservation.
    double integral=0;
    for(int e=0;e<elements;e++) for(int j=0;j<n;j++) for(int i=0;i<n;i++) {
      double volume=0;
      for(int a=0;a<n;a++) {
        volume-=2/hx*cpu.D[i*n+a]*soundSpeed*state[cpu.id(e,j,a,1,n)];
        volume-=2/hy*cpu.D[j*n+a]*soundSpeed*state[cpu.id(e,a,i,2,n)];
      }
      integral+=(surface[(e*fields)*np+j*n+i]+volume)*hx*hy*.25*cpu.weights[i]*cpu.weights[j];
    }
    if(std::abs(integral)>2e-11)throw std::runtime_error("index transport does not conserve mass");
  }
};

occa::json kernelProperties(const std::string& root,int degree,int fields) {
  occa::json props;
  props["defines/dfloat"]="double";
  props["defines/dlong"]="int";
  props["defines/p_Nq"]=degree+1;
  props["defines/p_Np"]=(degree+1)*(degree+1);
  props["defines/p_Nfp"]=degree+1;
  props["defines/p_Nfaces"]=4;
  props["defines/p_Nfields"]=fields;
  props["defines/p_NblockS"]=4;
  props["defines/p_Nsgeo"]=4;
  props["defines/p_NXID"]=0;
  props["defines/p_NYID"]=1;
  props["defines/p_SJID"]=2;
  props["defines/p_WIJID"]=3;
  if(fields==12)props["defines/p_HCZ"]=1;
  else {
    // Leave p_HCZ undefined to exercise the ordinary BNS default build.
    props["defines/p_Npmlfields"]=2*fields;
    props["includes"].asArray();
    props["includes"]+=root+"/solvers/bns/data/bnsUniform2D.h";
  }
  return props;
}

} // namespace hczKernelTest
#endif
