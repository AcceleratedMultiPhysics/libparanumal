// Operator probe used by testHcz.py; no solver/OCCA/MPI dependencies.
#include "../solvers/bns/hczQuad2D.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc,char** argv) {
  if(argc!=3)return 2;
  std::ifstream f(argv[1]);std::ofstream out(argv[2]);
  int degree,nx,ny,periodic;double hx,hy;
  f>>degree>>nx>>ny>>periodic>>hx>>hy;
  hcz::Vec r(degree+1),w(degree+1),qr(degree+2),qw(degree+2);
  for(auto* v:{&r,&w,&qr,&qw})for(double& x:*v)f>>x;
  hcz::QuadOperator op(degree,nx*ny,nx*ny,hx,hy,r,w,qr,qw);
  auto& p=op.p;
  // The original Python reference uses component-wise LF splitting.
  p.characteristicWeno=false;
  f>>p.c>>p.tauPhi>>p.tauG>>p.rhoL>>p.rhoH>>p.phiL>>p.phiH>>p.eosB>>p.eosRatio>>p.gravity>>p.kappa;
  double dt,a,b;f>>dt>>a>>b;
  for(int e=0;e<nx*ny;e++)for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
    int x=(e%nx+dx+nx)%nx,y=e/nx+dy;if(periodic)y=(y+ny)%ny;
    op.neighbor[e][(dy+1)*3+dx+1]=y<0 || y>=ny?-1:x+y*nx;
  }
  hcz::Vec q(nx*ny*op.n*op.n*12),res(q.size()),mask(nx*ny);
  for(auto* v:{&q,&res,&mask})for(double& x:*v)f>>x;
  if(!f)return 3;
  out<<std::setprecision(17);
  auto write=[&](const hcz::Vec& v){for(double x:v)out<<x<<"\n";};
  try {
    auto dg=op.dg(q);write(dg);write(op.hybrid(q,dg,mask));
    write(op.project(q,op.n,op.ns,op.A));
    write(op.project(op.project(q,op.n,op.ns,op.A),op.ns,op.n,op.R,true));
    auto savedQ=q,savedRes=res;
    write(op.stage(q,res,dt,a,b));
    if(q!=savedQ || res!=savedRes)return 5;
  }catch(const std::exception& ex){std::cerr<<ex.what()<<"\n";return 4;}
}
