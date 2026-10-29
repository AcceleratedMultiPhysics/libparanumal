// Radial C5 circle and continuum mechanical pressure, used only at startup.
#ifndef BNS_HCZ_BUBBLE_HPP
#define BNS_HCZ_BUBBLE_HPP
#include <array>
#include <cmath>
#include <vector>
#include <algorithm>
namespace hcz {
inline double bubbleSmooth(double t) {
  t=std::max(0.,std::min(1.,t));
  const int choose[]={462,330,165,55,11,1};
  double s=0;for(int k=6;k<=11;k++)s+=choose[k-6]*std::pow(t,k)*std::pow(1-t,11-k);
  return s;
}
inline double bubbleTransitionSpan() {
  static const double span=[](){
    double a=0,b=.5;
    for(int i=0;i<60;i++){double t=.5*(a+b);if(bubbleSmooth(t)<.1)a=t;else b=t;}
    return 1-a-b;
  }();
  return span;
}
inline std::array<double,3> compactBubble(double r,double radius,double width,
                                         double low,double high,bool light) {
  const double length=width/bubbleTransitionSpan(),a=radius-.5*length;
  const double t=(r-a)/length,delta=(light?1:-1)*(high-low);
  const double inside=light?low:high;
  if(t<=0)return {inside,0,0};
  if(t>=1)return {inside+delta,0,0};
  const double d=2772*std::pow(t,5)*std::pow(1-t,5)/length;
  const double dd=13860*std::pow(t,4)*std::pow(1-t,4)*(1-2*t)/(length*length);
  return {inside+delta*bubbleSmooth(t),delta*d,delta*dd};
}
class BubblePressure {
  double radius=0,width=0,low=0,high=0,dr=0;
  bool light=false;
  std::vector<double> tail;
public:
  void setup(double R,double W,double lo,double hi,bool isLight) {
    radius=R;width=W;low=lo;high=hi;light=isLight;
    const int intervals=16384;
    dr=(radius+.5*width/bubbleTransitionSpan())/intervals;
    tail.assign(intervals+1,0);
    auto f=[&](double r){auto p=compactBubble(r,radius,width,low,high,light);return r>0?p[1]*p[1]/r:0.;};
    for(int i=intervals-1;i>=0;i--)
      tail[i]=tail[i+1]+dr*(f(i*dr)+4*f((i+.5)*dr)+f((i+1)*dr))/6;
  }
  double value(double r) const {
    if(tail.empty() || r>=dr*(tail.size()-1))return 0;
    const auto p=compactBubble(r,radius,width,low,high,light);
    const int i=std::min(int(r/dr),int(tail.size())-2);
    const double end=(i+1)*dr;
    auto f=[&](double s){auto a=compactBubble(s,radius,width,low,high,light);return s>0?a[1]*a[1]/s:0.;};
    const double integral=tail[i+1]+(end-r)*(f(r)+4*f(.5*(r+end))+f(end))/6;
    // p_r = kappa*phi*(lap phi)_r, with p_cap(outside)=0.
    return p[0]*(p[2]+(r>0?p[1]/r:0))-.5*p[1]*p[1]+integral;
  }
};
}
#endif
