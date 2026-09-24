// SARK4 relaxation and projected highest-mode accuracy using the actual stepper.
#include "timeStepper.hpp"
#include "hczQuad2DKernelTest.hpp"
using namespace libp;

struct Relaxation: solver_t {
  hcz::QuadOperator op;
  kernel_t split;
  double cap,atol,modeFactor;
  bool high,rejected=false,halfCollision=false;
  int rejectKind;
  double reportError=0;
  static hcz::QuadOperator reference() {
    hcz::Vec r,w,qr,qw;hczKernelTest::quadrature(2,r,w,qr,qw);
    return hcz::QuadOperator(2,1,1,1,1,r,w,qr,qw);
  }
  Relaxation(platform_t& p,settings_t& s,const std::string& root,double dt,bool mode,int reject=0,double tol=100):
    solver_t(p,s,p.comm),op(reference()),cap(dt),atol(tol),modeFactor(mode?.16:1),high(mode),rejectKind(reject) {
    op.p.tauPhi=.002;op.p.tauG=.003;
    auto props=p.props();props["defines/p_Np"]=9;
    split=p.buildKernel(root+"/solvers/bns/okl/bnsHczSarkSplitQuad2D.okl","bnsHczSarkSplitQuad2D",props);
  }
  hcz::Vec state(double time) {
    hcz::Vec q(9*12,0);
    for(int j=0;j<3;j++)for(int i=0;i<3;i++) {
      q[(3*j+i)*12]=.15;q[(3*j+i)*12+6]=1;
      double shape=high?.5*(3*op.nodes[i]*op.nodes[i]-1)*.5*(3*op.nodes[j]*op.nodes[j]-1):1;
      for(int k=0;k<12;k++)if(op.sarkLambda(k)!=0)
        q[(3*j+i)*12+k]=.01*shape*std::exp(modeFactor*op.sarkLambda(k)*time);
    }
    return q;
  }
  hcz::Vec unpack(deviceMemory<dfloat>& input) {
    memory<dfloat> native(108);input.copyTo(native);
    hcz::Vec q(108);for(int n=0;n<9;n++)for(int k=0;k<12;k++)q[12*n+k]=native[9*k+n];
    return q;
  }
  void pack(const hcz::Vec& q,deviceMemory<dfloat>& output) {
    memory<dfloat> native(108);for(int n=0;n<9;n++)for(int k=0;k<12;k++)native[9*k+n]=q[12*n+k];
    output.copyFrom(native);
  }
  void rhsf(deviceMemory<dfloat>& input,deviceMemory<dfloat>& output,const dfloat time) override {
    auto q=unpack(input),cq=op.project(q,3,4,op.I);
    for(int j=0;j<4;j++)for(int i=0;i<4;i++) {
      auto collision=hcz::collision(op.get(cq,0,j,i,4),op.p);
      for(int k=0;k<12;k++)cq[(4*j+i)*12+k]=collision[k];
    }
    auto full=op.project(cq,4,3,op.P);
    if(halfCollision)for(auto& value:full)value*=.5;
    pack(full,output);
    split(1,1/op.p.tauPhi,1/op.p.tauG,input,output);
  }
  dfloat sark4MaxTimeStep(const dfloat time) override {return cap;}
  void sark4Tolerances(dfloat& a,dfloat& r) override {a=atol;r=0;}
  bool sark4Admissible(deviceMemory<dfloat>& q,const dfloat time) override {
    // Deliberately reject one internal or output-sampling stage on one rank.
    // A globally agreed false must roll back without committing or reporting.
    const double target=rejectKind==1?.001:.0013;
    int bad=rejectKind && !rejected && comm.rank()==comm.size()-1 && std::abs(time-target)<1e-12;
    int global=0;MPI_Allreduce(&bad,&global,1,MPI_INT,MPI_MAX,comm.comm());
    if(global)rejected=true;
    return global==0;
  }
  deviceMemory<dfloat> solution;
  void Report(dfloat time,int tstep) override {
    const auto want=state(time),actual=unpack(solution);
    for(int i=0;i<108;i++)reportError=std::max(reportError,std::abs(want[i]-actual[i]));
  }
  double run() {
    memory<dfloat> lambda(12);for(int k=0;k<12;k++)lambda[k]=op.sarkLambda(k);
    timeStepper_t stepper;stepper.Setup<TimeStepper::sark4>(1,0,9,12,lambda,platform,comm);
    solution=platform.malloc<dfloat>(108);pack(state(0),solution);
    stepper.SetTimeStep(20*cap); // Also exercise the cap on the first step.
    stepper.Run(*this,solution,0,.0047); // Final time is not divisible by cap.
    auto actual=unpack(solution),want=state(.0047);double err=0;
    for(int i=0;i<108;i++)err=std::max(err,std::abs(actual[i]-want[i]));
    if(rejectKind && !rejected)throw std::runtime_error("rejection fixture was not exercised");
    return err;
  }
};

struct ReplayedRelaxation: Relaxation {
  using Relaxation::Relaxation;
  bool replayed=false;
  Sark4StageStatus sark4CheckStage(deviceMemory<dfloat>& q,const dfloat time,const int stage) override {
    // Change the spatial operator on one rank after old residuals have been
    // stored. A correct full replay solves the new ODE from the original q;
    // retaining even the first stale residual destroys this comparison.
    int request=!replayed && stage==2 && comm.rank()==comm.size()-1,global=0;
    MPI_Allreduce(&request,&global,1,MPI_INT,MPI_MAX,comm.comm());
    if(global) {
      replayed=true;halfCollision=true;modeFactor*=.5;
      return Sark4StageStatus::Restart;
    }
    return solver_t::sark4CheckStage(q,time,stage);
  }
};

struct RejectedRhsRelaxation: Relaxation {
  using Relaxation::Relaxation;
  double rejectTime=.001;
  bool rhsRejected=false;
  bool sark4Rhs(deviceMemory<dfloat>& q,deviceMemory<dfloat>& rhs,const dfloat time) override {
    // An unrecoverable subcell mean is discovered during RHS assembly.
    // Exercise the same collective failure hook, including output sampling.
    int bad=!rhsRejected && comm.rank()==comm.size()-1 && std::abs(time-rejectTime)<1e-12,global=0;
    MPI_Allreduce(&bad,&global,1,MPI_INT,MPI_MAX,comm.comm());
    if(global){rhsRejected=true;return false;}
    return solver_t::sark4Rhs(q,rhs,time);
  }
};

struct OutputStateRelaxation: Relaxation {
  using Relaxation::Relaxation;
  double stamp=0,savedStamp=0;
  int saves=0,restores=0;
  void sark4StartStep(deviceMemory<dfloat>&,const dfloat time,const dfloat dt) override {stamp=time+dt;}
  void sark4SaveOutputState() override {savedStamp=stamp;saves++;}
  void sark4RestoreOutputState() override {stamp=savedStamp;restores++;}
  void Report(dfloat time,int step) override {
    if(std::abs(stamp-time)>1e-12)throw std::runtime_error("wrong auxiliary state in SARK4 output");
    Relaxation::Report(time,step);
  }
};

int main(int argc,char** argv) {
  Comm::Init(argc,argv);
  if(argc!=4){Comm::Finalize();return 2;}
  try {
    comm_t comm(Comm::World().Dup());platformSettings_t settings(comm);
    settings.changeSetting("THREAD MODEL",argv[2]);settings.changeSetting("DEVICE NUMBER","0");
    settings.changeSetting("CACHE DIR",argv[3]);settings.newSetting("OUTPUT INTERVAL",".0013");
    platform_t platform(settings);
    for(int reject=0;reject<=2;reject++) {
      Relaxation test(platform,settings,argv[1],.002,false,reject);
      if(test.run()>2e-13 || test.reportError>2e-13)throw std::runtime_error("homogeneous relaxation/rollback failed");
    }
    double previous=0;
    for(double dt: {.001,.0005,.00025,.000125}) {
      Relaxation test(platform,settings,argv[1],dt,true);
      double err=test.run();
      if(!std::isfinite(err) || err==0 || (previous && previous/err<10))
        throw std::runtime_error("projected relaxation did not converge at fourth order");
      if(comm.rank()==0)std::cout<<"Projected relaxation dt="<<dt<<" error="<<err<<" ratio="<<(previous?previous/err:0)<<"\n";
      previous=err;
    }
    Relaxation adaptive(platform,settings,argv[1],.002,true,0,1e-11);
    if(adaptive.run()>2e-10 || adaptive.reportError>2e-10)throw std::runtime_error("adaptive projected relaxation failed");
    ReplayedRelaxation replay(platform,settings,argv[1],.00025,true);
    if(replay.run()>2e-9 || replay.reportError>2e-9 || !replay.replayed)
      throw std::runtime_error("SARK4 retained stale residual history after replay");
    for(double target: {.001,.0013}) {
      RejectedRhsRelaxation rhsFailure(platform,settings,argv[1],.002,false);
      rhsFailure.rejectTime=target;
      if(rhsFailure.run()>2e-13 || rhsFailure.reportError>2e-13 || !rhsFailure.rhsRejected)
        throw std::runtime_error("SARK4 failed to roll back after RHS rejection");
    }
    for(int reject: {0,2}) {
      OutputStateRelaxation output(platform,settings,argv[1],.002,false,reject);
      if(output.run()>2e-13 || output.reportError>2e-13 || output.saves==0 ||
         output.saves!=output.restores || std::abs(output.stamp-.0047)>1e-12)
        throw std::runtime_error("SARK4 output sampling did not restore accepted auxiliary state");
    }
    if(comm.rank()==0)std::cout<<argv[2]<<" MPI"<<comm.size()<<": SARK4 relaxation/accuracy/rejection checks passed\n";
  } catch(const std::exception& error){std::cerr<<error.what()<<"\n";MPI_Abort(MPI_COMM_WORLD,1);return 1;}
  Comm::Finalize();return 0;
}
