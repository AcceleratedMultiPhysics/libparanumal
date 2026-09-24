// Actual libp OGS exchanges for HCZ detector bounds and multi-layer masks.
#include "../solvers/bns/hczDeviceMask.hpp"
#include "hczQuad2DKernelTest.hpp"
#include <map>
#include <set>
using namespace libp;

namespace {
struct Case {
  platform_t& platform;
  comm_t comm;
  int nx,ny,local,total;
  std::vector<int> gids;
  memory<dlong> ids,neighbors;
  deviceMemory<dlong> o_ids,o_neighbors;
  ogs::halo_t halo;
  kernel_t expandKernel;
  hcz::QuadOperator reference;
  int checks=0;

  static hcz::QuadOperator makeReference(int nx,int ny,int degree) {
    hcz::Vec r,w,qr,qw;hczKernelTest::quadrature(degree,r,w,qr,qw);
    return hcz::QuadOperator(degree,nx*ny,nx*ny,.31,.47,r,w,qr,qw);
  }

  Case(platform_t& p,const std::string& root,int x,int y,bool wall,int degree):
    platform(p),comm(p.comm),nx(x),ny(y),reference(makeReference(x,y,degree)) {
    const int px=comm.size()==4?2:comm.size(),py=comm.size()/px;
    auto owner=[&](int gid){return ((gid%nx)*px/nx)+px*((gid/nx)*py/ny);};
    for(int gid=0;gid<nx*ny;gid++) {
      if(owner(gid)==comm.rank())gids.push_back(gid);
      for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
        const int xx=(gid%nx+dx+nx)%nx;
        int yy=gid/nx+dy;if(!wall)yy=(yy+ny)%ny;
        reference.neighbor[gid][(dy+1)*3+dx+1]=yy<0 || yy>=ny?-1:xx+nx*yy;
      }
    }
    local=gids.size();
    std::map<int,int> slot;
    for(int e=0;e<local;e++)slot[gids[e]]=e;
    std::set<int> ghosts;
    for(int e=0;e<local;e++)for(int nb:reference.neighbor[gids[e]])
      if(nb>=0 && !slot.count(nb))ghosts.insert(nb);
    for(int gid:ghosts){slot[gid]=gids.size();gids.push_back(gid);}
    total=gids.size();ids.malloc(local);neighbors.malloc(9*local);
    memory<hlong> haloIds(total);
    for(int e=0;e<total;e++)haloIds[e]=(e<local?1LL:-1LL)*(gids[e]+1LL);
    for(int e=0;e<local;e++) {
      ids[e]=local-1-e; // Nontrivial selected-list order.
      for(int k=0;k<9;k++) {
        const int nb=reference.neighbor[gids[e]][k];
        neighbors[9*e+k]=nb<0?-1:slot.at(nb);
      }
    }
    halo.Setup(total,haloIds,comm,ogs::Pairwise,false,platform);
    o_ids=platform.malloc<dlong>(std::max(1,local));
    o_neighbors=platform.malloc<dlong>(std::max(1,local*9));
    if(local){o_ids.copyFrom(ids);o_neighbors.copyFrom(neighbors);}
    properties_t props=platform.props();
    expandKernel=platform.buildKernel(root+"/solvers/bns/okl/bnsHczExpandMaskQuad2D.okl",
                                      "bnsHczExpandMaskQuad2D",props);
  }

  void require(bool ok,const std::string& what) {
    int bad=ok?0:1,global=0;
    MPI_Allreduce(&bad,&global,1,MPI_INT,MPI_MAX,comm.comm());
    if(global)throw std::runtime_error(what);
  }

  void expand(const hcz::Vec& globalMask,int layers) {
    reference.p.buffer=layers;
    const auto expected=reference.expand(globalMask);
    memory<int> initial(std::max(1,total)+1,-79),actual(initial.length());
    for(int e=0;e<local;e++)initial[e]=int(globalMask[gids[e]]);
    auto o_mask=platform.malloc<int>(initial);
    memory<int> seed(std::max(1,local)+1,91),scratch(seed.length());
    auto o_scratch=platform.malloc<int>(seed);
    hczExpandDeviceMask(halo,expandKernel,local,o_ids,o_neighbors,o_mask,o_scratch,layers);
    o_mask.copyTo(actual);o_scratch.copyTo(scratch);
    bool ok=true;
    for(int e=0;e<local;e++)ok &= actual[e]==int(expected[gids[e]]);
    ok &= actual[actual.length()-1]==-79 && scratch[scratch.length()-1]==91;
    // Host expand leaves ghosts from the input to the last layer. A zero-
    // layer call must preserve all bytes, including deliberately stale halos.
    reference.p.buffer=std::max(0,layers-1);
    const auto previous=reference.expand(globalMask);
    for(int e=local;e<total;e++)ok &= actual[e]==(layers?int(previous[gids[e]]):-79);
    if(!layers)for(size_t e=0;e<seed.length();e++)ok &= scratch[e]==seed[e];
    require(ok,"buffered mask/halo/sentinel differs from host expand");
    checks++;
  }

  void masks() {
    for(int pattern=0;pattern<5;pattern++) {
      hcz::Vec mask(nx*ny,0);
      for(int gid=0;gid<nx*ny;gid++) {
        if(pattern==1)mask[gid]=1;
        if(pattern==2 && gid==0)mask[gid]=1;
        if(pattern==3 && gid==nx/2+nx*(ny/2))mask[gid]=1;
        if(pattern==4)mask[gid]=((gid%nx+2*(gid/nx))%7)==0;
      }
      for(int layers: {0,1,2,3})expand(mask,layers);
    }
    // The low-level layer kernel supports subsets. OGS first supplies the
    // actual old halo; selected output is checked without applying copyback.
    memory<int> old(std::max(1,total),0),after(std::max(1,total)+1,91);
    for(int e=0;e<local;e++)old[e]=gids[e]==0;
    auto o_old=platform.malloc<int>(old),o_after=platform.malloc<int>(after);
    halo.Exchange(o_old,1);o_old.copyTo(old);
    for(int count: {0,std::min(1,local),local/2,local}) {
      for(size_t i=0;i<after.length();i++)after[i]=91;
      o_after.copyFrom(after);
      // Avoid a zero grid launch on CUDA; the host helper uses this guard.
      if(count)expandKernel(count,o_ids,o_neighbors,o_old,o_after);
      o_after.copyTo(after);bool ok=after[after.length()-1]==91;
      for(int e=0;e<total;e++) {
        int expected=91;
        if(e>=local-count && e<local) {
          expected=old[e]!=0;
          for(int k=0;k<9;k++)if(neighbors[9*e+k]>=0 && old[neighbors[9*e+k]])expected=1;
        }
        ok &= after[e]==expected;
      }
      memory<int> unchanged(old.length());o_old.copyTo(unchanged);
      for(size_t e=0;e<old.length();e++)ok &= old[e]==unchanged[e];
      require(ok,"single buffer layer violated subset or snapshot contract");checks++;
    }
  }

  void detector(const std::string& root) {
    const int n=reference.n,np=n*n;
    properties_t props=platform.props();
    props["defines/p_Nq"]=n;props["defines/p_Np"]=np;
    props["defines/p_cubNq"]=reference.nc;props["defines/p_Nfields"]=12;
    const std::string file=root+"/solvers/bns/okl/bnsHczDetectorQuad2D.okl";
    auto boundsKernel=platform.buildKernel(file,"bnsHczIndexBoundsQuad2D",props);
    auto detectKernel=platform.buildKernel(file,"bnsHczDetectorQuad2D",props);
    memory<dfloat> interp(n*reference.nc);
    for(int i=0;i<n;i++)for(int j=0;j<reference.nc;j++)interp[i*reference.nc+j]=reference.I[j*n+i];
    auto o_interp=platform.malloc<dfloat>(interp);
    hcz::Vec global(nx*ny*np*12,0),trial;
    for(int e=0;e<nx*ny;e++)for(int node=0;node<np;node++)
      global[(e*np+node)*12]=.14+.025*std::sin(.4*e+.7*node);
    trial=global;
    // One nodal nonfinite stress guarantees a rejection on only its owner;
    // a second finite index excursion exercises DMP across rank boundaries.
    trial[((nx/2+nx*(ny/2))*np)*12+11]=std::numeric_limits<double>::quiet_NaN();
    trial[0]=.23;
    auto raw=reference.detect(global,trial);
    const dfloat poison=std::numeric_limits<dfloat>::quiet_NaN();
    memory<dfloat> q(std::max(1,local*np*12),poison),rhs(q.length(),0),res(q.length(),poison);
    for(int e=0;e<local;e++)for(int node=0;node<np;node++)for(int k=0;k<12;k++) {
      q[(e*12+k)*np+node]=global[(gids[e]*np+node)*12+k];
      rhs[(e*12+k)*np+node]=trial[(gids[e]*np+node)*12+k]-q[(e*12+k)*np+node];
    }
    auto o_q=platform.malloc<dfloat>(q),o_rhs=platform.malloc<dfloat>(rhs),o_res=platform.malloc<dfloat>(res);
    memory<dfloat> bounds(2*std::max(1,total),poison);
    auto o_bounds=platform.malloc<dfloat>(bounds);
    if(local)boundsKernel(local,o_ids,o_q,o_bounds);
    halo.ExchangeStart(o_bounds,2);halo.ExchangeFinish(o_bounds,2);
    o_bounds.copyTo(bounds);bool ok=true;
    for(int e=0;e<total;e++) {
      double lo=1,hi=0;
      for(int node=0;node<np;node++) {
        double phi=global[(gids[e]*np+node)*12];lo=std::min(lo,phi);hi=std::max(hi,phi);
      }
      ok &= bounds[2*e]==lo && bounds[2*e+1]==hi;
    }
    require(ok,"OGS detector bounds differ from global reference");
    memory<int> mask(std::max(1,total),-79);auto o_mask=platform.malloc<int>(mask);
    const auto& p=reference.p;
    if(local)detectKernel(local,o_ids,o_neighbors,o_bounds,o_interp,dfloat(1),dfloat(0),dfloat(1),
                         p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.floor(),p.dmpDelta,p.dmpEps,p.boundTol,
                         o_q,o_res,o_rhs,o_mask);
    o_mask.copyTo(mask);ok=true;
    for(int e=0;e<local;e++)ok &= mask[e]==int(raw[gids[e]]);
    for(int e=local;e<total;e++)ok &= mask[e]==-79;
    require(ok,"distributed raw detector differs from global host detector");checks++;
    auto o_scratch=platform.malloc<int>(std::max(1,local));
    // Feed the actual device detector output into the expansion helper.
    hczExpandDeviceMask(halo,expandKernel,local,o_ids,o_neighbors,o_mask,o_scratch,2);
    reference.p.buffer=2;auto buffered=reference.expand(raw);
    o_mask.copyTo(mask);ok=true;
    for(int e=0;e<local;e++)ok &= mask[e]==int(buffered[gids[e]]);
    require(ok,"distributed detector-to-buffer sequence differs from host");checks++;
  }
  void subcellSources(const std::string& root) {
    const int ns=reference.ns,np=ns*ns;
    auto props=platform.props();props["defines/p_Nfields"]=12;
    props["defines/p_Nq"]=reference.n;props["defines/p_Np"]=reference.n*reference.n;
    auto gradient=platform.buildKernel(root+"/solvers/bns/okl/bnsHczSubcellGradientQuad2D.okl",
                                       "bnsHczSubcellGradientQuad2D",props);
    props["defines/p_Nq"]=ns;props["defines/p_Np"]=np;props["defines/p_NblockV"]=std::max(1,256/np);
    auto thermo=platform.buildKernel(root+"/solvers/bns/okl/bnsHczThermodynamicsQuad2D.okl",
                                     "bnsHczThermodynamicsQuad2D",props);
    auto source=platform.buildKernel(root+"/solvers/bns/okl/bnsHczSourceQuad2D.okl",
                                     "bnsHczSourceQuad2D",props);
    auto& p=reference.p;p.c=1.3;p.kappa=0;p.gravity=reference.neighbor[0][1]<0?.04:0;
    hcz::Vec global(nx*ny*np*12,0),expected(global.size(),0);
    for(int e=0;e<nx*ny;e++)for(int j=0;j<ns;j++)for(int i=0;i<ns;i++) {
      for(int k=0;k<12;k++)global[reference.id(e,j,i,k,ns)]
        =k==0?.15+.02*std::sin(.3*e+.7*i+.4*j):(k==6?0:.01*std::sin(.2*k+.5*e+i+j));
    }
    reference.addSources(global,expected,true);
    memory<dfloat> q(std::max(1,local*np*12),0),rhs(q.length(),0);
    for(int e=0;e<local;e++)for(int node=0;node<np;node++)for(int k=0;k<12;k++)
      q[(e*12+k)*np+node]=global[(gids[e]*np+node)*12+k];
    memory<dfloat> potentials(std::max(1,total*2*np),-79),rho(std::max(1,local*np),0);
    auto o_q=platform.malloc<dfloat>(q),o_rhs=platform.malloc<dfloat>(rhs);
    auto o_p=platform.malloc<dfloat>(potentials),o_rho=platform.malloc<dfloat>(rho);
    auto o_g=platform.malloc<dfloat>(std::max(1,local*4*np));
    if(local)thermo(local,o_ids,p.c,p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.eosRatio,o_q,o_rho,o_p);
    halo.ExchangeStart(o_p,2*np);halo.ExchangeFinish(o_p,2*np);
    o_p.copyTo(potentials);bool ok=true;
    for(int e=0;e<total;e++)for(int node=0;node<np;node++) {
      double phi=global[(gids[e]*np+node)*12];
      double psi=p.cs(phi)-p.c*p.c*phi,psiRho=global[(gids[e]*np+node)*12+6]-p.c*p.c*p.rho(phi);
      ok &= std::isfinite(potentials[e*2*np+node]) && std::abs(potentials[e*2*np+node]-psi)<3e-12;
      ok &= std::isfinite(potentials[(e*2+1)*np+node]) && std::abs(potentials[(e*2+1)*np+node]-psiRho)<3e-12;
    }
    require(ok,"OGS subcell potential exchange differs from host");checks++;
    if(local)gradient(local,o_ids,o_neighbors,reference.hx,reference.hy,o_p,o_g);
    for(int repeat=1;repeat<=2;repeat++) {
      if(local)source(local,o_ids,p.c,p.gravity,p.rhoL,p.rhoH,p.phiL,p.phiH,o_q,o_g,o_rhs);
      o_rhs.copyTo(rhs);ok=true;
      for(int e=0;e<local;e++)for(int node=0;node<np;node++)for(int k=0;k<12;k++) {
        const double value=rhs[(e*12+k)*np+node],want=repeat*expected[(gids[e]*np+node)*12+k];
        ok &= std::isfinite(value) && std::abs(value-want)<3e-11;
        if(k==0 || k==7)ok &= value==0;
      }
      require(ok,"distributed subcell sources differ from global host reference");checks++;
    }
  }
};
} // namespace

int main(int argc,char** argv) {
  Comm::Init(argc,argv);
  if(argc!=4){Comm::Finalize();return 2;}
  try {
    comm_t comm(Comm::World().Dup());
    platformSettings_t settings(comm);
    settings.changeSetting("THREAD MODEL",argv[2]);
    settings.changeSetting("DEVICE NUMBER","0");settings.changeSetting("CACHE DIR",argv[3]);
    platform_t platform(settings);
    ogs::InitializeKernels(platform,ogs::Int32,ogs::Add);
    ogs::InitializeKernels(platform,ogs::Dfloat,ogs::Add);
    for(bool wall: {false,true})for(int degree: {1,2,3}) {
      Case test(platform,argv[1],8,8,wall,degree);
      test.masks();test.detector(argv[1]);test.subcellSources(argv[1]);
      if(comm.rank()==0)std::cout<<argv[2]<<" MPI"<<comm.size()<<" N="<<degree
        <<(wall?" walls":" periodic")<<": "<<test.checks<<" mask/source/halo checks passed\n";
    }
    // More than 256 elements exercises full and partial kernel blocks.
    Case large(platform,argv[1],34,18,true,2);large.masks();
    // Empty ranks, self-periodic maps and no wall neighbors on a one-cell box.
    Case tiny(platform,argv[1],1,1,true,1);tiny.masks();tiny.detector(argv[1]);tiny.subcellSources(argv[1]);
    if(comm.rank()==0)std::cout<<"HCZ mask/halo comparisons passed, including partial blocks and empty ranks.\n";
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<"\n";MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
  Comm::Finalize();return 0;
}
