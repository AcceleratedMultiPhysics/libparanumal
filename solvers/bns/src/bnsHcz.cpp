#include "bns.hpp"
#include "hczQuad2D.hpp"
#include "hczDeviceMask.hpp"
#include "hczBubble.hpp"
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <climits>
#include <sstream>

// Retain the host reference, with an optional OCCA DG/subcell spatial path. Both use
// the HCZ Cartesian topology and the configured periodic/wall boundaries.
// OCCA physics uses native field-major storage and libp's OGS element halos.
class bnsHcz_t {
public:
  bns_t& b;
  std::unique_ptr<hcz::QuadOperator> op;
  int nx=0,ny=0,frame=0;
  double width=0,height=0,interfaceWidth=0,amplitude=0,initialMass=0,initialIndex=0;
  bool periodicY=false,periodicX=true;
  double bubbleRadius=0,bubbleX=0,bubbleY=0;
  bool lightBubble=false;
  bool compactCircle=false,mechanicalBubble=false;
  hcz::BubblePressure bubblePressure;
  std::string initial;
  std::string dgBackend;
  bool useOcca=false, verifyOcca=false;
  bool useSark=false;
  dfloat sarkDtMax=0,sarkAtol=0,sarkRtol=0;
  kernel_t sarkSplitKernel;
  kernel_t sarkDetectorKernel, maskClearKernel;
  deviceMemory<dfloat> o_sarkReference;
  hcz::Vec sarkReference,sarkMask;
  bool sarkHybrid=false;
  int sarkPass=0;
  deviceMemory<int> o_savedMask;
  hcz::Vec savedSarkMask;
  int savedTroubled=0,savedSarkPass=0;
  bool savedSarkHybrid=false;
  kernel_t interfaceMaskKernel;
  deviceMemory<int> o_interfaceMask,o_savedInterfaceMask;
  hcz::Vec savedInterfaceMask;
  long long interfaceChecks=0;
  long long sarkAttempts=0,sarkBudgetRejects=0,sarkMeanRejects=0;
  long long splitChecks=0;
  double maxSplitError=0,maxRelativeSplitError=0;
  long long rhsChecks=0;
  long long hybridChecks=0;
  long long detectorChecks=0;
  long long bufferChecks=0, maskUpdateChecks=0;
  bool deviceStageActive=false;
  long long stageDownloads=0,stageUploads=0;
  long long admissibilityChecks=0;
  memory<int> checkResult{1};
  double maxRhsError=0, maxRelativeRhsError=0;
  double maxHybridError=0, maxRelativeHybridError=0;
  ogs::halo_t elementHalo;
  deviceMemory<dlong> o_elementIds, o_vmapM, o_vmapP;
  deviceMemory<int> o_boundary;
  deviceMemory<int> o_checkFlags, o_checkResult;
  deviceMemory<dfloat> o_state, o_density, o_potentials, o_gradients;
  kernel_t thermodynamicsKernel, gradVolumeKernel, gradSurfaceKernel, sourceKernel;
  kernel_t admissibilityKernel, admissibilityReduceKernel;
  deviceMemory<dfloat> o_capillaryScalars, o_capillaryGradients;
  kernel_t capillaryPrepareKernel, capillarySourceKernel, capillaryDivergenceSurfaceKernel;
  deviceMemory<dfloat> o_subCapillaryStress, o_subCapillaryRhs;
  kernel_t subCapillaryStressKernel, subCapillarySourceKernel, subCapillaryAddKernel, capillaryReconstructKernel;
  deviceMemory<dlong> o_neighbors;
  deviceMemory<int> o_mask, o_nextMask, o_maskScratch;
  deviceMemory<dfloat> o_A, o_R, o_weights, o_baseRhs;
  deviceMemory<dfloat> o_sub, o_faces, o_subRhs, o_subDensity, o_subPotentials, o_subGradients;
  kernel_t projectSubcellKernel, scaleSubcellKernel, wenoFaceKernel, subcellRhsKernel;
  kernel_t subcellThermodynamicsKernel, subcellGradientKernel, subcellSourceKernel;
  kernel_t reconstructKernel, mixedSurfaceKernel;
  deviceMemory<dfloat> o_indexBounds;
  memory<int> detectorMask;
  kernel_t indexBoundsKernel, detectorKernel;
  kernel_t expandMaskKernel, maskDifferenceKernel, maskUnionKernel, maskCountKernel;
  std::vector<int> gids, sendElements, recvSlots, sendCounts, recvCounts, sendOffsets, recvOffsets;

  explicit bnsHcz_t(bns_t& solver):b(solver) {}
  double setting(const char* key) {double v; b.settings.getSetting(key,v);return v;}
  static std::vector<int> offsets(const std::vector<int>& counts) {
    std::vector<int> o(counts.size(),0); for(size_t i=1;i<counts.size();i++) o[i]=o[i-1]+counts[i-1];return o;
  }
  int neighborId(int gid,int dx,int dy) const {
    int x=gid%nx+dx,y=gid/nx+dy;
    if(periodicX)x=(x+nx)%nx;
    if(periodicY) y=(y+ny)%ny;
    return x<0 || x>=nx || y<0 || y>=ny?-1:x+y*nx;
  }
  void setup() {
    auto& m=b.mesh;
    std::string meshFile;
    m.settings.getSetting("MESH FILE",meshFile);
    LIBP_ABORT("HCZ supports only 2D Cartesian quadrilateral BOX meshes (no PML)",
      m.dim!=2 || m.elementType!=Mesh::QUADRILATERALS || meshFile!="BOX");
    useSark=b.settings.compareSetting("TIME INTEGRATOR","SARK4");
    LIBP_ABORT("HCZ requires TIME INTEGRATOR = LSERK4 or SARK4",
               !useSark && !b.settings.compareSetting("TIME INTEGRATOR","LSERK4"));
    LIBP_ABORT("HCZ requires polynomial degree >= 1",m.N<1);
    m.settings.getSetting("BOX GLOBAL NX",nx); m.settings.getSetting("BOX GLOBAL NY",ny);
    m.settings.getSetting("BOX DIMX",width); m.settings.getSetting("BOX DIMY",height);
    LIBP_ABORT("Invalid HCZ box dimensions",nx<1 || ny<1 || width<=0 || height<=0);
    periodicY=b.settings.compareSetting("HCZ Y BOUNDARY","PERIODIC");
    periodicX=b.settings.compareSetting("HCZ X BOUNDARY","PERIODIC");
    b.settings.getSetting("HCZ INITIAL CONDITION",initial);
    LIBP_ABORT("HCZ RT initialization requires y walls",initial=="RT" && periodicY);
    double hx=width/nx,hy=height/ny;
    int count=m.Nelements,size=b.comm.size();
    std::vector<int> counts(size),disp;
    MPI_Allgather(&count,1,MPI_INT,counts.data(),1,MPI_INT,b.comm.comm());disp=offsets(counts);
    LIBP_ABORT("HCZ BOX element count mismatch",disp.back()+counts.back()!=nx*ny);
    gids.resize(count);
    for(int e=0;e<count;e++) {
      double x=0,y=0;
      for(int v=0;v<4;v++) {x+=m.EX[4*e+v]/4;y+=m.EY[4*e+v]/4;}
      int ix=std::lround((x+width/2)/hx-.5),iy=std::lround((y+height/2)/hy-.5);
      LIBP_ABORT("HCZ invalid Cartesian element center",ix<0 || ix>=nx || iy<0 || iy>=ny);
      gids[e]=ix+iy*nx;
      for(int j=0;j<m.Nq;j++) for(int i=0;i<m.Nq;i++) {
        double expectedX=-width/2+(ix+.5+.5*m.gllz[i])*hx;
        double expectedY=-height/2+(iy+.5+.5*m.gllz[j])*hy;
        LIBP_ABORT("HCZ requires unwarped, axis-aligned Cartesian quads",
          std::abs(m.x[e*m.Np+j*m.Nq+i]-expectedX)>1e-9*width ||
          std::abs(m.y[e*m.Np+j*m.Nq+i]-expectedY)>1e-9*height);
      }
    }
    std::vector<int> all(nx*ny),owner(nx*ny,-1);
    MPI_Allgatherv(gids.data(),count,MPI_INT,all.data(),counts.data(),disp.data(),MPI_INT,b.comm.comm());
    for(int rank=0;rank<size;rank++) for(int e=0;e<counts[rank];e++) {
      int g=all[disp[rank]+e]; LIBP_ABORT("HCZ duplicate Cartesian element",g<0 || g>=nx*ny || owner[g]!=-1);owner[g]=rank;
    }
    std::map<int,int> slot;
    for(int e=0;e<count;e++) slot[gids[e]]=e;
    std::set<int> missing;
    for(int g:gids) for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
      int nb=neighborId(g,dx,dy);if(nb>=0 && !slot.count(nb)) missing.insert(nb);
    }
    // Requests grouped by owner. Owners receive global IDs and retain the
    // corresponding local element indices for every subsequent halo exchange.
    recvCounts.assign(size,0);
    std::vector<int> requests;
    for(int rank=0;rank<size;rank++) for(int g:missing) if(owner[g]==rank) {
      int s=slot.size();slot[g]=s;requests.push_back(g);recvSlots.push_back(s);recvCounts[rank]++;
    }
    recvOffsets=offsets(recvCounts); sendCounts.resize(size);
    MPI_Alltoall(recvCounts.data(),1,MPI_INT,sendCounts.data(),1,MPI_INT,b.comm.comm());
    sendOffsets=offsets(sendCounts); sendElements.resize(sendOffsets.back()+sendCounts.back());
    MPI_Alltoallv(requests.data(),recvCounts.data(),recvOffsets.data(),MPI_INT,
                  sendElements.data(),sendCounts.data(),sendOffsets.data(),MPI_INT,b.comm.comm());
    for(int& g:sendElements) g=slot.at(g);
    memory<dfloat> qr,qw;
    mesh_t::JacobiGQ(0,0,m.N+1,qr,qw);
    hcz::Vec r(m.Nq),w(m.Nq),gr(m.N+2),gw(m.N+2);
    for(int i=0;i<m.Nq;i++) {r[i]=m.gllz[i];w[i]=m.gllw[i];}
    for(int i=0;i<m.N+2;i++) {gr[i]=qr[i];gw[i]=qw[i];}
    op.reset(new hcz::QuadOperator(m.N,count,slot.size(),hx,hy,r,w,gr,gw));
    for(int e=0;e<count;e++) for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
      int nb=neighborId(gids[e],dx,dy);op->neighbor[e][(dy+1)*3+dx+1]=nb<0?-1:slot.at(nb);
    }
    op->exchange=[this](hcz::Vec& q,int stride) { exchange(q,stride); };
    op->any=[this](bool bad) { int in=bad,out=0;MPI_Allreduce(&in,&out,1,MPI_INT,MPI_MAX,b.comm.comm());return out!=0; };
    auto& p=op->p;
    p.c=setting("SPEED OF SOUND"); b.c=p.c;b.nu=setting("VISCOSITY");b.RT=p.c*p.c;
    p.tauG=b.nu/b.RT;p.tauPhi=setting("HCZ TAU PHI");if(p.tauPhi==0)p.tauPhi=p.tauG;
    p.rhoL=setting("HCZ RHO LIGHT");p.rhoH=setting("HCZ RHO HEAVY");
    p.phiL=setting("HCZ PHI LIGHT");p.phiH=setting("HCZ PHI HEAVY");
    p.eosB=setting("HCZ EOS B");p.eosRatio=setting("HCZ EOS A OVER B C2");
    p.gravity=setting("HCZ GRAVITY");p.kappa=setting("HCZ KAPPA");
    p.characteristicWeno=b.settings.compareSetting("HCZ WENO FLUX","CHARACTERISTIC");
    p.alpha=setting("HCZ WENO ALPHA");p.wenoEps=setting("HCZ WENO EPSILON");
    p.dmpDelta=setting("HCZ DMP DELTA");p.dmpEps=setting("HCZ DMP EPSILON");p.boundTol=setting("HCZ BOUND TOLERANCE");
    b.settings.getSetting("HCZ BUFFER LAYERS",p.buffer);b.settings.getSetting("HCZ RECOMPUTE PASSES",p.passes);
    p.subcell=b.settings.compareSetting("HCZ SUBCELL WENO","TRUE");
    p.interfaceSubcells=b.settings.compareSetting("HCZ INTERFACE SUBCELLS","TRUE");
    p.interfaceThreshold=setting("HCZ INTERFACE THRESHOLD");
    LIBP_ABORT("HCZ INTERFACE SUBCELLS requires HCZ SUBCELL WENO = TRUE",p.interfaceSubcells && !p.subcell);
    LIBP_ABORT("HCZ INTERFACE THRESHOLD must be finite and strictly between 0 and 0.5",
               !std::isfinite(p.interfaceThreshold) || p.interfaceThreshold<=0 || p.interfaceThreshold>=.5);
    interfaceWidth=setting("HCZ INTERFACE WIDTH");amplitude=setting("HCZ PERTURBATION");
    bubbleRadius=setting("HCZ BUBBLE RADIUS");
    bubbleX=setting("HCZ BUBBLE CENTER X");bubbleY=setting("HCZ BUBBLE CENTER Y");
    lightBubble=b.settings.compareSetting("HCZ BUBBLE PHASE","LIGHT");
    compactCircle=b.settings.compareSetting("HCZ BUBBLE PROFILE","COMPACT");
    mechanicalBubble=b.settings.compareSetting("HCZ BUBBLE PRESSURE","MECHANICAL");
    LIBP_ABORT("HCZ mechanical bubble pressure requires BUBBLE and COMPACT profile",
               mechanicalBubble && (initial!="BUBBLE" || !compactCircle));
    if(initial=="BUBBLE") {
      LIBP_ABORT("Invalid HCZ bubble geometry",!std::isfinite(bubbleRadius) || bubbleRadius<0 ||
        !std::isfinite(bubbleX) || !std::isfinite(bubbleY) ||
        (bubbleX<0 && bubbleX!=-1) || (bubbleY<0 && bubbleY!=-1));
      if(bubbleRadius==0)bubbleRadius=.2*std::min(width,height);
      if(bubbleX==-1)bubbleX=width/2;
      if(bubbleY==-1)bubbleY=height/2;
      LIBP_ABORT("HCZ bubble circle must fit strictly inside the box",bubbleX-bubbleRadius<=0 ||
        bubbleX+bubbleRadius>=width || bubbleY-bubbleRadius<=0 || bubbleY+bubbleRadius>=height);
      if(compactCircle) {
        const double half=.5*interfaceWidth/hcz::bubbleTransitionSpan();
        LIBP_ABORT("HCZ compact bubble transition must fit between the origin and walls",
          !std::isfinite(half) || half<=0 || half>=bubbleRadius ||
          bubbleX-bubbleRadius-half<=0 || bubbleX+bubbleRadius+half>=width ||
          bubbleY-bubbleRadius-half<=0 || bubbleY+bubbleRadius+half>=height);
      }
      if(mechanicalBubble)bubblePressure.setup(bubbleRadius,interfaceWidth,p.phiL,p.phiH,lightBubble);
    }
    bool finite=true;
    for(double v:{p.c,b.nu,p.tauPhi,p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.eosRatio,p.gravity,p.kappa,
        p.alpha,p.wenoEps,p.dmpDelta,p.dmpEps,p.boundTol,interfaceWidth,amplitude}) finite &= std::isfinite(v);
    LIBP_ABORT("Invalid HCZ physical/numerical parameters",!finite || p.c<=0 || b.nu<=0 || p.tauPhi<=0 ||
      p.rhoL<=0 || p.rhoH<=p.rhoL || p.phiL<=0 || p.phiH<=p.phiL || p.eosB<=0 ||
      p.eosB*p.phiH>=3.999 || p.eosRatio<=0 || p.gravity<0 || p.kappa<0 || interfaceWidth<=0 ||
      p.alpha<std::sqrt(3.0) || p.wenoEps<=0 || p.dmpDelta<0 || p.dmpEps<0 || p.boundTol<0 || p.buffer<0 || p.passes<1);
    LIBP_ABORT("Periodic HCZ y boundaries require zero gravity",periodicY && p.gravity!=0);
    b.settings.getSetting("HCZ DG BACKEND",dgBackend);
    useOcca=dgBackend!="HOST";verifyOcca=dgBackend=="VERIFY";
    LIBP_ABORT("HCZ SARK4 requires HCZ KAPPA = 0",useSark && p.kappa!=0);
    b.tauInv=1/p.tauG;b.Nfields=12;b.Npmlfields=0;b.semiAnalytic=useSark;
    dlong entries=count*m.Np*12;
    b.q.malloc(entries);b.o_q=b.platform.malloc<dfloat>(entries);
    if(useSark) {
      sarkAtol=setting("HCZ SARK4 ATOL");sarkRtol=setting("HCZ SARK4 RTOL");
      LIBP_ABORT("Invalid HCZ SARK4 tolerances",!std::isfinite(sarkAtol) || sarkAtol<=0 ||
                 !std::isfinite(sarkRtol) || sarkRtol<0);
      memory<dfloat> lambda(12);
      for(int k=0;k<12;k++)lambda[k]=op->sarkLambda(k);
      b.timeStepper.Setup<TimeStepper::sark4>(count,0,m.Np,12,lambda,b.platform,b.comm);
    } else b.timeStepper.Setup<TimeStepper::lserk4>(count,0,m.Np,12,b.platform,b.comm);
    if(useOcca) setupOcca(slot);
    if(b.comm.rank()==0) std::cout << "HCZ: 12 moments, 2D Cartesian quads, "
      <<(useSark?"SARK4":"LSERK4")<<", no PML.\n"
      << "HCZ spatial backend: "<<dgBackend
      << (useOcca?(p.subcell?"; OCCA DG/subcell kernels and OGS halos; device detector, masks and recomputation.\n"
                           :"; OCCA DG kernels, OGS element halos, device admissibility checks.\n")
                 :"; host reference implementation, MPI element halos.\n");
  }

  void setupOcca(const std::map<int,int>& slot) {
    auto& m=b.mesh;
    m.CubatureSetup();
    ogs::InitializeKernels(b.platform,ogs::Dfloat,ogs::Add);
    // One signed global element ID per owned/ghost slot. Exchange k=Np*Nfields
    // entries at a time, preserving native [element][field][node] storage.
    // Reuse the reference topology, including its corner ghosts for now.
    memory<hlong> haloIds(op->total);
    for(const auto& entry:slot)
      haloIds[entry.second]=(entry.second<op->local?1LL:-1LL)*(static_cast<hlong>(entry.first)+1);
    elementHalo.Setup(op->total,haloIds,b.comm,ogs::Pairwise,false,b.platform);

    memory<dlong> ids(op->local),mapM(op->local*m.Nfaces*m.Nfp),mapP(mapM.length());
    memory<int> boundary(op->local*m.Nfaces,0);
    for(int e=0;e<op->local;e++) {
      ids[e]=e;
      for(int face=0;face<4;face++) {
        const int axis=(face==1 || face==3)?0:1;
        const int sign=(face==0 || face==3)?-1:1;
        const int nb=op->neighbor[e][4+sign*(axis?3:1)];
        boundary[e*4+face]=nb<0?(axis?1:2):0;
        for(int t=0;t<m.Nfp;t++) {
          const int i=axis?t:(sign<0?0:m.Nq-1);
          const int j=axis?(sign<0?0:m.Nq-1):t;
          const int node=j*m.Nq+i;
          LIBP_ABORT("HCZ Quad2D face ordering mismatch",m.faceNodes[face*m.Nfp+t]!=node);
          const int opposite=axis?(sign<0?m.Nq-1:0)*m.Nq+i:j*m.Nq+(sign<0?m.Nq-1:0);
          const dlong sk=(e*4+face)*m.Nfp+t;
          mapM[sk]=e*m.Np+node;
          // Transport reads an exterior state before constructing the wall
          // ghost; use the valid interior node at walls.
          mapP[sk]=nb<0?mapM[sk]:nb*m.Np+opposite;
        }
      }
    }
    o_elementIds=b.platform.malloc<dlong>(ids);
    o_vmapM=b.platform.malloc<dlong>(mapM);o_vmapP=b.platform.malloc<dlong>(mapP);
    o_boundary=b.platform.malloc<int>(boundary);
    o_state=b.platform.malloc<dfloat>(static_cast<dlong>(op->total)*m.Np*12);
    o_density=b.platform.malloc<dfloat>(static_cast<dlong>(op->total)*m.Np);
    o_potentials=b.platform.malloc<dfloat>(static_cast<dlong>(op->total)*m.Np*2);
    o_gradients=b.platform.malloc<dfloat>(static_cast<dlong>(op->local)*m.Np*4);
    o_checkFlags=b.platform.malloc<int>(std::max(1,op->local));
    o_checkResult=b.platform.malloc<int>(1);

    properties_t props=m.props;
    props["defines/p_HCZ"]=1;props["defines/p_Nfields"]=12;
    // Twelve fields double the BNS shared storage; keep the common 256-node
    // budget on CUDA too instead of the six-field solver's 512-node budget.
    const int blockMax=256;
    props["defines/p_NblockV"]=std::max(1,blockMax/m.Np);
    props["defines/p_NblockS"]=std::max(1,blockMax/std::max(m.Np,m.Nfp*m.Nfaces));
    props["defines/p_NblockCub"]=std::max(1,blockMax/m.cubNp);
    auto build=[&](const std::string& name) {
      return b.platform.buildKernel(std::string(DBNS)+"okl/"+name+".okl",name,props);
    };
    b.volumeKernel=build("bnsVolumeQuad2D");
    b.surfaceKernel=build("bnsSurfaceQuad2D");
    b.relaxationKernel=build("bnsRelaxationQuad2D");
    thermodynamicsKernel=build("bnsHczThermodynamicsQuad2D");
    gradVolumeKernel=build("bnsHczGradVolumeQuad2D");
    gradSurfaceKernel=build("bnsHczGradSurfaceQuad2D");
    sourceKernel=build("bnsHczSourceQuad2D");
    if(useSark)sarkSplitKernel=build("bnsHczSarkSplitQuad2D");
    admissibilityKernel=build("bnsHczAdmissibilityQuad2D");
    admissibilityReduceKernel=b.platform.buildKernel(std::string(DBNS)+"okl/bnsHczAdmissibilityQuad2D.okl",
                                                    "bnsHczAdmissibilityReduce",props);
    if(op->p.kappa>0 && !op->p.subcell) {
      o_capillaryScalars=b.platform.malloc<dfloat>(static_cast<dlong>(op->total)*m.Np*2);
      o_capillaryGradients=b.platform.malloc<dfloat>(static_cast<dlong>(op->local)*m.Np*4);
      const std::string file=std::string(DBNS)+"okl/bnsHczCapillaryQuad2D.okl";
      capillaryPrepareKernel=b.platform.buildKernel(file,"bnsHczCapillaryPrepareQuad2D",props);
      capillarySourceKernel=b.platform.buildKernel(file,"bnsHczCapillarySourceQuad2D",props);
      properties_t divProps=props;divProps["defines/p_HCZ_NEUMANN_FLUX"]=1;
      capillaryDivergenceSurfaceKernel=b.platform.buildKernel(std::string(DBNS)+"okl/bnsHczGradSurfaceQuad2D.okl",
                                                            "bnsHczGradSurfaceQuad2D",divProps);
    }
    if(op->p.subcell) setupOccaSubcell(props);
  }

  void setupOccaSubcell(properties_t props) {
    ogs::InitializeKernels(b.platform,ogs::Int32,ogs::Add);
    const int n=op->n,ns=op->ns,np=ns*ns;
    const dlong count=op->local,total=op->total;
    memory<dlong> neighbors(count*9);
    for(int e=0;e<count;e++)for(int k=0;k<9;k++)neighbors[e*9+k]=op->neighbor[e][k];
    memory<dfloat> A(n*ns),R(n*ns),weights(n);
    for(int i=0;i<n;i++) {
      weights[i]=op->weights[i];
      for(int s=0;s<ns;s++) {A[s+i*ns]=op->A[s*n+i];R[i+s*n]=op->R[i*ns+s];}
    }
    o_neighbors=b.platform.malloc<dlong>(neighbors);
    o_mask=b.platform.malloc<int>(std::max(dlong(1),total));
    o_nextMask=b.platform.malloc<int>(std::max(dlong(1),total));
    o_maskScratch=b.platform.malloc<int>(std::max(dlong(1),count));
    o_A=b.platform.malloc<dfloat>(A);o_R=b.platform.malloc<dfloat>(R);
    o_weights=b.platform.malloc<dfloat>(weights);
    o_baseRhs=b.platform.malloc<dfloat>(count*n*n*12);
    o_sub=b.platform.malloc<dfloat>(total*np*12);
    o_faces=b.platform.malloc<dfloat>(count*2*12*ns*(ns+1));
    o_subRhs=b.platform.malloc<dfloat>(count*np*12);
    o_subDensity=b.platform.malloc<dfloat>(count*np);
    o_subPotentials=b.platform.malloc<dfloat>(total*np*2);
    o_subGradients=b.platform.malloc<dfloat>(count*np*4);
    auto build=[&](const std::string& file,const std::string& name) {
      return b.platform.buildKernel(std::string(DBNS)+"okl/"+file+".okl",name,props);
    };
    projectSubcellKernel=build("bnsHczSubcellQuad2D","bnsHczProjectSubcellQuad2D");
    scaleSubcellKernel=build("bnsHczSubcellQuad2D","bnsHczScaleSubcellQuad2D");
    props["defines/p_HCZ_CHARACTERISTIC_WENO"]=op->p.characteristicWeno?1:0;
    wenoFaceKernel=build("bnsHczWenoFaceQuad2D","bnsHczWenoFaceQuad2D");
    subcellRhsKernel=build("bnsHczSubcellRhsQuad2D","bnsHczSubcellRhsQuad2D");
    subcellGradientKernel=build("bnsHczSubcellGradientQuad2D","bnsHczSubcellGradientQuad2D");
    props["defines/p_HCZ_MASKED_RECONSTRUCT"]=1;
    reconstructKernel=build("bnsHczReconstructQuad2D","bnsHczReconstructQuad2D");
    props["defines/p_HCZ_MASKED_RECONSTRUCT"]=0;
    mixedSurfaceKernel=build("bnsHczMixedSurfaceQuad2D","bnsHczMixedSurfaceQuad2D");
    if(op->p.kappa>0) {
      o_subCapillaryStress=b.platform.malloc<dfloat>(total*np*3);
      o_subCapillaryRhs=b.platform.malloc<dfloat>(count*np*12);
      const std::string file="bnsHczCapillarySubcellQuad2D";
      subCapillaryStressKernel=build(file,"bnsHczCapillaryStressQuad2D");
      subCapillarySourceKernel=build(file,"bnsHczCapillarySubcellSourceQuad2D");
      subCapillaryAddKernel=build(file,"bnsHczCapillarySubcellAddQuad2D");
      props["defines/p_HCZ_ADD_RECONSTRUCT"]=1;
      capillaryReconstructKernel=build("bnsHczReconstructQuad2D","bnsHczReconstructQuad2D");
      props["defines/p_HCZ_ADD_RECONSTRUCT"]=0;
    }
    o_indexBounds=b.platform.malloc<dfloat>(std::max(dlong(1),2*total));
    if(verifyOcca)detectorMask.malloc(count);
    indexBoundsKernel=build("bnsHczDetectorQuad2D","bnsHczIndexBoundsQuad2D");
    if(op->p.interfaceSubcells) {
      interfaceMaskKernel=build("bnsHczDetectorQuad2D","bnsHczInterfaceMaskQuad2D");
      o_interfaceMask=b.platform.malloc<int>(std::max(dlong(1),count));
      if(useSark)o_savedInterfaceMask=b.platform.malloc<int>(std::max(dlong(1),count));
    }
    detectorKernel=build("bnsHczDetectorQuad2D","bnsHczDetectorQuad2D");
    expandMaskKernel=build("bnsHczExpandMaskQuad2D","bnsHczExpandMaskQuad2D");
    maskDifferenceKernel=build("bnsHczMaskControlQuad2D","bnsHczMaskDifferenceQuad2D");
    maskUnionKernel=build("bnsHczMaskControlQuad2D","bnsHczMaskUnionQuad2D");
    maskCountKernel=build("bnsHczMaskControlQuad2D","bnsHczMaskCountQuad2D");
    if(useSark) {
      o_savedMask=b.platform.malloc<int>(std::max(dlong(1),count));
      maskClearKernel=build("bnsHczMaskControlQuad2D","bnsHczMaskClearQuad2D");
      props["defines/p_HCZ_DETECT_STATE"]=1;
      sarkDetectorKernel=build("bnsHczDetectorQuad2D","bnsHczDetectorQuad2D");
      props["defines/p_HCZ_DETECT_STATE"]=0;
    }
    // Pointwise physics is shared with DG, specialized to the subcell grid.
    props["defines/p_Nq"]=ns;props["defines/p_Np"]=np;
    props["defines/p_NblockV"]=std::max(1,256/np);
    subcellThermodynamicsKernel=build("bnsHczThermodynamicsQuad2D","bnsHczThermodynamicsQuad2D");
    subcellSourceKernel=build("bnsHczSourceQuad2D","bnsHczSourceQuad2D");
  }

  bool deviceAdmissible(deviceMemory<dfloat>& input,deviceMemory<dfloat>& residual,
                        deviceMemory<dfloat>& rhs,bool candidate,dfloat dt,dfloat a,dfloat rkB) {
    const auto& p=op->p;
    if(op->local)
      admissibilityKernel(op->local,o_elementIds,b.mesh.o_cubInterp,int(candidate),dt,a,rkB,
                          p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.floor(),input,residual,rhs,o_checkFlags);
    admissibilityReduceKernel(op->local,o_checkFlags,o_checkResult);
    int global=0;
    o_checkResult.copyTo(checkResult);
    MPI_Allreduce(checkResult.ptr(),&global,1,MPI_INT,MPI_BOR,b.comm.comm());
    const bool valid=global==0;
    if(verifyOcca) {
      auto state=unpack(input);
      if(candidate) {
        const auto force=unpack(rhs);
        hcz::Vec res;
        if(a!=0)res=unpack(residual);
        for(size_t i=0;i<size_t(op->local)*b.mesh.Np*12;i++)
          state[i]+=rkB*(dt*force[i]+(a==0?0:a*res[i]));
      }
      const bool reference=op->admissible(state);
      admissibilityChecks++;
      if(valid!=reference)
        throw std::runtime_error("HCZ device admissibility differs from host reference");
    }
    return valid;
  }

  std::string admissibilitySummary(const hcz::Vec& q) const {
    // Failure-only diagnostics. Keep successful stages entirely on device.
    const auto& p=op->p;
    double range[4]={INFINITY,INFINITY,INFINITY,INFINITY},global[4];
    long long bad[2]={},globalBad[2];
    const auto cub=op->project(q,op->n,op->nc,op->I);
    for(int e=0;e<op->local;e++) {
      for(int j=0;j<op->n;j++)for(int i=0;i<op->n;i++) {
        const auto s=op->get(q,e,j,i,op->n);
        bool invalid=!p.physical(s[0]);
        for(double v:s)invalid|=!std::isfinite(v);
        bad[0]+=invalid;
        if(std::isfinite(s[0])) {
          range[0]=std::min(range[0],s[0]);range[1]=std::min(range[1],-s[0]);
          range[2]=std::min(range[2],p.rho(s[0]));
        }
      }
      for(int j=0;j<op->nc;j++)for(int i=0;i<op->nc;i++) {
        const auto s=op->get(cub,e,j,i,op->nc);
        bool invalid=!(p.rho(s[0])>0);
        for(double v:s)invalid|=!std::isfinite(v);
        bad[1]+=invalid;
        if(std::isfinite(s[0]))range[3]=std::min(range[3],p.rho(s[0]));
      }
    }
    MPI_Allreduce(range,global,4,MPI_DOUBLE,MPI_MIN,b.comm.comm());
    MPI_Allreduce(bad,globalBad,2,MPI_LONG_LONG_INT,MPI_SUM,b.comm.comm());
    std::ostringstream out;
    out<<std::setprecision(12)<<"nodal phi=["<<global[0]<<","<<-global[1]
       <<"], min nodal rho="<<global[2]<<", min cubature rho="<<global[3]
       <<", invalid nodes="<<globalBad[0]<<", invalid cubature points="<<globalBad[1];
    return out.str();
  }

  std::string lserkCandidateFailure(deviceMemory<dfloat>& input,deviceMemory<dfloat>& residual,
                                   deviceMemory<dfloat>& rhs,dfloat dt,dfloat a,dfloat rkB) {
    auto state=unpack(input),candidate=state;
    const auto force=unpack(rhs);
    hcz::Vec res;if(a!=0)res=unpack(residual);
    for(size_t i=0;i<size_t(op->local)*b.mesh.Np*12;i++)
      candidate[i]+=rkB*(dt*force[i]+(a==0?0:a*res[i]));
    std::ostringstream out;
    out<<std::setprecision(12)<<"HCZ LSERK4 candidate remains nonphysical; dt="<<dt
       <<", index floor="<<op->p.floor()<<"\n"
       <<"    Input: "<<admissibilitySummary(state)<<"\n"
       <<"    Candidate: "<<admissibilitySummary(candidate)<<"\n"
       <<"    Compare smaller CFL and spatial/interface resolution; reducing CFL alone may not resolve this failure.";
    return out.str();
  }

  void occaCapillarySource(deviceMemory<dfloat>& output) {
    auto& m=b.mesh;const auto& p=op->p;const dlong count=op->local;
    for(int pass=0;pass<3;pass++) {
      if(count)capillaryPrepareKernel(count,o_elementIds,pass,
                                     pass==0?o_state:o_capillaryGradients,o_capillaryScalars);
      elementHalo.ExchangeStart(o_capillaryScalars,2*m.Np);
      if(count)gradVolumeKernel(count,o_elementIds,m.o_vgeo,m.o_D,o_capillaryScalars,o_capillaryGradients);
      elementHalo.ExchangeFinish(o_capillaryScalars,2*m.Np);
      if(count) {
        auto& surface=pass==1?capillaryDivergenceSurfaceKernel:gradSurfaceKernel;
        surface(count,o_elementIds,m.o_sgeo,o_vmapM,o_vmapP,o_boundary,o_capillaryScalars,o_capillaryGradients);
      }
    }
    if(count)capillarySourceKernel(count,o_elementIds,p.c,p.kappa,p.rhoL,p.rhoH,p.phiL,p.phiH,
                                  o_state,o_capillaryGradients,output);
  }

  void occaHybridCapillarySource(deviceMemory<dfloat>& output) {
    const auto& p=op->p;const dlong count=op->local;const int np=op->ns*op->ns;
    if(count) {
      projectSubcellKernel(count,o_elementIds,o_A,o_state,o_sub);
      scaleSubcellKernel(count,o_elementIds,p.floor(),o_sub,o_checkFlags);
    }
    admissibilityReduceKernel(count,o_checkFlags,o_checkResult);
    o_checkResult.copyTo(checkResult);
    int invalid=0;MPI_Allreduce(checkResult.ptr(),&invalid,1,MPI_INT,MPI_BOR,b.comm.comm());
    if(invalid)throw hcz::SubcellMeanError();
    elementHalo.ExchangeStart(o_sub,12*np);elementHalo.ExchangeFinish(o_sub,12*np);
    if(count)subCapillaryStressKernel(count,o_elementIds,o_neighbors,op->hx,op->hy,p.kappa,o_sub,o_subCapillaryStress);
    elementHalo.ExchangeStart(o_subCapillaryStress,3*np);elementHalo.ExchangeFinish(o_subCapillaryStress,3*np);
    if(count) {
      subCapillarySourceKernel(count,o_elementIds,o_neighbors,op->hx,op->hy,p.c,p.rhoL,p.rhoH,p.phiL,p.phiH,
                               o_sub,o_subCapillaryStress,o_subCapillaryRhs);
      capillaryReconstructKernel(count,o_elementIds,o_R,o_weights,o_subCapillaryRhs,output);
    }
  }

  void occaRhs(deviceMemory<dfloat>& input,deviceMemory<dfloat>& output,
               dfloat time) {
    // In input mode neither residual nor RHS is read; valid dummy buffers
    // let this use the same read-only checker as a prospective RK candidate.
    if(!deviceAdmissible(input,input,input,false,0,0,0))
      throw std::runtime_error("HCZ nonphysical nodal/cubature state entering DG RHS");
    auto& m=b.mesh;const auto& p=op->p;
    const dlong count=op->local,entries=count*m.Np*12;
    if(count)o_state.copyFrom(input,entries);
    elementHalo.ExchangeStart(o_state,12*m.Np);
    if(count) {
      b.volumeKernel(count,o_elementIds,m.o_vgeo,m.o_D,m.o_x,m.o_y,m.o_z,
                     time,p.c,b.nu,o_state,output);
      thermodynamicsKernel(count,o_elementIds,p.c,p.rhoL,p.rhoH,p.phiL,p.phiH,
                           p.eosB,p.eosRatio,o_state,o_density,o_potentials);
    }
    elementHalo.ExchangeFinish(o_state,12*m.Np);
    // The same OGS object can be reused after finishing the state exchange.
    elementHalo.ExchangeStart(o_potentials,2*m.Np);
    if(count) {
      b.surfaceKernel(count,o_elementIds,m.o_sgeo,m.o_LIFT,o_vmapM,o_vmapP,o_boundary,
                      m.o_x,m.o_y,m.o_z,time,p.c,b.nu,o_state,output);
      gradVolumeKernel(count,o_elementIds,m.o_vgeo,m.o_D,o_potentials,o_gradients);
      b.relaxationKernel(count,o_elementIds,m.o_vgeo,m.o_cubvgeo,m.o_cubInterp,m.o_cubProject,
                         0,1/p.tauG,p.c,1/p.tauPhi,p.rhoL,p.rhoH,p.phiL,p.phiH,o_state,output);
    }
    elementHalo.ExchangeFinish(o_potentials,2*m.Np);
    if(count) {
      gradSurfaceKernel(count,o_elementIds,m.o_sgeo,o_vmapM,o_vmapP,o_boundary,o_potentials,o_gradients);
      sourceKernel(count,o_elementIds,p.c,p.gravity,p.rhoL,p.rhoH,p.phiL,p.phiH,o_state,o_gradients,output);
    }
    if(p.kappa>0) {
      if(p.subcell)occaHybridCapillarySource(output);else occaCapillarySource(output);
    }
    if(verifyOcca) {
      const auto reference=op->dg(unpack(input)),actual=unpack(output);
      verifyRhs(reference,actual,false);
    }
  }

  void verifyRhs(const hcz::Vec& reference,const hcz::Vec& actual,bool hybrid,bool split=false) {
    double local[2]={0,1},global[2]={};
    for(size_t i=0;i<size_t(op->local)*b.mesh.Np*12;i++) {
      local[0]=(std::isfinite(actual[i]) && std::isfinite(reference[i]))
                 ?std::max(local[0],std::abs(actual[i]-reference[i]))
                 :std::numeric_limits<double>::infinity();
      local[1]=std::max(local[1],std::abs(reference[i]));
    }
    MPI_Allreduce(local,global,2,MPI_DOUBLE,MPI_MAX,b.comm.comm());
    auto& checks=split?splitChecks:(hybrid?hybridChecks:rhsChecks);
    auto& error=split?maxSplitError:(hybrid?maxHybridError:maxRhsError);
    auto& relative=split?maxRelativeSplitError:(hybrid?maxRelativeHybridError:maxRelativeRhsError);
    checks++;error=std::max(error,global[0]);relative=std::max(relative,global[0]/global[1]);
    const double tolerance=std::max(2e-11,100.0*std::numeric_limits<dfloat>::epsilon());
    if(!std::isfinite(global[0]) || !std::isfinite(global[1]) || global[0]>tolerance*global[1])
      throw std::runtime_error(split?"HCZ OCCA SARK4 split RHS differs from host reference":
                               (hybrid?"HCZ OCCA hybrid RHS differs from host reference"
                                      :"HCZ OCCA DG RHS differs from host reference"));
  }

  // Called collectively after occaRhs: o_state contains the unchanged stage
  // input and its DG halos; o_baseRhs preserves that stage's original DG RHS.
  // Recompute all owned subcell fluxes, including those on unmarked elements
  // needed for conservative corrections at mixed faces.
  void occaHybridRhs(deviceMemory<dfloat>& output) {
    const auto& p=op->p;
    const dlong count=op->local;
    const int np=op->ns*op->ns;
    if(count)output.copyFrom(o_baseRhs,count*b.mesh.Np*12);
    elementHalo.ExchangeStart(o_mask,1);
    if(count) {
      projectSubcellKernel(count,o_elementIds,o_A,o_state,o_sub);
      scaleSubcellKernel(count,o_elementIds,p.floor(),o_sub,o_checkFlags);
    }
    elementHalo.ExchangeFinish(o_mask,1);
    admissibilityReduceKernel(count,o_checkFlags,o_checkResult);
    o_checkResult.copyTo(checkResult);
    int invalid=0;
    MPI_Allreduce(checkResult.ptr(),&invalid,1,MPI_INT,MPI_BOR,b.comm.comm());
    if(invalid)throw hcz::SubcellMeanError();

    elementHalo.ExchangeStart(o_sub,12*np);
    if(count)
      subcellThermodynamicsKernel(count,o_elementIds,p.c,p.rhoL,p.rhoH,p.phiL,p.phiH,
                                 p.eosB,p.eosRatio,o_sub,o_subDensity,o_subPotentials);
    elementHalo.ExchangeFinish(o_sub,12*np);
    elementHalo.ExchangeStart(o_subPotentials,2*np);
    if(count) {
      wenoFaceKernel(count,o_elementIds,o_neighbors,p.c,p.alpha,p.wenoEps,o_sub,o_faces);
      subcellRhsKernel(count,o_elementIds,p.c,p.tauPhi,p.tauG,p.rhoL,p.rhoH,p.phiL,p.phiH,
                       op->hx,op->hy,o_sub,o_faces,o_subRhs);
    }
    elementHalo.ExchangeFinish(o_subPotentials,2*np);
    if(count) {
      subcellGradientKernel(count,o_elementIds,o_neighbors,op->hx,op->hy,o_subPotentials,o_subGradients);
      subcellSourceKernel(count,o_elementIds,p.c,p.gravity,p.rhoL,p.rhoH,p.phiL,p.phiH,
                          o_sub,o_subGradients,o_subRhs);
      if(p.kappa>0)subCapillaryAddKernel(count,o_elementIds,o_subCapillaryRhs,o_subRhs);
      reconstructKernel(count,o_elementIds,o_R,o_weights,o_subRhs,output,o_mask);
      mixedSurfaceKernel(count,o_elementIds,o_neighbors,o_mask,o_R,o_weights,p.c,op->hx,op->hy,
                         o_state,o_faces,output);
    }
  }

  void verifyDeviceMask(deviceMemory<int>& mask,const hcz::Vec& reference,const char* label) {
    if(op->local)mask.copyTo(detectorMask,op->local);
    int mismatch=0,global=0;
    for(int e=0;e<op->local;e++)if(detectorMask[e]!=reference[e])mismatch=1;
    MPI_Allreduce(&mismatch,&global,1,MPI_INT,MPI_MAX,b.comm.comm());
    if(global)throw std::runtime_error(std::string("HCZ device ")+label+" mask differs from host reference");
  }

  int deviceMaskCount(deviceMemory<int>& mask) {
    maskCountKernel(op->local,mask,o_checkResult);
    o_checkResult.copyTo(checkResult);
    return checkResult[0];
  }

  // Called after owned index bounds are built. Keep the raw seed separate
  // from the buffered/expanded computation mask for output and verification.
  void markDeviceInterface(const hcz::Vec& state) {
    const auto& p=op->p;
    if(op->local)interfaceMaskKernel(op->local,o_elementIds,o_indexBounds,
      p.phiL+p.interfaceThreshold*(p.phiH-p.phiL),
      p.phiH-p.interfaceThreshold*(p.phiH-p.phiL),o_interfaceMask);
    if(verifyOcca) {
      op->lastInterfaceMask=op->interfaceMask(state);
      verifyDeviceMask(o_interfaceMask,op->lastInterfaceMask,"interface seed");interfaceChecks++;
    }
  }

  void occaSubcellStage(deviceMemory<dfloat>& input,deviceMemory<dfloat>& residual,
                        deviceMemory<dfloat>& output,dfloat dt,dfloat a,dfloat rkB) {
    const auto& p=op->p;
    const dlong count=op->local,entries=count*b.mesh.Np*12;
    if(entries)o_baseRhs.copyFrom(output,entries);
    // All solution-array downloads below are confined to VERIFY. Normal
    // OCCA stages return only scalar counts/flags for host/MPI decisions.
    hcz::Vec state,base,res,mask,next;
    if(verifyOcca) {
      state=unpack(input);base=unpack(output);
      if(a!=0)res=unpack(residual);
    }
    if(count)indexBoundsKernel(count,o_elementIds,input,o_indexBounds);
    elementHalo.ExchangeStart(o_indexBounds,2);
    elementHalo.ExchangeFinish(o_indexBounds,2);
    auto detect=[&](deviceMemory<int>& target,hcz::Vec& reference) {
      if(count)detectorKernel(count,o_elementIds,o_neighbors,o_indexBounds,b.mesh.o_cubInterp,
                              dt,a,rkB,p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.floor(),
                              p.dmpDelta,p.dmpEps,p.boundTol,input,residual,output,target);
      if(verifyOcca) {
        auto rhs=unpack(output),trial=state;
        for(dlong i=0;i<entries;i++)trial[i]+=rkB*(dt*rhs[i]+(a==0?0:a*res[i]));
        reference=op->detect(state,trial);
        verifyDeviceMask(target,reference,"detector");detectorChecks++;
      }
    };
    auto expand=[&](deviceMemory<int>& target,hcz::Vec& reference) {
      libp::hczExpandDeviceMask(elementHalo,expandMaskKernel,count,o_elementIds,o_neighbors,
                                target,o_maskScratch,p.buffer);
      if(verifyOcca) {
        reference=op->expand(reference);
        verifyDeviceMask(target,reference,"buffered");bufferChecks++;
      }
    };
    op->lastTroubled=0;op->candidateChecks++;
    detect(o_mask,mask);
    if(p.interfaceSubcells) {
      markDeviceInterface(state);
      if(count)maskUnionKernel(count,o_interfaceMask,o_mask);
      if(verifyOcca) {
        for(int e=0;e<count;e++)mask[e]=std::max(mask[e],op->lastInterfaceMask[e]);
        verifyDeviceMask(o_mask,mask,"interface/detector union");
      }
    }
    expand(o_mask,mask);
    if(op->any(deviceMaskCount(o_mask)>0)) {
      bool unresolved=false;
      for(int pass=0;pass<p.passes;pass++) {
        occaHybridRhs(output);
        op->recomputations++;
        op->maxRecomputePasses=std::max(op->maxRecomputePasses,pass+1);
        if(verifyOcca)verifyRhs(op->hybrid(state,base,mask),unpack(output),true);
        detect(o_nextMask,next);
        if(count)maskDifferenceKernel(count,o_mask,o_nextMask);
        if(verifyOcca) {
          for(int e=0;e<count;e++)if(mask[e])next[e]=0;
          verifyDeviceMask(o_nextMask,next,"newly troubled");maskUpdateChecks++;
        }
        unresolved=op->any(deviceMaskCount(o_nextMask)>0);
        if(!unresolved)break;
        if(pass+1<p.passes) {
          expand(o_nextMask,next);
          if(count)maskUnionKernel(count,o_nextMask,o_mask);
          if(verifyOcca) {
            for(int e=0;e<count;e++)mask[e]=std::max(mask[e],next[e]);
            verifyDeviceMask(o_mask,mask,"merged");maskUpdateChecks++;
          }
        }
      }
      if(unresolved)throw std::runtime_error("HCZ LSERK4 detector needs more recompute passes or a smaller CFL");
      op->lastTroubled=deviceMaskCount(o_mask);
    }
    // The caller performs the final device admissibility check. Only then
    // does the existing LSERK4 updater commit the accepted RHS/residual.
  }
  void sarkStart(deviceMemory<dfloat>& input) {
    sarkAttempts++;sarkPass=0;sarkHybrid=false;op->lastTroubled=0;
    if(!useOcca || verifyOcca) {
      sarkReference=unpack(input);sarkMask.assign(op->total,0);
    }
    if(useOcca) {
      // The stepper keeps this input unchanged throughout every replay.
      o_sarkReference=input;
      if(op->local) {
        maskClearKernel(op->local,o_mask);
        indexBoundsKernel(op->local,o_elementIds,input,o_indexBounds);
      }
      elementHalo.ExchangeStart(o_indexBounds,2);
      elementHalo.ExchangeFinish(o_indexBounds,2);
    }
    if(op->p.interfaceSubcells) {
      if(useOcca) {
        markDeviceInterface(sarkReference);
        if(op->local)o_mask.copyFrom(o_interfaceMask,op->local);
        libp::hczExpandDeviceMask(elementHalo,expandMaskKernel,op->local,o_elementIds,o_neighbors,
                                  o_mask,o_maskScratch,op->p.buffer);
        op->lastTroubled=deviceMaskCount(o_mask);
      }
      if(!useOcca || verifyOcca) {
        op->lastInterfaceMask=op->interfaceMask(sarkReference);
        sarkMask=op->expand(op->lastInterfaceMask);
        if(!useOcca)for(int e=0;e<op->local;e++)op->lastTroubled+=sarkMask[e]!=0;
      }
      if(verifyOcca)verifyDeviceMask(o_mask,sarkMask,"buffered interface seed");
      // All ranks must evaluate the hybrid operator when any rank is marked.
      // Seeding does not consume a recomputation pass: no RHS exists yet.
      sarkHybrid=op->any(op->lastTroubled>0);
    }
  }

  solver_t::Sark4StageStatus sarkCheck(deviceMemory<dfloat>& candidate,int stage) {
    using Status=solver_t::Sark4StageStatus;
    const auto& p=op->p;const dlong count=op->local;
    // Stage 0 equals the step input; stage 5 repeats the final tableau row.
    // Both still receive hard physical checks in the caller.
    if(stage==0 || stage==5)return Status::Accept;
    op->candidateChecks++;
    hcz::Vec next;
    if(!useOcca || verifyOcca)next=op->detect(sarkReference,unpack(candidate));
    if(useOcca) {
      if(count)sarkDetectorKernel(count,o_elementIds,o_neighbors,o_indexBounds,b.mesh.o_cubInterp,
                                 0,0,0,p.rhoL,p.rhoH,p.phiL,p.phiH,p.eosB,p.floor(),
                                 p.dmpDelta,p.dmpEps,p.boundTol,o_sarkReference,o_sarkReference,candidate,o_nextMask);
      if(verifyOcca){verifyDeviceMask(o_nextMask,next,"SARK4 detector");detectorChecks++;}
      if(count)maskDifferenceKernel(count,o_mask,o_nextMask);
    }
    if(!useOcca || verifyOcca)for(int e=0;e<count;e++)if(sarkMask[e])next[e]=0;
    if(verifyOcca){verifyDeviceMask(o_nextMask,next,"SARK4 newly troubled");maskUpdateChecks++;}
    int marked=0;
    if(useOcca)marked=deviceMaskCount(o_nextMask);
    else for(int e=0;e<count;e++)marked+=next[e]!=0;
    if(!op->any(marked>0))return Status::Accept;
    if(sarkPass>=p.passes) {
      sarkBudgetRejects++;
      if(sarkBudgetRejects==1 && b.comm.rank()==0)
        std::cout<<"HCZ SARK4 recomputation budget exhausted; retrying with a smaller timestep. "
                 <<"Increase HCZ RECOMPUTE PASSES if retries persist."<<std::endl;
      return Status::Reject;
    }
    if(useOcca)
      libp::hczExpandDeviceMask(elementHalo,expandMaskKernel,count,o_elementIds,o_neighbors,
                                o_nextMask,o_maskScratch,p.buffer);
    if(!useOcca || verifyOcca)next=op->expand(next);
    if(verifyOcca){verifyDeviceMask(o_nextMask,next,"SARK4 buffered");bufferChecks++;}
    if(useOcca && count)maskUnionKernel(count,o_nextMask,o_mask);
    if(!useOcca || verifyOcca)for(int e=0;e<count;e++)sarkMask[e]=std::max(sarkMask[e],next[e]);
    if(verifyOcca){verifyDeviceMask(o_mask,sarkMask,"SARK4 merged");maskUpdateChecks++;}
    sarkHybrid=true;sarkPass++;op->recomputations++;
    op->maxRecomputePasses=std::max(op->maxRecomputePasses,sarkPass);
    op->lastTroubled=0;
    if(useOcca)op->lastTroubled=deviceMaskCount(o_mask);
    else for(int e=0;e<count;e++)op->lastTroubled+=sarkMask[e]!=0;
    return Status::Restart;
  }

  void evaluateRhs(deviceMemory<dfloat>& input,deviceMemory<dfloat>& output,dfloat time) {
    const bool hybrid=useSark && op->p.subcell && sarkHybrid;
    if(useOcca) {
      occaRhs(input,output,time);
      if(hybrid) {
        if(op->local)o_baseRhs.copyFrom(output,static_cast<dlong>(op->local)*b.mesh.Np*12);
        hcz::Vec reference;
        if(verifyOcca)reference=op->hybrid(unpack(input),unpack(output),sarkMask);
        occaHybridRhs(output);
        if(verifyOcca)verifyRhs(reference,unpack(output),true);
      }
      if(useSark) {
        if(op->local)sarkSplitKernel(op->local,1/op->p.tauPhi,1/op->p.tauG,input,output);
        if(verifyOcca) {
          const auto state=unpack(input);
          auto reference=op->dg(state);
          if(hybrid)reference=op->hybrid(state,reference,sarkMask);
          for(size_t i=0;i<size_t(op->local)*b.mesh.Np*12;i++)
            if(op->sarkLambda(i%12)!=0)reference[i]-=op->sarkLambda(i%12)*state[i];
          verifyRhs(reference,unpack(output),false,true);
        }
      }
    } else {
      const auto state=unpack(input);
      auto rhs=op->dg(state);
      if(hybrid)rhs=op->hybrid(state,rhs,sarkMask);
      if(useSark)for(size_t i=0;i<size_t(op->local)*b.mesh.Np*12;i++)
        if(op->sarkLambda(i%12)!=0)rhs[i]-=op->sarkLambda(i%12)*state[i];
      pack(rhs,output);
    }
  }
  void exchange(hcz::Vec& q,int stride) {
    if(b.comm.size()==1)return;
    std::vector<int> sc=sendCounts,rc=recvCounts,sd=sendOffsets,rd=recvOffsets;
    for(size_t i=0;i<sc.size();i++) {sc[i]*=stride;rc[i]*=stride;sd[i]*=stride;rd[i]*=stride;}
    hcz::Vec send(sendElements.size()*stride),recv(recvSlots.size()*stride);
    for(size_t a=0;a<sendElements.size();a++) std::copy_n(q.begin()+size_t(sendElements[a])*stride,stride,send.begin()+a*stride);
    MPI_Alltoallv(send.data(),sc.data(),sd.data(),MPI_DOUBLE,recv.data(),rc.data(),rd.data(),MPI_DOUBLE,b.comm.comm());
    for(size_t a=0;a<recvSlots.size();a++) std::copy_n(recv.begin()+a*stride,stride,q.begin()+size_t(recvSlots[a])*stride);
  }
  hcz::Vec unpack(deviceMemory<dfloat>& input) {
    if(deviceStageActive)stageDownloads++;
    int np=b.mesh.Np;memory<dfloat> host(op->local*np*12);input.copyTo(host);
    hcz::Vec out(size_t(op->total)*np*12,0);
    for(int e=0;e<op->local;e++) for(int a=0;a<np;a++) for(int k=0;k<12;k++) out[(size_t(e)*np+a)*12+k]=host[(e*12+k)*np+a];
    return out;
  }
  void pack(const hcz::Vec& input,deviceMemory<dfloat>& output) {
    if(deviceStageActive)stageUploads++;
    int np=b.mesh.Np;memory<dfloat> host(op->local*np*12);
    for(int e=0;e<op->local;e++) for(int a=0;a<np;a++) for(int k=0;k<12;k++) host[(e*12+k)*np+a]=input[(size_t(e)*np+a)*12+k];
    output.copyFrom(host);
  }
  double phi(double x,double y) const {
    auto& p=op->p;double d=0;
    if(initial=="UNIFORM")return p.phiL;
    if(initial=="RT") {
      double wave=2*M_PI/width, slope=-amplitude*width*wave*std::sin(wave*x);
      d=(y-height/2-amplitude*width*std::cos(wave*x))/std::sqrt(1+slope*slope);
    } else if(initial=="SLAB") d=.2*width-std::abs(x-width/2);
    else {
      const double radius=std::hypot(x-bubbleX,y-bubbleY);
      if(compactCircle)return hcz::compactBubble(radius,bubbleRadius,interfaceWidth,p.phiL,p.phiH,lightBubble)[0];
      d=bubbleRadius-radius;
      if(lightBubble)d=-d;
    }
    return p.phiL+.5*(1+std::tanh(d*2*std::atanh(.8)/interfaceWidth))*(p.phiH-p.phiL);
  }
  void initialize() {
    sarkMask.assign(op->total,0);op->lastSubcellMask.assign(op->local,0);
    if(useOcca && op->p.subcell && op->local) {
      memory<int> zeros(op->local,0);o_mask.copyFrom(zeros,op->local);
    }
    hcz::Vec q(size_t(op->total)*op->n*op->n*12,0);
    double pref=.5*(op->p.cs(op->p.phiL)+op->p.cs(op->p.phiH));
    // Unique global GLL y coordinates. Integrate sampled density by the same
    // columnwise trapezoid rule as the Python RT driver, without gathering it.
    hcz::Vec ys(ny*(op->n-1)+1);
    for(int ey=0;ey<ny;ey++)for(int j=0;j<op->n;j++)ys[ey*(op->n-1)+j]=(ey+.5+.5*op->nodes[j])*op->hy;
    for(int e=0;e<op->local;e++)for(int i=0;i<op->n;i++) {
      int ex=gids[e]%nx,ey=gids[e]/nx;double x=(ex+.5+.5*op->nodes[i])*op->hx;
      hcz::Vec pressure(ys.size(),pref);
      if(!periodicY && op->p.gravity!=0) for(int j=int(ys.size())-2;j>=0;j--)
        pressure[j]=pressure[j+1]+.5*op->p.gravity*(ys[j+1]-ys[j])*(op->p.rho(phi(x,ys[j+1]))+op->p.rho(phi(x,ys[j])));
      for(int j=0;j<op->n;j++) {
        int yid=ey*(op->n-1)+j;
        q[op->id(e,j,i,0,op->n)]=phi(x,ys[yid]);q[op->id(e,j,i,6,op->n)]=pressure[yid];
        if(mechanicalBubble)q[op->id(e,j,i,6,op->n)]+=op->p.kappa*bubblePressure.value(std::hypot(x-bubbleX,ys[yid]-bubbleY));
      }
    }
    LIBP_ABORT("HCZ initial profile has nonphysical nodal/cubature values; resolve or widen the interface",!op->admissible(q));
    pack(q,b.o_q);
    op->lastInterfaceMask=op->interfaceMask(q);
    if(useOcca && op->p.interfaceSubcells && op->local) {
      memory<int> seed(op->local);
      for(int e=0;e<op->local;e++)seed[e]=op->lastInterfaceMask[e]!=0;
      o_interfaceMask.copyFrom(seed,op->local);
    }
    auto integrals=integrate(q);initialIndex=integrals[0];initialMass=integrals[1];
  }
  std::array<double,2> integrate(const hcz::Vec& q) {
    double local[2]={},global[2]={};
    for(int e=0;e<op->local;e++)for(int j=0;j<op->n;j++)for(int i=0;i<op->n;i++) {
      double phi=q[op->id(e,j,i,0,op->n)],w=.25*op->hx*op->hy*op->weights[i]*op->weights[j];
      local[0]+=w*phi;local[1]+=w*op->p.rho(phi);
    }
    MPI_Allreduce(local,global,2,MPI_DOUBLE,MPI_SUM,b.comm.comm());return {global[0],global[1]};
  }
  void report(double time,int step) {
    hcz::Vec q=unpack(b.o_q);auto totals=integrate(q);
    double lo=1e300,hi=-1e300,speed=0,globalLo,globalHi,globalSpeed;
    for(int e=0;e<op->local;e++)for(int j=0;j<op->n;j++)for(int i=0;i<op->n;i++) {
      auto s=op->get(q,e,j,i,op->n);lo=std::min(lo,s[0]);hi=std::max(hi,s[0]);speed=std::max(speed,std::hypot(s[7],s[8])/(op->p.rho(s[0])*op->p.c));
    }
    MPI_Allreduce(&lo,&globalLo,1,MPI_DOUBLE,MPI_MIN,b.comm.comm());MPI_Allreduce(&hi,&globalHi,1,MPI_DOUBLE,MPI_MAX,b.comm.comm());
    MPI_Allreduce(&speed,&globalSpeed,1,MPI_DOUBLE,MPI_MAX,b.comm.comm());
    int troubled;MPI_Allreduce(&op->lastTroubled,&troubled,1,MPI_INT,MPI_SUM,b.comm.comm());
    std::string prefix;b.settings.getSetting("OUTPUT FILE NAME",prefix);
    if(initial=="BUBBLE" && b.settings.compareSetting("OUTPUT TO FILE","TRUE")) {
      // Unclipped diffuse phase fraction: its integral follows conserved phi.
      // Reporting uses the same GLL quadrature as the conservation diagnostic.
      double local[5]={},global[5]={};
      for(int e=0;e<op->local;e++)for(int j=0;j<op->n;j++)for(int i=0;i<op->n;i++) {
        const auto s=op->get(q,e,j,i,op->n);
        double a=(s[0]-op->p.phiL)/(op->p.phiH-op->p.phiL);
        if(lightBubble)a=1-a;
        const double w=.25*op->hx*op->hy*op->weights[i]*op->weights[j]*a;
        const double x=(gids[e]%nx+.5+.5*op->nodes[i])*op->hx;
        const double y=(gids[e]/nx+.5+.5*op->nodes[j])*op->hy;
        local[0]+=w;local[1]+=w*x;local[2]+=w*y;
        local[3]+=w*s[7]/(op->p.rho(s[0])*op->p.c);
        local[4]+=w*s[8]/(op->p.rho(s[0])*op->p.c);
      }
      MPI_Allreduce(local,global,5,MPI_DOUBLE,MPI_SUM,b.comm.comm());
      if(b.comm.rank()==0) {
        std::ofstream csv(prefix+"_bubble.csv",frame==0?std::ios::out:std::ios::app);
        LIBP_ABORT("Cannot open HCZ bubble diagnostics",!csv);
        if(frame==0)csv<<"time,area,centroid_x,centroid_y,mean_u,rise_velocity\n";
        csv<<std::setprecision(17)<<time<<","<<global[0];
        for(int k=1;k<5;k++)csv<<","<<(global[0]>0?global[k]/global[0]:std::numeric_limits<double>::quiet_NaN());
        csv<<"\n";
      }
    }
    if(b.comm.rank()==0) {
      std::cout << std::setprecision(12) << "HCZ t="<<time<<" step="<<step<<" mass="<<totals[1]<<" mass error="<<totals[1]-initialMass
        <<" phi=["<<globalLo<<","<<globalHi<<"] max speed="<<globalSpeed<<" troubled="<<troubled<<"\n";
      if(b.settings.compareSetting("OUTPUT TO FILE","TRUE")) {
        std::ofstream csv(prefix+"_diagnostics.csv",frame==0?std::ios::out:std::ios::app);
        LIBP_ABORT("Cannot open HCZ diagnostics output",!csv);
        if(frame==0)csv<<"time,step,index,mass,index_error,mass_error,phi_min,phi_max,max_speed,troubled,candidate_checks,recomputations,max_recompute_passes\n";
        csv<<std::setprecision(17)<<time<<","<<step<<","<<totals[0]<<","<<totals[1]<<","<<totals[0]-initialIndex<<","<<totals[1]-initialMass
          <<","<<globalLo<<","<<globalHi<<","<<globalSpeed<<","<<troubled<<","<<op->candidateChecks<<","<<op->recomputations<<","<<op->maxRecomputePasses<<"\n";
      }
    }
    if(b.settings.compareSetting("OUTPUT TO FILE","TRUE")) {
      // Reporting-only transfer: never infer this mask by redetecting q.
      // Initial data has not undergone an update, so its mask is zero.
      memory<int> active(op->local,0);
      memory<int> interfaceMarked(op->local,0);
      if(op->p.interfaceSubcells) {
        if(useOcca) {if(op->local)o_interfaceMask.copyTo(interfaceMarked,op->local);}
        else for(int e=0;e<op->local;e++)interfaceMarked[e]=op->lastInterfaceMask[e]!=0;
      }
      if(frame>0 && op->p.subcell) {
        if(useOcca) {if(op->local)o_mask.copyTo(active,op->local);}
        else {
          const auto& mask=useSark?sarkMask:op->lastSubcellMask;
          for(int e=0;e<op->local;e++)active[e]=mask[e]!=0;
        }
      }
      char name[4096];snprintf(name,sizeof(name),"%s_%04d_%04d.vtu",prefix.c_str(),b.comm.rank(),frame);
      std::ofstream f(name);LIBP_ABORT("Cannot open HCZ VTU output",!f);f<<std::setprecision(17);
      int np=op->local*op->n*op->n,ncells=op->local*(op->n-1)*(op->n-1);
      f<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\"><UnstructuredGrid>"
       <<"<FieldData><DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">"<<time<<"</DataArray></FieldData>"
       <<"<Piece NumberOfPoints=\""<<np<<"\" NumberOfCells=\""<<ncells<<"\">\n<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
      for(int e=0;e<op->local;e++)for(int j=0;j<op->n;j++)for(int i=0;i<op->n;i++)
        f<<(gids[e]%nx+.5+.5*op->nodes[i])*op->hx<<" "<<(gids[e]/nx+.5+.5*op->nodes[j])*op->hy<<" 0\n";
      f<<"</DataArray></Points><PointData>\n";
      const char* names[]={"Index","Density","Pressure","Velocity"};
      for(int field=0;field<4;field++) {
        f<<"<DataArray type=\"Float64\" Name=\""<<names[field]<<"\" NumberOfComponents=\""<<(field==3?3:1)<<"\" format=\"ascii\">\n";
        for(int e=0;e<op->local;e++)for(int j=0;j<op->n;j++)for(int i=0;i<op->n;i++) {
          auto s=op->get(q,e,j,i,op->n);double rho=op->p.rho(s[0]);
          if(field==0)f<<s[0];else if(field==1)f<<rho;else if(field==2)f<<s[6];else f<<s[7]/(rho*op->p.c)<<" "<<s[8]/(rho*op->p.c)<<" 0";f<<"\n";
        }f<<"</DataArray>\n";
      }
      f<<"</PointData><CellData Scalars=\"SubcellActive\">\n"
       <<"<DataArray type=\"UInt8\" Name=\"SubcellActive\" format=\"ascii\">\n";
      for(int e=0;e<op->local;e++)for(int a=0;a<(op->n-1)*(op->n-1);a++)f<<active[e]<<" ";
      f<<"\n</DataArray><DataArray type=\"UInt8\" Name=\"InterfaceMarked\" format=\"ascii\">\n";
      for(int e=0;e<op->local;e++)for(int a=0;a<(op->n-1)*(op->n-1);a++)f<<interfaceMarked[e]<<" ";
      f<<"\n</DataArray><DataArray type=\"Int64\" Name=\"ElementId\" format=\"ascii\">\n";
      for(int e=0;e<op->local;e++)for(int a=0;a<(op->n-1)*(op->n-1);a++)f<<gids[e]<<" ";
      f<<"\n</DataArray></CellData><Cells><DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
      for(int e=0;e<op->local;e++)for(int j=0;j<op->n-1;j++)for(int i=0;i<op->n-1;i++) {int a=(e*op->n+j)*op->n+i;f<<a<<" "<<a+1<<" "<<a+op->n+1<<" "<<a+op->n<<"\n";}
      f<<"</DataArray><DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";for(int a=1;a<=ncells;a++)f<<4*a<<" ";
      f<<"</DataArray><DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";for(int a=0;a<ncells;a++)f<<"9 ";
      f<<"</DataArray></Cells></Piece></UnstructuredGrid></VTKFile>\n";
    }
    frame++;
  }
};

void bns_t::HczSetup() {
  hcz=std::make_shared<bnsHcz_t>(*this);hcz->setup();
}
dfloat bns_t::sark4MaxTimeStep(const dfloat time) {
  return hcz?hcz->sarkDtMax:std::numeric_limits<dfloat>::max();
}
void bns_t::sark4Tolerances(dfloat& atol,dfloat& rtol) {
  if(hcz) {atol=hcz->sarkAtol;rtol=hcz->sarkRtol;}
}
bool bns_t::sark4Admissible(deviceMemory<dfloat>& input,const dfloat time) {
  if(!hcz)return true;
  if(!hcz->useOcca)return hcz->op->admissible(hcz->unpack(input));
  hcz->deviceStageActive=true;
  const bool valid=hcz->deviceAdmissible(input,input,input,false,0,0,0);
  hcz->deviceStageActive=false;
  return valid;
}
void bns_t::sark4StartStep(deviceMemory<dfloat>& input,const dfloat time,const dfloat dt) {
  if(hcz && hcz->op->p.subcell) {
    hcz->deviceStageActive=hcz->useOcca;
    hcz->sarkStart(input);
    hcz->deviceStageActive=false;
  }
}
void bns_t::sark4SaveOutputState() {
  if(!hcz || !hcz->op->p.subcell)return;
  hcz->savedTroubled=hcz->op->lastTroubled;
  hcz->savedSarkPass=hcz->sarkPass;hcz->savedSarkHybrid=hcz->sarkHybrid;
  if(hcz->useOcca && hcz->op->local)hcz->o_savedMask.copyFrom(hcz->o_mask,hcz->op->local);
  if(!hcz->useOcca || hcz->verifyOcca)hcz->savedSarkMask=hcz->sarkMask;
  if(hcz->op->p.interfaceSubcells) {
    if(hcz->useOcca && hcz->op->local)hcz->o_savedInterfaceMask.copyFrom(hcz->o_interfaceMask,hcz->op->local);
    if(!hcz->useOcca || hcz->verifyOcca)hcz->savedInterfaceMask=hcz->op->lastInterfaceMask;
  }
}
void bns_t::sark4RestoreOutputState() {
  if(!hcz || !hcz->op->p.subcell)return;
  hcz->op->lastTroubled=hcz->savedTroubled;
  hcz->sarkPass=hcz->savedSarkPass;hcz->sarkHybrid=hcz->savedSarkHybrid;
  if(hcz->useOcca && hcz->op->local)hcz->o_mask.copyFrom(hcz->o_savedMask,hcz->op->local);
  if(!hcz->useOcca || hcz->verifyOcca)hcz->sarkMask=hcz->savedSarkMask;
  if(hcz->op->p.interfaceSubcells) {
    if(hcz->useOcca && hcz->op->local)hcz->o_interfaceMask.copyFrom(hcz->o_savedInterfaceMask,hcz->op->local);
    if(!hcz->useOcca || hcz->verifyOcca)hcz->op->lastInterfaceMask=hcz->savedInterfaceMask;
  }
}
solver_t::Sark4StageStatus bns_t::sark4CheckStage(deviceMemory<dfloat>& input,const dfloat time,const int stage) {
  if(hcz && hcz->op->p.subcell) {
    hcz->deviceStageActive=hcz->useOcca;
    const auto status=hcz->sarkCheck(input,stage);
    hcz->deviceStageActive=false;
    if(status!=Sark4StageStatus::Accept)return status;
  }
  return sark4Admissible(input,time)?Sark4StageStatus::Accept:Sark4StageStatus::Reject;
}
bool bns_t::sark4Rhs(deviceMemory<dfloat>& input,deviceMemory<dfloat>& output,const dfloat time) {
  if(!hcz)return solver_t::sark4Rhs(input,output,time);
  try {
    hcz->deviceStageActive=hcz->useOcca;
    hcz->evaluateRhs(input,output,time);
    hcz->deviceStageActive=false;
    return true;
  } catch(const hcz::SubcellMeanError&) {
    hcz->deviceStageActive=false;hcz->sarkMeanRejects++;return false;
  } catch(const std::exception& ex) {
    LIBP_FORCE_ABORT("HCZ SARK4 RHS at t="<<time<<": "<<ex.what());
    return false;
  }
}
void bns_t::rhsf(deviceMemory<dfloat>& input,deviceMemory<dfloat>& output,const dfloat T) {
  LIBP_ABORT("Non-PML rhsf is reserved for HCZ in BNS",!hcz);
  try {
    hcz->deviceStageActive=hcz->useOcca && hcz->useSark;
    hcz->evaluateRhs(input,output,T);
    hcz->deviceStageActive=false;
  }
  catch(const std::exception& ex) {LIBP_FORCE_ABORT("HCZ RHS at t="<<T<<": "<<ex.what());}
}
void bns_t::rhsf_lserk4(deviceMemory<dfloat>& input,deviceMemory<dfloat>& residual,
                       deviceMemory<dfloat>& output,const dfloat T,const dfloat dt,const dfloat a,const dfloat b) {
  LIBP_ABORT("HCZ LSERK4 callback requires HCZ model",!hcz);
  try {
    if(hcz->useOcca) {
      hcz->deviceStageActive=true;
      hcz->occaRhs(input,output,T);
      if(hcz->op->p.subcell)hcz->occaSubcellStage(input,residual,output,dt,a,b);
      else hcz->op->lastTroubled=0;
      if(!hcz->deviceAdmissible(input,residual,output,true,dt,a,b))
        throw std::runtime_error(hcz->lserkCandidateFailure(input,residual,output,dt,a,b));
      hcz->deviceStageActive=false;
    } else {
      auto state=hcz->unpack(input);hcz::Vec res;
      if(a!=0)res=hcz->unpack(residual);
      hcz->pack(hcz->op->stage(state,res,dt,a,b),output);
    }
  } catch(const std::exception& ex) {LIBP_FORCE_ABORT("HCZ LSERK4 stage at t="<<T<<": "<<ex.what());}
}
void bns_t::HczRun() {
  hcz->initialize();auto& op=*hcz->op;
  double start,finish,cfl,collisionCfl,subCfl,interval;
  settings.getSetting("START TIME",start);settings.getSetting("FINAL TIME",finish);settings.getSetting("OUTPUT INTERVAL",interval);
  settings.getSetting("CFL NUMBER",cfl);settings.getSetting("HCZ COLLISION CFL",collisionCfl);settings.getSetting("HCZ SUBCELL CFL",subCfl);
  LIBP_ABORT("Invalid HCZ run times/CFL",!std::isfinite(start) || !std::isfinite(finish) || finish<start || !std::isfinite(interval) || interval<=0 ||
    !std::isfinite(cfl) || cfl<=0 || !std::isfinite(collisionCfl) || collisionCfl<=0 || !std::isfinite(subCfl) || subCfl<=0);
  const double advDt=cfl/(std::sqrt(3.0)*c*op.n*op.n*(1/op.hx+1/op.hy));
  hcz->sarkDtMax=advDt;
  double dt=hcz->useSark?advDt:std::min(advDt,collisionCfl*std::min(op.p.tauPhi,op.p.tauG));
  if(op.p.subcell) {
    const double subDt=subCfl/(op.p.alpha*c*op.ns*(1/op.hx+1/op.hy));
    dt=std::min(dt,subDt);hcz->sarkDtMax=std::min(advDt,subDt);
  }
  // A nonzero capillary force adds a third-derivative stiffness. This bound
  // is conservative and must still be checked under refinement for each case.
  if(op.p.kappa>0) {
    double h=std::min(op.hx,op.hy)/(op.p.subcell?op.ns:op.n*op.n);
    dt=std::min(dt,cfl*std::sqrt(op.p.rhoL*std::pow(h,4)/(op.p.kappa*op.p.phiH*op.p.phiH)));
  }
  timeStepper.SetTimeStep(dt);
  if(comm.rank()==0)std::cout<<"HCZ dt="<<std::setprecision(17)<<dt<<"\n";
  if(hcz->useSark && comm.rank()==0)
    std::cout<<"HCZ SARK4: max_dt="<<hcz->sarkDtMax<<" atol="<<hcz->sarkAtol<<" rtol="<<hcz->sarkRtol<<"\n";
  timeStepper.Run(*this,o_q,start,finish);
  if(hcz->verifyOcca && comm.rank()==0)
    std::cout<<"HCZ OCCA verification: rhs_checks="<<hcz->rhsChecks
             <<" max_error="<<hcz->maxRhsError
             <<" max_scaled_error="<<hcz->maxRelativeRhsError
             <<" admissibility_checks="<<hcz->admissibilityChecks
             <<" hybrid_checks="<<hcz->hybridChecks
             <<" max_hybrid_error="<<hcz->maxHybridError
             <<" max_scaled_hybrid_error="<<hcz->maxRelativeHybridError
             <<" detector_checks="<<hcz->detectorChecks
             <<" buffer_checks="<<hcz->bufferChecks
             <<" mask_update_checks="<<hcz->maskUpdateChecks
             <<" split_checks="<<hcz->splitChecks<<" max_split_error="<<hcz->maxSplitError
             <<" max_scaled_split_error="<<hcz->maxRelativeSplitError
             <<" interface_checks="<<hcz->interfaceChecks<<"\n";
  if(hcz->useOcca) {
    long long localTransfers[2]={hcz->stageDownloads,hcz->stageUploads},globalTransfers[2]={};
    MPI_Allreduce(localTransfers,globalTransfers,2,MPI_LONG_LONG_INT,MPI_SUM,comm.comm());
    if(comm.rank()==0)
      std::cout<<"HCZ device stages: full_array_downloads="<<globalTransfers[0]
               <<" full_array_uploads="<<globalTransfers[1]<<"\n";
  }
  if(hcz->useSark && op.p.subcell && comm.rank()==0)
    std::cout<<"HCZ SARK4 hybrid: attempts="<<hcz->sarkAttempts
             <<" replays="<<op.recomputations<<" budget_rejections="<<hcz->sarkBudgetRejects
             <<" mean_rejections="<<hcz->sarkMeanRejects<<"\n";
  // LSERK4 only reports scheduled intermediate samples; always emit the final state.
  HczReport(finish,-1);
}
void bns_t::HczReport(dfloat time,int tstep) {hcz->report(time,tstep);}
