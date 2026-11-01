/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "bns.hpp"

//settings for bns solver
bnsSettings_t::bnsSettings_t(comm_t& _comm):
  settings_t(_comm) {

  newSetting("BNS MODEL", "BNS", "Boltzmann formulation",
             {"BNS", "HCZ"});
  newSetting("HCZ DG BACKEND", "HOST", "HCZ backend; OCCA/VERIFY use device DG/subcell stages and require kappa=0",
             {"HOST", "OCCA", "VERIFY"});
  newSetting("HCZ RHO LIGHT", "1", "Physical light-phase density");
  newSetting("HCZ RHO HEAVY", "3", "Physical heavy-phase density");
  newSetting("HCZ PHI LIGHT", "0.02381", "Light-phase index endpoint");
  newSetting("HCZ PHI HEAVY", "0.2508", "Heavy-phase index endpoint");
  newSetting("HCZ EOS B", "4", "Carnahan-Starling b");
  newSetting("HCZ EOS A OVER B C2", "3", "Carnahan-Starling a/(b*c*c)");
  newSetting("HCZ TAU PHI", "0", "Index relaxation time; 0 uses viscosity/c^2");
  newSetting("HCZ SARK4 ATOL", "1e-8", "SARK4 absolute error tolerance for HCZ");
  newSetting("HCZ SARK4 RTOL", "1e-6", "SARK4 relative error tolerance for HCZ");
  newSetting("HCZ KAPPA", "0", "Index capillarity coefficient");
  newSetting("HCZ GRAVITY", "0.04", "Downward gravitational acceleration");
  newSetting("HCZ INITIAL CONDITION", "RT", "HCZ initial geometry",
             {"RT", "SLAB", "BUBBLE", "UNIFORM", "KH"});
  newSetting("HCZ KH VELOCITY TOP", "0.1", "KH: x velocity of the upper (light) layer");
  newSetting("HCZ KH VELOCITY BOTTOM", "-0.1", "KH: x velocity of the lower (heavy) layer");
  newSetting("HCZ KH PERTURBATION", "VELOCITY", "KH: velocity disturbance (Guo et al. 2024 Eq. 31) or interface displacement (Eq. 19)",
             {"VELOCITY", "DISPLACEMENT"});
  newSetting("HCZ KH AMPLITUDE", "0.006", "KH: disturbance amplitude; VELOCITY: du=dv (velocity units), DISPLACEMENT: A0 (length units)");
  newSetting("HCZ KH WAVES", "1", "KH: number of disturbance wavelengths across the box width");
  newSetting("HCZ KH SHEAR WIDTH", "0", "KH: 10-90 width of the tanh velocity jump; 0 uses HCZ INTERFACE WIDTH");
  newSetting("HCZ Y BOUNDARY", "WALL", "Vertical kinetic boundary; SLIP is a y-normal specular wall",
             {"WALL", "SLIP", "PERIODIC"});
  newSetting("HCZ X BOUNDARY", "PERIODIC", "Horizontal kinetic boundary",
             {"PERIODIC", "SLIP"});
  newSetting("HCZ INTERFACE WIDTH", "0.03125", "Initial index-profile 10-90 width");
  newSetting("HCZ PERTURBATION", "0.1", "RT amplitude as fraction of width");
  newSetting("HCZ BUBBLE RADIUS", "0", "Circle radius; 0 uses 0.2 min(Lx,Ly)");
  newSetting("HCZ BUBBLE CENTER X", "-1", "Circle center in [0,Lx]; -1 uses Lx/2");
  newSetting("HCZ BUBBLE CENTER Y", "-1", "Circle center in [0,Ly]; -1 uses Ly/2");
  newSetting("HCZ BUBBLE PHASE", "HEAVY", "Phase inside the circle", {"HEAVY", "LIGHT"});
  newSetting("HCZ BUBBLE PROFILE", "TANH", "Radial index profile", {"TANH", "COMPACT"});
  newSetting("HCZ BUBBLE PRESSURE", "NONE", "Capillary pressure initialization; MECHANICAL requires COMPACT",
             {"NONE", "MECHANICAL"});
  newSetting("HCZ SUBCELL WENO", "TRUE", "Candidate-triggered WENO3-Z",
             {"TRUE", "FALSE"});
  newSetting("HCZ INTERFACE SUBCELLS", "FALSE", "Seed WENO cells from the evolving index interface",
             {"TRUE", "FALSE"});
  newSetting("HCZ INTERFACE THRESHOLD", "0.01", "Normalized index band: threshold < alpha < 1-threshold");
  newSetting("HCZ DMP DELTA", "0.005", "Absolute DMP tolerance / index jump");
  newSetting("HCZ DMP EPSILON", "0.05", "Relative DMP tolerance");
  newSetting("HCZ BOUND TOLERANCE", "0.01", "Phase-bound tolerance / index jump");
  newSetting("HCZ BUFFER LAYERS", "1", "Troubled-cell buffer layers");
  newSetting("HCZ RECOMPUTE PASSES", "2", "Maximum stage recomputations");
  newSetting("HCZ WENO FLUX", "CHARACTERISTIC", "WENO3-Z reconstruction and flux",
             {"CHARACTERISTIC", "LF"});
  newSetting("HCZ WENO ALPHA", "1.75", "Subcell CFL speed bound / c; also LF penalty in LF mode; at least sqrt(3)");
  newSetting("HCZ WENO EPSILON", "1e-12", "WENO-Z epsilon");
  newSetting("HCZ COLLISION CFL", "0.2", "Explicit BGK time-step factor");
  newSetting("HCZ SUBCELL CFL", "0.18", "Subcell transport time-step factor");
  newSetting("HCZ VTU EVERY", "1", "Write a VTU file every N reports; 0 writes none (CSV diagnostics are unaffected)");

  newSetting("DATA FILE",
             "data/bnsTalorVortex2D.h",
             "Boundary and Initial conditions header");

  newSetting("SPEED OF SOUND",
             "1.0",
             "Speed of sound in fluid");

  newSetting("VISCOSITY",
             "1.0",
             "Fluid viscosity");

  newSetting("PML PROFILE ORDER",
             "4",
             "Polynomial degree of PML damping");

  newSetting("PML SIGMAX MAX",
             "100",
             "Coefficent of polynomial PML damping in x direction");

  newSetting("PML SIGMAY MAX",
             "100",
             "Coefficent of polynomial PML damping in y direction");

  newSetting("PML SIGMAZ MAX",
             "100",
             "Coefficent of polynomial PML damping in z direction");

  newSetting("PML INTEGRATION",
             "COLLOCATION",
             "Type of integration rule to PML damping profile",
             {"COLLOCATION", "CUBATURE"});

  newSetting("TIME INTEGRATOR",
             "DOPRI5",
             "Time integration method",
             {"AB3", "SAAB3", "DOPRI5", "LSERK4", "SARK4", "SARK5", "MRAB3", "MRSAAB3"});

  newSetting("CFL NUMBER",
             "1.0",
             "Multiplier for timestep stability bound");

  newSetting("START TIME",
             "0",
             "Start time for time integration");

  newSetting("FINAL TIME",
             "10",
             "End time for time integration");

  newSetting("OUTPUT INTERVAL",
             ".1",
             "Time between printing output data");

  newSetting("OUTPUT TO FILE",
             "FALSE",
             "Flag for writing fields to VTU files",
             {"TRUE", "FALSE"});

  newSetting("OUTPUT FILE NAME",
             "bns");
}

void bnsSettings_t::report() {

  if (comm.rank()==0) {
    std::cout << "BNS Settings:\n\n";
    reportSetting("BNS MODEL");
    if (compareSetting("BNS MODEL", "HCZ")) {
      for (const char* name : {"SPEED OF SOUND", "VISCOSITY", "TIME INTEGRATOR",
                              "CFL NUMBER", "START TIME", "FINAL TIME",
                              "OUTPUT INTERVAL", "OUTPUT TO FILE", "OUTPUT FILE NAME"})
        reportSetting(name);
      for (const auto& entry : settings)
        if (entry.first.find("HCZ ")==0) reportSetting(entry.first);
      return;
    }
    reportSetting("DATA FILE");
    reportSetting("SPEED OF SOUND");
    reportSetting("VISCOSITY");
    reportSetting("PML PROFILE ORDER");
    reportSetting("PML SIGMAX MAX");
    reportSetting("PML SIGMAY MAX");
    reportSetting("PML SIGMAZ MAX");
    reportSetting("PML INTEGRATION");
    reportSetting("TIME INTEGRATOR");
    reportSetting("START TIME");
    reportSetting("FINAL TIME");
    reportSetting("OUTPUT INTERVAL");
    reportSetting("OUTPUT TO FILE");
    reportSetting("OUTPUT FILE NAME");
  }
}

void bnsSettings_t::parseFromFile(platformSettings_t& platformSettings,
                                  meshSettings_t& meshSettings,
                                  const std::string filename) {
  //read all settings from file
  settings_t s(comm);
  s.readSettingsFromFile(filename);

  for(auto it = s.settings.begin(); it != s.settings.end(); ++it) {
    setting_t& set = it->second;
    const std::string name = set.getName();
    const std::string val = set.getVal<std::string>();
    if (platformSettings.hasSetting(name))
      platformSettings.changeSetting(name, val);
    else if (meshSettings.hasSetting(name))
      meshSettings.changeSetting(name, val);
    else if (hasSetting(name)) //self
      changeSetting(name, val);
    else  {
      LIBP_FORCE_ABORT("Unknown setting: [" << name << "] requested");
    }
  }
}
