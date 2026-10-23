#!/usr/bin/env python3
"""
================================================================================
 cd_nozzle_solver_axiconv_openplume_hybrid.py
 2D compressible Euler solver for a converging-diverging (CD) nozzle.
 Planar by default (full channel, y in [-h(x), +h(x)]), axisymmetric optional.
 Single file, self-contained.
================================================================================

 HYBRID-FLUX VARIANT  (this file = ..._openplume.py + an anti-carbuncle flux)
 -------------------------------------------------------------------------
 HLLC resolves contact and shear waves exactly, but on a fine mesh a strong
 normal shock aligned with the grid (e.g. a Mach disc on the axis) can grow
 the "carbuncle" instability: spurious back-flow and total-pressure loss
 behind the shock.  HLL (HLLE) has no contact wave, is more dissipative and
 is carbuncle-free.  --flux hybrid (default here) uses HLL only on faces next
 to a strong-shock cell (pressure ratio to any neighbour > 1 + --hll-thr,
 default 0.5, flag dilated by --hll-dilate cells, default 1) and HLLC
 everywhere else.  --flux hllc reproduces the open-plume solver exactly;
 --flux hll uses HLL everywhere.

 OPEN-PLUME VARIANT  (this file = cd_nozzle_solver_axiconv.py + 2 changes)
 -------------------------------------------------------------------------
 In the original, the plume block's outer boundary started AT the lip radius
 and widened only over 0.5*L_plume, and the lateral far field was a 1D
 Riemann-invariant condition that mixes the hot jet's sound speed with the
 ambient one.  Together they made the region past the lip behave like a
 continuation of the diverging nozzle: the jet kept expanding along the
 boundary, dropped towards vacuum and then needed one strong normal shock to
 recover p_amb.  A normal shock in a widening duct has no unique position, so
 every mesh settled in a different state (shock at x~2.2, ~3.4 or ~7.5).
   1. --plume-ramp L : the outer boundary now opens to its full radius over a
      short length L after the lip (default 0.5*D_exit) instead of 0.5*L_plume.
   2. Lateral boundary of the plume block is a PRESSURE OPENING
      (entrainment) condition: outflow -> p = p_amb, rho and velocity from the
      interior (supersonic outflow fully extrapolated); inflow -> ambient
      p_amb, T_amb, normal velocity from the interior, tangential velocity 0.
      A constant-pressure edge is the physical free-jet boundary condition.
 Everything else (numerics, gas model, nozzle, inlet/outlet, post-processing,
 JSON layout) is unchanged.  --legacy-plume restores the original behaviour.

 WHAT THIS IS
 ------------
 A consolidated rewrite of cd_nozzle_solver.py.  The original solved one case
 correctly -- the fully-expanded choked nozzle -- but by construction it could
 not represent a shock wave at all, carried no thermodynamic model beyond
 gamma = 1.4, produced no JSON, no grid numbering and no thrust.  Everything
 asked for has been added and the defects listed in the CHANGELOG at the bottom
 of this file have been fixed.

 PLANAR -> AXISYMMETRIC JSON  (this file vs cd_nozzle_solver_v2.py)
 --------------------------------------------------------------------
 The solver is identical to v2.  The only addition is a post-processing
 converter, planar_to_axisym():  after a PLANAR run has converged, its data
 is stitched onto an axisymmetric configuration (upper-half nodes reused as
 r = y >= 0, each cell averaged with its -y mirror) and the JSON is written
 for that axisymmetric configuration INSTEAD of the planar one -- grid,
 fields, mass flow, thrust, shock/wave analysis and theory all recomputed on
 the axisymmetric geometry.  The console summary, the plot and the .npz
 checkpoint remain those of the planar solve.  --planar-json restores the
 original planar JSON; --axisym runs are solved and written as in v2.

 PHYSICS / NUMERICS
 ------------------
   * Governing eqs : 2D compressible Euler (inviscid), conservative form.
                     Axisymmetric mode uses the r-weighted conservative form
                       d/dt(y U A) + sum_faces y_f (F.n) L = (0, 0, p A, 0)
                     so the axis needs no 1/y source term.
   * Discretisation: cell-centred finite volume, body-fitted structured mesh
   * Riemann solver: HLLC in the rotated face-normal frame (Toro)
   * Reconstruction: 1st order (default) or 2nd-order MUSCL + minmod, with
                     bounds-preserving face clamping
   * Time march    : SSP-RK2 (Heun), local time stepping, implicit residual
                     smoothing, checkpoint / restart with grid sequencing
   * Closure       : cpg / tpg / idg -- see GAS MODELS below

 GAS MODELS  (--gas-model)
   cpg   calorically perfect .... constant gamma, constant R  (the original)
   tpg   thermally perfect ...... gamma(T) from a multi-mode harmonic-
                                  oscillator vibrational model; R constant
                                  (default)
   idg   ideal dissociating gas . Lighthill / Freeman:
                                  alpha^2/(1-alpha) = (rho_d/rho) exp(-Td/T)
                                  R_eff(T,rho) = R0 (1 + alpha)
                                  gamma_frozen  = (4 + alpha)/3
   So both R and gamma become functions of the thermodynamic state, and every
   isentropic / stagnation / shock relation in the code is solved for that gas
   rather than with a single frozen gamma.

 SHOCK WAVES
 -----------
 The original could not produce one: the outlet was hard-wired to supersonic
 extrapolation, so no back pressure could ever be felt and no external domain
 existed.  Now:
   * --outlet pressure --pb <Pa>   puts a normal shock inside the divergent
   * --plume                       adds an external block so the overexpanded /
                                   underexpanded jet, its oblique shock train,
                                   barrel shock and Mach disc are captured
   * initiation parameters : design / 1st / 2nd / 3rd critical pressure ratios,
                             the pb/p0 window for an in-nozzle shock, and a
                             regime classification
   * detection             : 2D sensor (normalised pressure jump + compression
                             + shock-normal Mach), connected-structure labelling
   * per-structure output  : wave angle beta, deflection theta, M1, Mn1, M2,
                             p2/p1, rho2/rho1, T2/T1, entropy rise, p02/p01,
                             both from the solution and from a real-gas
                             Rankine-Hugoniot solve
   * propagation           : Mach angle, Prandtl-Meyer angle, shock-cell
                             spacing vs Prandtl, Mach-disc location vs
                             Ashkenas-Sherman

 OUTPUT
 ------
 Every cell carries a global element number
      cell_id = j * nx + i        (j transverse, i axial; C-order ravel)
 The JSON writes grid.elements FIRST and every array under "fields" follows
 exactly that order: T, p, rho, |V|, u, v, flow direction, M, a, gamma, R, cp,
 cv, alpha, T0, p0, entropy, Mach angle, Prandtl-Meyer angle, dynamic pressure,
 mass flux, total enthalpy, pressure gradient, divergence, vorticity, shock
 sensor, shock-normal Mach, shock flag and shock structure id.

 THRUST
 ------
 Computed three independent ways -- exit-plane momentum integral, wall-pressure
 plus inlet-momentum control-volume balance, and the ideal quasi-1D value --
 together with Isp, c*, C_F, discharge coefficient and vacuum thrust.  The
 disagreement between the first two routes is reported: it is a direct measure
 of how well the discrete momentum equation has converged.

 MESH NOTE
 ---------
 --cluster-mode throat (default) refines the axial mesh towards x = 0.  The
 original distribution, documented as "clustering toward the throat", actually
 refined the inlet and the outlet: at Nx=160 the throat cell came out 5.9x the
 inlet cell.  Use --cluster-mode ends only to reproduce old runs.
 Likewise --profile ellipse is now parallel at the exit lip; the original
 contour is available as --profile ellipse-legacy.

 USAGE
 -----
   python3 cd_nozzle_solver_axiconv_openplume_hybrid.py                     # default validated case
   python3 cd_nozzle_solver_axiconv_openplume_hybrid.py --nx 240 --ny 96
   python3 cd_nozzle_solver_axiconv_openplume_hybrid.py --order 2
   python3 cd_nozzle_solver_axiconv_openplume_hybrid.py --outlet pressure --pb 0.9e5   # shock inside
   python3 cd_nozzle_solver_axiconv_openplume_hybrid.py --plume --p-amb 1.013e5        # shock diamonds
   python3 cd_nozzle_solver_axiconv_openplume_hybrid.py --axisym --gas h2 --T0 2500 --gas-model tpg
   python3 cd_nozzle_solver_axiconv_openplume_hybrid.py --list-presets
   python3 cd_nozzle_solver_axiconv_openplume_hybrid.py --help

 DEPENDENCIES
 ------------
 numpy (required), matplotlib (only for the figure; use --no-plot to skip).

 LIMITATIONS (state them in any report)
   * Inviscid: no boundary layers, no separation, no turbulence quantities.
     Any "turbulent kinetic energy" from an Euler solution is meaningless.
   * Single species.  The dissociation model is a two-parameter phenomeno-
     logical fit, not finite-rate chemistry; theta_d / rho_d are tunable.
   * Steady solver.  A shock foot near the exit at low NPR is a genuinely
     unsteady, separated regime and can limit-cycle instead of converging.
   * 1st order smears shocks; 2nd order is sharper but can stall near 1e-3
     residual because of limiter chatter.

 AUTHOR : Shashank Suresh Srinivasan and a teammate -- OVGU, Chemical & Energy Engineering
          (MachViz team project; written with an AI coding assistant)
================================================================================
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np

R_UNIVERSAL = 8314.462618      # J/(kmol*K)
G0 = 9.80665                   # m/s^2, for Isp


# =============================================================================
# 0.  SMALL NUMERICAL HELPERS
# =============================================================================
def bisect(f, lo, hi, tol=1e-13, maxit=200):
    """Robust scalar bisection; returns the better endpoint if not bracketed."""
    flo, fhi = f(lo), f(hi)
    if not np.isfinite(flo) or not np.isfinite(fhi):
        return lo if np.isfinite(flo) else hi
    if flo * fhi > 0.0:
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or 0.5 * (hi - lo) < tol * max(1.0, abs(mid)):
            return mid
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def _fv_cv(x):
    """Einstein function x^2 e^x/(e^x-1)^2  ->  cv contribution of one mode/R."""
    x = np.clip(np.asarray(x, dtype=float), 1e-10, 80.0)
    ex = np.exp(x)
    return x * x * ex / np.expm1(x) ** 2


def _fv_e(x):
    """x/(e^x-1)   ->  e_vib of one mode = R*T*_fv_e(theta/T)."""
    x = np.clip(np.asarray(x, dtype=float), 1e-10, 80.0)
    return x / np.expm1(x)


def _fv_s(x):
    """Vibrational entropy of one mode / R:  x/(e^x-1) - ln(1-e^-x)."""
    x = np.clip(np.asarray(x, dtype=float), 1e-10, 80.0)
    return x / np.expm1(x) - np.log1p(-np.exp(-x))


def _jsafe(x):
    """JSON-safe scalar: NaN/Inf -> None, numpy -> python."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def sig(x, n=6):
    """Round to n significant digits, JSON safe."""
    v = _jsafe(x)
    return None if v is None else float(f"{v:.{n}g}")


def sig_arr(a, n=6):
    a = np.asarray(a, dtype=float).ravel()
    out = []
    for x in a:
        out.append(float(f"{x:.{n}g}") if math.isfinite(x) else None)
    return out


# =============================================================================
# 1.  GAS MODEL  --  gamma(T)  and (optionally)  R(T, rho)
# =============================================================================
class Gas:
    """Ideal-gas thermodynamics with temperature dependent properties.

    Internal degrees of freedom
    ---------------------------
        cv(T)/R = n_base + sum_i g_i * E(theta_i/T)      E = Einstein function
        n_base  = 3/2 (monatomic) | 5/2 (linear) | 3 (non-linear)
        e(T)/R  = n_base*T + sum_i g_i * theta_i/(exp(theta_i/T)-1)
        gamma(T)= (cv+R)/cv

    The mode list carries a *degeneracy* g_i, which also lets a mixture be
    represented by mole-fraction weights (air = 0.781 N2 + 0.209 O2).

    Dissociation ("idg", Lighthill / Freeman ideal dissociating gas)
    ---------------------------------------------------------------
        alpha^2/(1-alpha) = (rho_d/rho) exp(-theta_d/T)
        p = rho * R0 * (1+alpha) * T          ->  R_eff(T,rho) = R0 (1+alpha)
        e = R0 * (3 T + alpha * theta_d)
        frozen  cv = 3 R0,  cp = R0 (4+alpha),  gamma_f = (4+alpha)/3
    This is the mechanism by which the *gas constant* becomes a function of
    temperature: the mean molar mass drops as the molecule splits.
    theta_d and rho_d are phenomenological and can be overridden on the CLI.
    """

    NBASE = {"mono": 1.5, "linear": 2.5, "nonlinear": 3.0}

    #  name : molar mass [kg/kmol], geometry, [(theta_vib [K], degeneracy), ...]
    #         theta_d [K] and rho_d [kg/m^3] for the dissociation model
    LIB = {
        "h2":  dict(M=2.01588,   geom="linear",    modes=[(6332.0, 1.0)],
                    theta_d=52000.0,  rho_d=1.0e4),
        "n2":  dict(M=28.0134,   geom="linear",    modes=[(3393.0, 1.0)],
                    theta_d=113500.0, rho_d=1.3e5),
        "o2":  dict(M=31.9988,   geom="linear",    modes=[(2273.0, 1.0)],
                    theta_d=59500.0,  rho_d=1.5e5),
        "air": dict(M=28.9647,   geom="linear",    modes=[(3393.0, 0.7808),
                                                          (2273.0, 0.2095)],
                    theta_d=113500.0, rho_d=1.3e5),
        "co":  dict(M=28.0101,   geom="linear",    modes=[(3122.0, 1.0)],
                    theta_d=129000.0, rho_d=1.4e5),
        "co2": dict(M=44.0095,   geom="linear",    modes=[(960.0, 2.0),
                                                          (1918.0, 1.0),
                                                          (3380.0, 1.0)]),
        "h2o": dict(M=18.01528,  geom="nonlinear", modes=[(2295.0, 1.0),
                                                          (5261.0, 1.0),
                                                          (5404.0, 1.0)]),
        "ch4": dict(M=16.0425,   geom="nonlinear", modes=[(1879.0, 3.0),
                                                          (2207.0, 2.0),
                                                          (4197.0, 1.0),
                                                          (4343.0, 3.0)]),
        "he":  dict(M=4.002602,  geom="mono",      modes=[]),
        "ar":  dict(M=39.948,    geom="mono",      modes=[]),
    }

    T_FLOOR = 5.0          # K, keeps exp() and the Newton solves finite
    E_FLOOR = 1.0          # J/kg

    def __init__(self, name="h2", model="tpg", gamma_const=None,
                 R_override=None, T0_ref=2500.0,
                 theta_d=None, rho_d=None):
        name = name.lower()
        if name not in self.LIB:
            raise ValueError(f"unknown gas '{name}'; choose from {sorted(self.LIB)}")
        d = self.LIB[name]
        self.name = name
        self.M = d["M"]
        self.geom = d["geom"]
        self.modes = list(d["modes"])
        self.n_base = self.NBASE[self.geom]
        self.R0 = float(R_override) if R_override is not None else R_UNIVERSAL / self.M
        self.theta_d = theta_d if theta_d is not None else d.get("theta_d")
        self.rho_d = rho_d if rho_d is not None else d.get("rho_d")

        model = model.lower()
        if gamma_const is not None:
            model = "cpg"
        if model == "idg" and (self.theta_d is None or self.rho_d is None):
            raise ValueError(f"no dissociation data for '{name}'; "
                             f"supply --theta-d and --rho-d or use --gas-model tpg")
        if model == "tpg" and not self.modes:
            model = "cpg"          # monatomic: gamma is exactly 5/3, constant
        self.model = model

        if gamma_const is not None:
            self.gamma_c = float(gamma_const)
        else:
            self.gamma_c = (self.n_base + 1.0) / self.n_base     # 5/3, 7/5, 4/3
        self.R = self.R0                       # nominal (frozen) gas constant

        # reference (stagnation) values, used for reporting and for the CPG
        # fallbacks inside the plotting/labelling code
        self.T0_ref = float(T0_ref)
        self.gamma_ref = float(np.asarray(self.gamma(np.array(T0_ref))).ravel()[0])
        self.R_ref = float(np.asarray(self.R_eff(np.array(T0_ref),
                                                 np.array(1.0))).ravel()[0])

    # ------------------------------------------------------------------ alpha
    def _K(self, T, rho):
        Tc = np.maximum(np.asarray(T, dtype=float), self.T_FLOOR)
        rr = np.maximum(np.asarray(rho, dtype=float), 1e-12)
        return (self.rho_d / rr) * np.exp(-np.minimum(self.theta_d / Tc, 700.0))

    def dalpha_dT(self, T, rho):
        """d(alpha)/dT at constant rho -- needed to invert e(T) reliably: the
        dissociation energy alpha*theta_d*R0 is comparable to 3 R0 T, so a
        Newton step that uses only the frozen cv diverges."""
        if self.model != "idg":
            return np.zeros_like(np.asarray(T, dtype=float))
        Tc = np.maximum(np.asarray(T, dtype=float), self.T_FLOOR)
        K = self._K(Tc, rho)
        rt = np.sqrt(np.maximum(K * K + 4.0 * K, 1e-300))
        dadK = 0.5 * (-1.0 + (K + 2.0) / rt)
        dKdT = K * self.theta_d / (Tc * Tc)
        return np.clip(dadK * dKdT, 0.0, 1e6)

    def cv_equilibrium(self, T, rho):
        """de/dT at constant rho, including the shift in composition."""
        if self.model != "idg":
            return self.cv(T, rho)
        return self.R0 * (3.0 + self.theta_d * self.dalpha_dT(T, rho))

    def alpha(self, T, rho):
        """Equilibrium dissociation fraction (0 unless model == 'idg')."""
        if self.model != "idg":
            return np.zeros_like(np.asarray(T, dtype=float))
        Tc = np.maximum(np.asarray(T, dtype=float), self.T_FLOOR)
        rr = np.maximum(np.asarray(rho, dtype=float), 1e-12)
        K = (self.rho_d / rr) * np.exp(-np.minimum(self.theta_d / Tc, 700.0))
        # alpha^2 + K alpha - K = 0
        return np.clip(0.5 * (-K + np.sqrt(K * K + 4.0 * K)), 0.0, 1.0)

    # -------------------------------------------------------- R, cv, cp, gamma
    def R_eff(self, T, rho=None):
        """Specific gas constant [J/kg/K].  Function of T (and rho) for 'idg'."""
        if self.model != "idg":
            return np.full_like(np.asarray(T, dtype=float), self.R0)
        return self.R0 * (1.0 + self.alpha(T, rho))

    def cv(self, T, rho=None):
        """Frozen specific heat at constant volume [J/kg/K]."""
        T = np.asarray(T, dtype=float)
        if self.model == "cpg":
            return np.full_like(T, self.R0 / (self.gamma_c - 1.0))
        if self.model == "idg":
            return np.full_like(T, 3.0 * self.R0)
        Tc = np.maximum(T, self.T_FLOOR)
        s = np.full_like(Tc, self.n_base)
        for th, g in self.modes:
            s = s + g * _fv_cv(th / Tc)
        return self.R0 * s

    def cp(self, T, rho=None):
        return self.cv(T, rho) + self.R_eff(T, rho)

    def gamma(self, T, rho=None):
        """Frozen ratio of specific heats gamma(T) = cp/cv."""
        T = np.asarray(T, dtype=float)
        if self.model == "cpg":
            return np.full_like(T, self.gamma_c)
        cv = self.cv(T, rho)
        return (cv + self.R_eff(T, rho)) / cv

    # ------------------------------------------------------------- e, h, T, s
    def e(self, T, rho=None):
        """Specific internal energy [J/kg] (zero at T = 0)."""
        T = np.asarray(T, dtype=float)
        if self.model == "cpg":
            return self.R0 / (self.gamma_c - 1.0) * T
        Tc = np.maximum(T, self.T_FLOOR)
        if self.model == "idg":
            return self.R0 * (3.0 * Tc + self.alpha(Tc, rho) * self.theta_d)
        s = self.n_base * Tc
        for th, g in self.modes:
            s = s + g * Tc * _fv_e(th / Tc)
        return self.R0 * s

    def h(self, T, rho=None):
        T = np.maximum(np.asarray(T, dtype=float), self.T_FLOOR)
        return self.e(T, rho) + self.R_eff(T, rho) * T

    def T_from_e(self, e, rho=None):
        """Invert e(T) by Newton iteration (quadratic; 8 passes is ample)."""
        e = np.maximum(np.asarray(e, dtype=float), self.E_FLOOR)
        if self.model == "cpg":
            return np.maximum(e * (self.gamma_c - 1.0) / self.R0, self.T_FLOOR)
        T = np.maximum(e / (3.5 * self.R0), self.T_FLOOR)
        for _ in range(20):
            f = self.e(T, rho) - e
            df = np.maximum(self.cv_equilibrium(T, rho), 1e-6)
            dT = np.clip(-f / df, -0.5 * T, 1.0 * T)      # damped, monotone
            Tn = np.maximum(T + dT, self.T_FLOOR)
            if np.all(np.abs(Tn - T) < 1e-8 * np.maximum(T, 1.0)):
                T = Tn
                break
            T = Tn
        return T

    def T_from_h(self, h, rho=None):
        h = np.maximum(np.asarray(h, dtype=float), self.E_FLOOR)
        T = np.maximum(h / (4.5 * self.R0), self.T_FLOOR)
        for _ in range(25):
            f = self.h(T, rho) - h
            df = np.maximum(self.cv_equilibrium(T, rho) + self.R_eff(T, rho), 1e-6)
            dT = np.clip(-f / df, -0.5 * T, 1.0 * T)
            Tn = np.maximum(T + dT, self.T_FLOOR)
            if np.all(np.abs(Tn - T) < 1e-8 * np.maximum(T, 1.0)):
                T = Tn
                break
            T = Tn
        return T

    def T_from_rho_p(self, rho, p):
        """Invert p = rho R_eff(T,rho) T for T."""
        rho = np.maximum(np.asarray(rho, dtype=float), 1e-12)
        p = np.maximum(np.asarray(p, dtype=float), 1e-12)
        if self.model != "idg":
            return np.maximum(p / (rho * self.R0), self.T_FLOOR)
        T = np.maximum(p / (rho * self.R0), self.T_FLOOR)
        for _ in range(30):                      # fixed point, monotone & safe
            Tn = np.maximum(p / (rho * self.R0 * (1.0 + self.alpha(T, rho))),
                            self.T_FLOOR)
            Tn = 0.5 * (T + Tn)                  # under-relaxed for stability
            if np.all(np.abs(Tn - T) < 1e-9 * np.maximum(T, 1.0)):
                return Tn
            T = Tn
        return T

    def entropy(self, T, rho):
        """Specific entropy [J/kg/K], arbitrary additive constant."""
        T = np.maximum(np.asarray(T, dtype=float), self.T_FLOOR)
        rho = np.maximum(np.asarray(rho, dtype=float), 1e-12)
        if self.model == "cpg" or self.model == "idg":
            cv = self.cv(T, rho)
            return cv * np.log(T) - self.R_eff(T, rho) * np.log(rho)
        s = self.n_base * np.log(T)
        for th, g in self.modes:
            s = s + g * _fv_s(th / T)
        return self.R0 * (s - np.log(rho))

    # ---------------------------------------------------------- wave speeds
    def sound_speed_Trho(self, T, rho):
        """Frozen sound speed a = sqrt(gamma R T)."""
        T = np.maximum(np.asarray(T, dtype=float), self.T_FLOOR)
        return np.sqrt(np.maximum(self.gamma(T, rho) * self.R_eff(T, rho) * T, 1e-12))

    def sound_speed(self, rho, p):
        T = self.T_from_rho_p(rho, p)
        return self.sound_speed_Trho(T, rho)

    # -------------------------------------------------------- state vectors
    def cons_to_prim(self, U):
        rho = np.maximum(U[0], 1e-9)
        u, v = U[1] / rho, U[2] / rho
        e = np.maximum(U[3] / rho - 0.5 * (u * u + v * v), self.E_FLOOR)
        T = self.T_from_e(e, rho)
        p = np.maximum(rho * self.R_eff(T, rho) * T, 1e-9)
        return np.stack([rho, u, v, p], axis=0)

    def prim_to_cons(self, W):
        rho = np.maximum(W[0], 1e-9)
        u, v, p = W[1], W[2], np.maximum(W[3], 1e-9)
        T = self.T_from_rho_p(rho, p)
        return np.stack([rho, rho * u, rho * v,
                         rho * self.e(T, rho) + 0.5 * rho * (u * u + v * v)], axis=0)

    def temperature(self, W):
        return self.T_from_rho_p(W[0], W[3])

    def mach(self, W):
        return np.sqrt(W[1] ** 2 + W[2] ** 2) / self.sound_speed(W[0], W[3])

    # --------------------------------------------------- stagnation (var. g)
    def stagnation(self, T, rho, V):
        """Isentropic bring-to-rest.  Returns (T0, p0, rho0) for arrays."""
        T = np.maximum(np.asarray(T, dtype=float), self.T_FLOOR)
        rho = np.maximum(np.asarray(rho, dtype=float), 1e-12)
        V = np.asarray(V, dtype=float)
        h0 = self.h(T, rho) + 0.5 * V * V
        T0 = self.T_from_h(h0, rho)
        # integrate ds = 0  ->  dln p/dln T = cp/R   with 8 log-T sub-steps
        lnT, lnT0 = np.log(T), np.log(T0)
        n = 8
        p = rho * self.R_eff(T, rho) * T
        lnp = np.log(np.maximum(p, 1e-30))
        for k in range(n):
            a = lnT + (lnT0 - lnT) * (k / n)
            b = lnT + (lnT0 - lnT) * ((k + 1) / n)
            Ta, Tb = np.exp(a), np.exp(b)
            ra = np.exp(lnp) / (self.R_eff(Ta, rho) * Ta)
            fa = self.cp(Ta, ra) / self.R_eff(Ta, ra)
            rb = np.exp(lnp) / (self.R_eff(Tb, rho) * Tb)
            fb = self.cp(Tb, rb) / self.R_eff(Tb, rb)
            lnp = lnp + 0.5 * (fa + fb) * (b - a)
        p0 = np.exp(lnp)
        rho0 = p0 / (self.R_eff(T0, rho) * T0)
        return T0, p0, rho0

    # ------------------------------------------------------------- reporting
    def describe(self):
        return {
            "gas": self.name,
            "model": self.model,
            "model_meaning": {"cpg": "calorically perfect (constant gamma, constant R)",
                              "tpg": "thermally perfect: gamma(T), R constant",
                              "idg": "ideal dissociating gas: gamma(T,rho), R(T,rho)"}[self.model],
            "molar_mass_kg_per_kmol": sig(self.M),
            "R0_J_per_kgK": sig(self.R0),
            "molecular_geometry": self.geom,
            "translational_rotational_dof_over_2": sig(self.n_base),
            "vibrational_modes_theta_K_and_degeneracy":
                [[sig(t), sig(g)] for t, g in self.modes],
            "theta_dissociation_K": sig(self.theta_d),
            "rho_dissociation_kg_per_m3": sig(self.rho_d),
            "cv_formula": ("cv/R = n_base + sum_i g_i x_i^2 e^{x_i}/(e^{x_i}-1)^2,"
                           " x_i = theta_i/T" if self.model == "tpg" else
                           ("cv = 3 R0 (Lighthill)" if self.model == "idg"
                            else "cv = R/(gamma-1), constant")),
            "R_formula": ("R_eff = R0 (1 + alpha(T,rho)),"
                          " alpha^2/(1-alpha) = (rho_d/rho) exp(-theta_d/T)"
                          if self.model == "idg" else "R = R0 = Ru/M, constant"),
            "gamma_at_T0_ref": sig(self.gamma_ref, 6),
            "R_at_T0_ref_J_per_kgK": sig(self.R_ref, 6),
            "T0_ref_K": sig(self.T0_ref),
        }


# =============================================================================
# 2.  QUASI-1D THEORY WITH VARIABLE gamma(T)  AND REAL-GAS SHOCK JUMPS
# =============================================================================
class Isentrope:
    """Tabulated isentrope through the stagnation state (T0, p0).

    For a thermally/calorically perfect gas the *normalised* isentrope
    (T/T0, p/p0, rho/rho0, M, A/A*) depends only on T0, not on p0, because
    cp and R are functions of T alone.  One table therefore serves every
    isentrope with the same stagnation temperature -- including the one
    downstream of a shock (same T0, smaller p0).  That is what makes the
    variable-gamma "shock inside the nozzle" algorithm cheap.

    Columns (all monotone in T, stored with T *descending* so that M ascends):
        T, T/T0, p/p0, rho/rho0, V, a, M, gamma, A/A*, nu (Prandtl-Meyer)
    """

    def __init__(self, gas, T0, p0=1.0, n=4000, T_min_frac=2.0e-3):
        self.gas, self.T0, self.p0_ref = gas, float(T0), float(p0)
        Tmin = max(gas.T_FLOOR * 1.5, T_min_frac * T0)
        T = np.geomspace(T0, Tmin, n)                       # descending
        lnT = np.log(T)

        # ds = 0  ->  d ln p / d ln T = cp/R   (trapezoid on the log-T grid).
        # For 'idg' the properties also depend on rho, so the integration and
        # the density are iterated to consistency (a no-op for cpg/tpg).
        rho = np.full_like(T, p0 / (gas.R0 * T0)) * (T0 / T)
        p_p0 = np.ones_like(T)
        for _ in range(6 if gas.model == "idg" else 1):
            R = gas.R_eff(T, rho)
            cp = gas.cp(T, rho)
            f = cp / R
            lnp = np.zeros_like(T)
            lnp[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(lnT))
            p_p0 = np.exp(lnp)                              # p/p0, starts at 1
            p_abs = p_p0 * p0
            for _ in range(20):                             # rho from p = rho R T
                rho_n = p_abs / (gas.R_eff(T, rho) * T)
                if np.allclose(rho_n, rho, rtol=1e-10):
                    rho = rho_n
                    break
                rho = 0.5 * (rho + rho_n)

        h0 = gas.h(np.array(T0), rho[0])
        h = gas.h(T, rho)
        V = np.sqrt(np.maximum(2.0 * (h0 - h), 0.0))
        a = gas.sound_speed_Trho(T, rho)
        M = V / a
        self.rho_abs, self.p_abs = rho, p_p0 * p0
        rho_rho0 = rho / rho[0]

        # A/A* from mass conservation: rho V A = const
        i_s = int(np.argmin(np.abs(M - 1.0)))
        # refine the sonic point by interpolation in M
        Gstar = np.interp(1.0, M, rho_rho0 * V)
        G = rho_rho0 * V
        with np.errstate(divide="ignore", invalid="ignore"):
            AR = np.where(G > 1e-30, Gstar / np.maximum(G, 1e-30), np.inf)
        AR[0] = np.inf

        # Prandtl-Meyer for a general gas:  nu = int_{M=1}^{M} sqrt(M^2-1) dV/V
        nu = np.zeros_like(M)
        sup = M >= 1.0
        idx = np.where(sup)[0]
        if idx.size > 1:
            Vs, Ms = V[idx], M[idx]
            integ = np.sqrt(np.maximum(Ms ** 2 - 1.0, 0.0))
            lnV = np.log(np.maximum(Vs, 1e-30))
            acc = np.zeros_like(Vs)
            acc[1:] = np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(lnV))
            nu[idx] = acc
        self.T, self.T_T0, self.p_p0 = T, T / T0, p_p0
        self.rho_rho0, self.V, self.a, self.M = rho_rho0, V, a, M
        self.gamma = gas.gamma(T, rho)
        self.R_tab = gas.R_eff(T, rho)
        self.AR, self.nu = AR, nu
        # Outgoing Riemann invariant along the isentrope, R- = V - 2a/(g-1).
        # Monotone increasing with M, so it inverts uniquely: this is what the
        # characteristic subsonic-inflow BC needs.
        self.Rminus = V - 2.0 * a / (self.gamma - 1.0)
        self.i_sonic = i_s
        self.M_max = float(M[-1])
        self.gamma_throat = float(np.interp(1.0, M, self.gamma))
        self.p_p0_throat = float(np.interp(1.0, M, p_p0))
        self.T_T0_throat = float(np.interp(1.0, M, T / T0))
        self.Gstar_over_rho0a0 = float(Gstar /
                                       gas.sound_speed_Trho(np.array(T0), rho[0]))

    # ---- interpolation helpers (M is ascending along the stored arrays) ----
    def _at_M(self, key, M):
        return float(np.interp(np.clip(M, 0.0, self.M_max), self.M,
                               getattr(self, key)))

    def p_ratio(self, M):    return self._at_M("p_p0", M)
    def T_ratio(self, M):    return self._at_M("T_T0", M)
    def rho_ratio(self, M):  return self._at_M("rho_rho0", M)
    def area_ratio(self, M): return self._at_M("AR", M)
    def gamma_at(self, M):   return self._at_M("gamma", M)
    def pm_angle(self, M):   return self._at_M("nu", M)

    def M_from_area(self, AR, branch="supersonic"):
        """Invert A/A*.  Sub- and supersonic branches handled separately."""
        AR = max(float(AR), 1.0)
        i = self.i_sonic
        if branch == "subsonic":
            Ms = self.M[:i + 1][::-1]          # M descending -> ascending after flip
            As = self.AR[:i + 1][::-1]         # AR ascending
            return float(np.interp(AR, As, Ms))
        Ms, As = self.M[i:], self.AR[i:]       # AR ascending with M
        return float(np.interp(AR, As, Ms))

    def M_from_p_ratio(self, pr):
        """Invert p/p0 (monotone decreasing with M)."""
        pr = min(max(float(pr), self.p_p0[-1]), 1.0)
        return float(np.interp(-pr, -self.p_p0, self.M))

    def state(self, M, p0, T0=None):
        """Dimensional static state on this isentrope at Mach M.

        Exact when p0 == the p0 the table was built with (always true for
        cpg/tpg, where the normalised isentrope is independent of p0).  For
        'idg' a different p0 (e.g. downstream of a shock) makes this an
        approximation, because alpha depends on density as well as on T."""
        T0 = self.T0 if T0 is None else T0
        T = self.T_ratio(M) * T0
        p = self.p_ratio(M) * p0
        rho = p / (self.gas.R0 * T)
        for _ in range(25):
            rn = p / (float(np.asarray(self.gas.R_eff(np.array(T),
                                                      np.array(rho))).ravel()[0]) * T)
            if abs(rn - rho) < 1e-12 * max(rho, 1e-30):
                rho = rn
                break
            rho = 0.5 * (rho + rn)
        a = float(np.asarray(self.gas.sound_speed_Trho(np.array(T), np.array(rho))).ravel()[0])
        return dict(M=float(M), T=float(T), p=float(p), rho=float(rho),
                    a=a, V=float(M) * a,
                    gamma=float(np.asarray(self.gas.gamma(np.array(T), np.array(rho))).ravel()[0]),
                    R=float(np.asarray(self.gas.R_eff(np.array(T), np.array(rho))).ravel()[0]))


# ---------------------------------------------------------------- shock jump
def normal_shock_real(gas, rho1, u1, p1):
    """Rankine-Hugoniot jump for the real (variable-gamma) gas.

    Solves, for v = rho1/rho2 in (0,1):
        u2 = v u1
        p2 = p1 + rho1 u1^2 (1 - v)                       (momentum)
        h2 = h1 + u1^2 (1 - v^2)/2                        (energy)
        p2 = rho2 R(T2) T2 with h2 = h(T2)                (EOS closure)
    Returns a dict with the full downstream state and the jump ratios.
    """
    rho1, u1, p1 = float(rho1), float(abs(u1)), float(p1)
    T1 = float(np.asarray(gas.T_from_rho_p(np.array(rho1), np.array(p1))).ravel()[0])
    h1 = float(np.asarray(gas.h(np.array(T1), np.array(rho1))).ravel()[0])
    a1 = float(np.asarray(gas.sound_speed_Trho(np.array(T1), np.array(rho1))).ravel()[0])
    M1 = u1 / a1
    if M1 <= 1.0 + 1e-9:                      # no shock possible
        return None

    def resid(v):
        rho2 = rho1 / max(v, 1e-9)
        p2 = p1 + rho1 * u1 * u1 * (1.0 - v)
        h2 = h1 + 0.5 * u1 * u1 * (1.0 - v * v)
        T2 = float(np.asarray(gas.T_from_h(np.array(h2), np.array(rho2))).ravel()[0])
        R2 = float(np.asarray(gas.R_eff(np.array(T2), np.array(rho2))).ravel()[0])
        return p2 - rho2 * R2 * T2

    g = float(np.asarray(gas.gamma(np.array(T1), np.array(rho1))).ravel()[0])
    v_pg = ((g - 1.0) * M1 ** 2 + 2.0) / ((g + 1.0) * M1 ** 2)      # first guess
    lo, hi = max(1e-5, 0.3 * v_pg), min(0.999999, max(1.7 * v_pg, 0.05))
    if resid(lo) * resid(hi) > 0.0:                                 # widen once
        vs = np.linspace(1e-4, 0.9999, 400)
        rs = np.array([resid(x) for x in vs])
        sc = np.where(np.sign(rs[1:]) != np.sign(rs[:-1]))[0]
        if sc.size == 0:
            return None
        k = sc[int(np.argmin(np.abs(0.5 * (vs[sc] + vs[sc + 1]) - v_pg)))]
        lo, hi = vs[k], vs[k + 1]
    v = bisect(resid, lo, hi, tol=1e-12)

    rho2 = rho1 / v
    u2 = v * u1
    p2 = p1 + rho1 * u1 * u1 * (1.0 - v)
    h2 = h1 + 0.5 * u1 * u1 * (1.0 - v * v)
    T2 = float(np.asarray(gas.T_from_h(np.array(h2), np.array(rho2))).ravel()[0])
    a2 = float(np.asarray(gas.sound_speed_Trho(np.array(T2), np.array(rho2))).ravel()[0])
    s1 = float(np.asarray(gas.entropy(np.array(T1), np.array(rho1))).ravel()[0])
    s2 = float(np.asarray(gas.entropy(np.array(T2), np.array(rho2))).ravel()[0])
    T01, p01, _ = gas.stagnation(np.array(T1), np.array(rho1), np.array(u1))
    T02, p02, _ = gas.stagnation(np.array(T2), np.array(rho2), np.array(u2))
    return dict(M1=M1, M2=u2 / a2, rho1=rho1, rho2=rho2, u1=u1, u2=u2,
                p1=p1, p2=p2, T1=T1, T2=T2, a1=a1, a2=a2,
                p2_p1=p2 / p1, rho2_rho1=rho2 / rho1, T2_T1=T2 / T1,
                u2_u1=u2 / u1, ds_J_per_kgK=s2 - s1,
                p02_p01=float(p02) / float(p01),
                T01=float(T01), T02=float(T02), p01=float(p01), p02=float(p02),
                gamma1=float(np.asarray(gas.gamma(np.array(T1), np.array(rho1))).ravel()[0]),
                gamma2=float(np.asarray(gas.gamma(np.array(T2), np.array(rho2))).ravel()[0]),
                R1=float(np.asarray(gas.R_eff(np.array(T1), np.array(rho1))).ravel()[0]),
                R2=float(np.asarray(gas.R_eff(np.array(T2), np.array(rho2))).ravel()[0]))


def oblique_from_normal(jump, beta):
    """Given a normal-shock solution for the normal component and the wave
    angle beta, return the deflection angle and the downstream Mach number.
    Valid for any equation of state: tan(beta-theta) = (rho1/rho2) tan(beta)."""
    if jump is None:
        return None
    r = jump["rho1"] / jump["rho2"]
    theta = beta - math.atan(r * math.tan(beta))
    Mn2 = jump["M2"]
    M2 = Mn2 / max(math.sin(beta - theta), 1e-9)
    return dict(beta_deg=math.degrees(beta), theta_deg=math.degrees(theta),
                M2_oblique=M2)


# ------------------------------------------------- shock inside the divergent
def exit_state_for_shock(iso, gas, p0, T0, As_At, Ae_At):
    """Normal shock at area ratio As_At; returns the exit state (real gas)."""
    M1 = iso.M_from_area(As_At, "supersonic")
    st1 = iso.state(M1, p0, T0)
    jump = normal_shock_real(gas, st1["rho"], st1["V"], st1["p"])
    if jump is None:
        return None
    M2 = jump["M2"]
    Astar2_At = As_At / max(iso.area_ratio(M2), 1e-12)
    p02 = jump["p02"]
    ARe2 = Ae_At / max(Astar2_At, 1e-12)
    Me = iso.M_from_area(max(ARe2, 1.0), "subsonic")
    pe = iso.p_ratio(Me) * p02
    return dict(M1=M1, M2=M2, Me=Me, pe=pe, pe_p0=pe / p0, p02=p02,
                p02_p01=jump["p02_p01"], Astar2_At=Astar2_At, jump=jump,
                state1=st1)


def find_shock_area(iso, gas, p0, T0, Ae_At, pb):
    """Locate A_shock/A_t so that the 1D exit pressure equals pb.
    Returns (As_At or None, pe_design, pe_shock_at_exit, pe_subsonic)."""
    st_thr = exit_state_for_shock(iso, gas, p0, T0, 1.0 + 1e-7, Ae_At)
    st_exi = exit_state_for_shock(iso, gas, p0, T0, Ae_At, Ae_At)
    pe_sub = st_thr["pe"] if st_thr else None          # shock at the throat
    pe_2nd = st_exi["pe"] if st_exi else None          # shock at the exit plane
    pe_des = iso.p_ratio(iso.M_from_area(Ae_At, "supersonic")) * p0
    if pe_sub is None or pe_2nd is None:
        return None, pe_des, pe_2nd, pe_sub
    if pb >= pe_sub or pb <= pe_2nd:
        return None, pe_des, pe_2nd, pe_sub
    f = lambda A: exit_state_for_shock(iso, gas, p0, T0, A, Ae_At)["pe"] - pb
    As = bisect(f, 1.0 + 1e-7, Ae_At, tol=1e-10)
    return As, pe_des, pe_2nd, pe_sub


def npr_regimes(iso, gas, p0, T0, Ae_At, p_back):
    """Everything needed to decide WHETHER and WHERE a shock is initiated."""
    Me_sup = iso.M_from_area(Ae_At, "supersonic")
    Me_sub = iso.M_from_area(Ae_At, "subsonic")
    pe_design = iso.p_ratio(Me_sup) * p0                 # 3rd critical
    pe_1st = iso.p_ratio(Me_sub) * p0                    # 1st critical (sonic throat)
    st_exi = exit_state_for_shock(iso, gas, p0, T0, Ae_At, Ae_At)
    pe_2nd = st_exi["pe"] if st_exi else None            # 2nd critical
    As, _, _, _ = find_shock_area(iso, gas, p0, T0, Ae_At, p_back)

    if p_back >= p0:
        regime = "no flow (pb >= p0)"
    elif p_back > pe_1st:
        regime = "subsonic throughout (venturi, throat not choked)"
    elif abs(p_back - pe_1st) / p0 < 1e-6:
        regime = "first critical: throat exactly sonic, exit subsonic"
    elif As is not None:
        regime = "normal shock inside the divergent section"
    elif abs(p_back - pe_design) / max(pe_design, 1e-30) < 2e-3:
        regime = "design (perfectly expanded), shock free"
    elif pe_2nd is not None and p_back > pe_2nd:
        regime = "normal shock at / just outside the exit plane"
    elif p_back > pe_design:
        regime = "overexpanded: oblique shock train outside the nozzle"
    else:
        regime = "underexpanded: expansion fan, barrel shock and Mach disc"
    return dict(
        NPR=p0 / max(p_back, 1e-30),
        NPR_design=p0 / max(pe_design, 1e-30),
        NPR_first_critical=p0 / max(pe_1st, 1e-30),
        NPR_second_critical=(p0 / pe_2nd) if pe_2nd else None,
        p_exit_design_Pa=pe_design, p_exit_first_critical_Pa=pe_1st,
        p_exit_second_critical_Pa=pe_2nd, p_back_Pa=p_back,
        pb_over_p0=p_back / p0, pe_design_over_pb=pe_design / max(p_back, 1e-30),
        Me_design=Me_sup, Me_subsonic_branch=Me_sub,
        shock_in_nozzle_area_ratio=As,
        shock_in_nozzle_window_pb_p0=[pe_2nd / p0 if pe_2nd else None,
                                      pe_1st / p0],
        regime=regime)


def theory_1d(mesh, gas, cfg, iso):
    """Quasi-1D reference solution along the nozzle with variable gamma."""
    AR = mesh.area_ratio_cells()
    x = mesh.xc[0, :]
    noz = ~mesh.plume_cols if mesh.nx_plume > 0 else np.ones(mesh.nx, bool)
    Ae_At = mesh.Ae_At
    p0, T0 = cfg["p0"], cfg["T0"]

    reg = npr_regimes(iso, gas, p0, T0, Ae_At, cfg["pb"])
    if cfg.get("outlet") == "supersonic" and mesh.nx_plume == 0:
        reg["regime"] = ("supersonic outlet BC: the exit is extrapolated, so no "
                         "back pressure is felt and the nozzle is forced to the "
                         "choked, fully expanded solution (no shock possible)")
        reg["back_pressure_is_felt"] = False
    else:
        reg["back_pressure_is_felt"] = True
    As = reg["shock_in_nozzle_area_ratio"]
    if mesh.nx_plume > 0:
        As = None                       # with a plume the exit is not the BC
    shock = {"x_shock": None, "regime": reg["regime"]}

    M = np.full(mesh.nx, np.nan)
    p = np.full(mesh.nx, np.nan)
    T = np.full(mesh.nx, np.nan)
    if As is None:
        for i in np.where(noz)[0]:
            M[i] = iso.M_from_area(AR[i], "subsonic" if x[i] < 0 else "supersonic")
            p[i] = iso.p_ratio(M[i]) * p0
            T[i] = iso.T_ratio(M[i]) * T0
    else:
        st = exit_state_for_shock(iso, gas, p0, T0, As, Ae_At)
        y_s = (math.sqrt(As) if mesh.axisym else As) * mesh.h_t
        hc = 0.5 * (mesh.hn[:-1] + mesh.hn[1:])
        d = x >= 0.0
        x_sh = float(np.interp(y_s, hc[d], x[d]))
        shock.update(As_At=As, x_shock=x_sh, M1=st["M1"], M2=st["M2"],
                     Me=st["Me"], p02_p01=st["p02_p01"],
                     Astar2_At=st["Astar2_At"], jump=st["jump"])
        p02 = st["p02"]
        for i in np.where(noz)[0]:
            if x[i] < 0.0:
                M[i] = iso.M_from_area(AR[i], "subsonic"); pt = p0
            elif x[i] <= x_sh:
                M[i] = iso.M_from_area(AR[i], "supersonic"); pt = p0
            else:
                M[i] = iso.M_from_area(max(AR[i] / st["Astar2_At"], 1.0),
                                       "subsonic"); pt = p02
            p[i] = iso.p_ratio(M[i]) * pt
            T[i] = iso.T_ratio(M[i]) * T0
    p0_post = p0 if As is None else exit_state_for_shock(
        iso, gas, p0, T0, As, Ae_At)["p02"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.full_like(p, np.nan)
        a = np.full_like(p, np.nan)
        u = np.full_like(p, np.nan)
        ok = np.isfinite(M)
        for i in np.where(ok)[0]:
            pt = p0 if (As is None or x[i] <= shock.get("x_shock", 1e30)) \
                else p0_post
            st = iso.state(M[i], pt, T0)
            rho[i], a[i], u[i] = st["rho"], st["a"], st["V"]
    return dict(x=x, M=M, p=p, T=T, rho=rho, u=u, noz=noz,
                shock=shock, regimes=reg, AR=AR)


# =============================================================================
# 3.  MESH  --  body fitted, numbered, conservative axisymmetric metrics
# =============================================================================
class Mesh:
    """Structured body-fitted mesh with global element numbering.

    axisym=True : r runs 0..h(z); the bottom boundary is the SYMMETRY AXIS
                  (a slip wall is exact there for inviscid flow); A/At=(r/rt)^2
    axisym=False: planar duct, y runs -h..+h;  A/At = h/ht
    plume       : appends a downstream block whose outer radius opens to Hmax

    NUMBERING (this is what the JSON "grid" block lists, and the order every
    other cell-wise array in the JSON follows):
        cell_id = j * nx + i      j = transverse index 0..ny-1 (0 = axis/lower wall)
                                  i = axial  index 0..nx-1 (0 = inlet)
        node_id = jn * (nx+1) + in
    which is exactly numpy C-order ravel() of an (ny, nx) array, so the export
    can simply .ravel() every field.

    AXISYMMETRIC FORM.  The Euler equations are multiplied by r and integrated
    over the (z,r) cell:
        d/dt (r_c U A) + sum_faces r_f (F.n) L = (0, 0, p A, 0)
    This is strictly conservative and needs no 1/r source, so it stays well
    behaved on the axis (the axis face has r_f = 0 -> zero flux, exactly).
    """

    def __init__(self, nx_noz=144, ny=64, nx_plume=0,
                 Lc=1.0, Ld=2.0, Lp=0.0, h_in=1.2, h_t=0.6, h_e=1.1,
                 Hmax=2.0, profile="cosine", cluster=1.6, axisym=True,
                 theta_n=22.0, theta_e=8.0, cluster_mode="throat",
                 se_n=3.0, se_p=2.0, se_trunc=1.0, plume_ramp=None):
        self.plume_ramp = (None if plume_ramp is None else float(plume_ramp))
        self.nx_noz, self.ny, self.nx_plume = int(nx_noz), int(ny), int(nx_plume)
        self.Lc, self.Ld, self.Lp = float(Lc), float(Ld), float(Lp)
        self.h_in, self.h_t, self.h_e = float(h_in), float(h_t), float(h_e)
        self.Hmax, self.profile, self.cluster = float(Hmax), profile, float(cluster)
        self.cluster_mode = cluster_mode
        self.axisym = bool(axisym)
        self.theta_n, self.theta_e = float(theta_n), float(theta_e)
        self.se_n, self.se_p = float(se_n), float(se_p)
        self.se_trunc = float(se_trunc)
        self.se_exit_angle_deg = None
        self.theta_e_actual = float(theta_e)
        self.theta_n_actual = float(theta_n)
        self.nx = self.nx_noz + self.nx_plume

        xn = self._x_nodes(self.nx_noz, self.Lc, self.Ld, self.cluster,
                           self.cluster_mode)
        if self.nx_plume > 0:
            dx0 = xn[-1] - xn[-2]
            q = bisect(lambda q: dx0 * (q ** self.nx_plume - 1) / (q - 1) - self.Lp,
                       1.000001, 1.5)
            xp, d = [self.Ld], dx0
            for _ in range(self.nx_plume):
                d *= q
                xp.append(xp[-1] + d)
            xp = np.array(xp[1:])
            xp = self.Ld + (xp - self.Ld) * (self.Lp / (xp[-1] - self.Ld))
            xn = np.concatenate([xn, xp])

        hn = self._wall(xn)
        if self.nx_plume > 0:                    # open the outer boundary smoothly
            pl = xn > self.Ld + 1e-12
            Lr = 0.5 * self.Lp if self.plume_ramp is None else self.plume_ramp
            ramp = np.clip((xn[pl] - self.Ld) / max(Lr, 1e-9), 0.0, 1.0)
            hn[pl] = self.h_e + (self.Hmax - self.h_e) * 0.5 * (1.0 - np.cos(np.pi * ramp))

        eta = np.linspace(0.0, 1.0, self.ny + 1)
        self.xn, self.hn = xn, hn
        self.Xn = np.broadcast_to(xn, (self.ny + 1, len(xn))).copy()
        self.Yn = (eta[:, None] * hn[None, :] if self.axisym
                   else (2.0 * eta[:, None] - 1.0) * hn[None, :])
        self._metrics()
        self.plume_cols = self.xc[0, :] > self.Ld + 1e-12
        self.lip = (int(np.where(~self.plume_cols)[0][-1]) if self.nx_plume > 0
                    else self.nx - 1)
        self.i_throat = int(np.argmin(np.abs(self.xc[0, :])))
        self.Ae_At = ((self.h_e / self.h_t) ** 2 if self.axisym
                      else self.h_e / self.h_t)
        self.A_throat = (np.pi * self.h_t ** 2 if self.axisym else 2.0 * self.h_t)
        self.A_exit = (np.pi * self.h_e ** 2 if self.axisym else 2.0 * self.h_e)
        self._plume_mesh_quality()
        self._numbering()


    def _plume_mesh_quality(self):
        """Cell sizes in the external plume block, measured after the length
        rescale.  A long plume on a fixed cell count stretches geometrically,
        and once the cells are much larger than a shock-cell spacing the
        downstream diamonds are smeared rather than resolved."""
        self.plume_dx_first = self.plume_dx_last = self.plume_growth = None
        if self.nx_plume <= 0:
            return
        d = np.diff(self.xn)[-self.nx_plume:]
        self.plume_dx_first = float(d[0])
        self.plume_dx_last = float(d[-1])
        self.plume_growth = float((d[-1] / max(d[0], 1e-30))
                                  ** (1.0 / max(self.nx_plume - 1, 1)))

    # ---------------------------------------------------------------- shapes
    @staticmethod
    def _x_nodes(n, Lc, Ld, cluster, mode="throat"):
        """Axial node distribution.

        mode="throat" (default): two one-sided tanh stretchings that BOTH
        refine towards z = 0.  This is what a nozzle needs -- the throat is
        where the flow chokes and where the wall curvature is largest.

        mode="ends": the original single tanh,
            s2 = (1 + tanh(c(2s-1))/tanh c)/2,
        which despite being documented as "throat clustering" actually refines
        the INLET and the OUTLET and leaves the coarsest cells in the middle of
        the divergent section (at nx=80, Lc=1, Ld=2 the throat cell was ~5x the
        inlet cell).  The entropy produced there was large enough to stop the
        nozzle choking at back-pressure ratios where 1D theory says it must.
        Kept only for reproducing old results.
        """
        if mode == "ends":
            s = np.linspace(0.0, 1.0, n + 1)
            s2 = 0.5 * (1.0 + np.tanh(cluster * (2.0 * s - 1.0)) / np.tanh(cluster))
            return -Lc + (Lc + Ld) * s2
        c = max(float(cluster), 1e-3)
        nc = int(round(n * Lc / (Lc + Ld)))
        nc = min(max(nc, 2), n - 2)
        nd = n - nc
        tc = np.linspace(0.0, 1.0, nc + 1)          # convergent: fine at z = 0
        xc = -Lc * (1.0 - np.tanh(c * tc) / np.tanh(c))
        td = np.linspace(0.0, 1.0, nd + 1)          # divergent: fine at z = 0
        xd = Ld * (1.0 - np.tanh(c * (1.0 - td)) / np.tanh(c))
        return np.concatenate([xc, xd[1:]])

    def _ellipse_k(self, dh):
        """k = L_div/A for the general ellipse, from the requested exit angle."""
        te = math.radians(min(max(self.theta_e, 0.1), 89.0))
        tan_min = 2.0 * dh / self.Ld                      # k -> 0 limit
        target = max(math.tan(te), tan_min * 1.000001)

        def f(k):
            c = math.sqrt(max(1.0 - k * k, 1e-16))
            return dh * k * k / (self.Ld * c * max(1.0 - c, 1e-16)) - target
        k = bisect(f, 1e-6, 1.0 - 1e-9, tol=1e-14)
        c = math.sqrt(max(1.0 - k * k, 1e-16))
        self.theta_e_actual = math.degrees(math.atan(
            dh * k * k / (self.Ld * c * max(1.0 - c, 1e-16))))
        return k

    def _bezier_bell(self, z):
        """Quadratic-Bezier bell (Rao-type approximation).

        Control point P1 = intersection of the line leaving the throat at
        theta_n with the line arriving at the lip at theta_e:
            x1 = (dh - tan(theta_e) L_div) / (tan(theta_n) - tan(theta_e))
        The mean wall angle is atan(dh/L_div), so a physical bell needs
            theta_e < atan(dh/L_div) < theta_n
        otherwise P1 lies above h_e, the quadratic turns back down and the
        exit wall angle comes out NEGATIVE.  Both angles are clamped into the
        admissible range and the achieved values are reported.
        """
        dh = self.h_e - self.h_t
        mean_deg = math.degrees(math.atan(dh / self.Ld))
        tn_deg = min(max(self.theta_n, mean_deg + 0.5), 80.0)
        te_deg = min(max(self.theta_e, 0.0), mean_deg - 0.5)
        if te_deg < 0.0:
            te_deg = 0.0
            tn_deg = max(tn_deg, mean_deg + 0.5)
        self.theta_n_actual, self.theta_e_actual = tn_deg, te_deg
        tn, te = math.radians(tn_deg), math.radians(te_deg)
        x1 = (dh - math.tan(te) * self.Ld) / max(math.tan(tn) - math.tan(te), 1e-9)
        x1 = min(max(x1, 1e-6), self.Ld - 1e-6)
        h1 = self.h_t + math.tan(tn) * x1
        t = np.linspace(0.0, 1.0, 4001)
        xb = (1 - t) ** 2 * 0.0 + 2 * t * (1 - t) * x1 + t ** 2 * self.Ld
        hb = (1 - t) ** 2 * self.h_t + 2 * t * (1 - t) * h1 + t ** 2 * self.h_e
        o = np.argsort(xb)
        return np.interp(z, xb[o], hb[o])

    def _wall(self, z):
        h = np.empty_like(z)
        conv = z < 0.0
        xi = (z[conv] + self.Lc) / self.Lc
        h[conv] = self.h_t + (self.h_in - self.h_t) * 0.5 * (1.0 + np.cos(np.pi * xi))
        d = ~conv
        xd = np.clip(z[d], 0.0, self.Ld)
        xr = xd / self.Ld
        dh = self.h_e - self.h_t
        if self.profile == "conical":
            h[d] = self.h_t + dh * xr
        elif self.profile == "cosine":                   # parallel at both ends
            h[d] = self.h_t + dh * 0.5 * (1.0 - np.cos(np.pi * xr))
        elif self.profile == "bell":
            h[d] = self._bezier_bell(xd)
        elif self.profile == "ellipse":
            # GENERAL ellipse arc: zero slope at the throat (as intended) and a
            # FINITE, prescribed exit wall angle.
            #     h(x) = H - B sqrt(1 - (x/A)^2),   H - B = h_t
            # The original used A = L_div exactly, i.e. a full quarter ellipse,
            # whose slope is INFINITE at the exit lip (60 deg already at
            # xr = 0.99).  Taking A > L_div uses only part of the arc and leaves
            # a finite exit angle.  With k = L_div/A and c = sqrt(1-k^2):
            #     B = dh/(1-c),   tan(theta_e) = dh k^2 / (L_div c (1-c))
            # The smallest attainable exit angle is atan(2 dh / L_div) (the
            # k -> 0 parabola limit); a smaller request is clamped to it and
            # reported in Mesh.theta_e_actual.
            k = self._ellipse_k(dh)
            A = self.Ld / k
            c = math.sqrt(max(1.0 - k * k, 0.0))
            B = dh / max(1.0 - c, 1e-12)
            H = self.h_t + B
            h[d] = H - B * np.sqrt(np.clip(1.0 - (xd / A) ** 2, 0.0, 1.0))
        elif self.profile == "ellipse-legacy":
            # exact reproduction of the original (vertical wall at the lip)
            h[d] = self.h_e - dh * np.sqrt(np.clip(1.0 - xr * xr, 0.0, 1.0))
        elif self.profile == "super-ellipse":
            # Super-ellipse (Lame curve), the contour from the MATLAB script:
            #     y = y_e + (y_t - y_e) (1 - (x/L_d)^p)^(1/n)      p = 2, n = 3
            # Written here with the throat and the exit pinned explicitly so
            # that a truncation still lands on both end radii:
            #     c  = (1 - trunc^p)^(1/n)
            #     B  = dh / (1 - c),   H = h_t + B
            #     u  = trunc * x/L_d
            #     h  = H - B (1 - u^p)^(1/n)
            # trunc = 1 reproduces the MATLAB equation exactly (and, like it,
            # is VERTICAL at the lip: the slope carries a factor
            # (1-u^p)^(1/n - 1) which diverges as u -> 1).  The MATLAB script
            # sidesteps that by evaluating only to 0.998*L_d, which also drops
            # the exit radius below y_e; setting --se-trunc 0.998 here keeps
            # both end radii exact and leaves a finite exit angle instead.
            p, nn = self.se_p, self.se_n
            tr = min(max(self.se_trunc, 1e-6), 1.0)
            c = (1.0 - tr ** p) ** (1.0 / nn)
            B = dh / max(1.0 - c, 1e-12)
            H = self.h_t + B
            u = tr * xr
            h[d] = H - B * (1.0 - np.clip(u, 0.0, 1.0) ** p) ** (1.0 / nn)
            self.se_exit_angle_deg = math.degrees(math.atan(
                B * p * tr ** p / (nn * self.Ld * max(c, 1e-12) ** (nn - 1.0))
            )) if c > 1e-9 else 90.0
        else:
            raise ValueError(f"unknown profile '{self.profile}'")
        return h

    # --------------------------------------------------------------- metrics
    def _metrics(self):
        Z, R = self.Xn, self.Yn
        x1, y1 = Z[:-1, :-1], R[:-1, :-1]
        x2, y2 = Z[:-1, 1:], R[:-1, 1:]
        x3, y3 = Z[1:, 1:], R[1:, 1:]
        x4, y4 = Z[1:, :-1], R[1:, :-1]
        cr = ((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) +
              (x3 * y4 - x4 * y3) + (x4 * y1 - x1 * y4))
        self.area = 0.5 * np.abs(cr)                       # planar cell area
        # exact polygon centroid (2nd-order accurate cell centre)
        sgn = np.where(cr >= 0, 1.0, -1.0)
        c1 = x1 * y2 - x2 * y1
        c2 = x2 * y3 - x3 * y2
        c3 = x3 * y4 - x4 * y3
        c4 = x4 * y1 - x1 * y4
        A6 = np.where(np.abs(cr) < 1e-30, 1e-30, cr) * 3.0
        self.xc = ((x1 + x2) * c1 + (x2 + x3) * c2 +
                   (x3 + x4) * c3 + (x4 + x1) * c4) / A6
        self.yc = ((y1 + y2) * c1 + (y2 + y3) * c2 +
                   (y3 + y4) * c3 + (y4 + y1) * c4) / A6
        _ = sgn

        dx, dy = Z[1:, :] - Z[:-1, :], R[1:, :] - R[:-1, :]
        L = np.sqrt(dx ** 2 + dy ** 2)
        self.i_nx, self.i_ny, self.i_len = dy / L, -dx / L, L        # +z faces
        self.i_y = 0.5 * (R[:-1, :] + R[1:, :])
        dx, dy = Z[:, 1:] - Z[:, :-1], R[:, 1:] - R[:, :-1]
        L = np.sqrt(dx ** 2 + dy ** 2)
        self.j_nx, self.j_ny, self.j_len = -dy / L, dx / L, L        # +r faces
        self.j_y = 0.5 * (R[:, :-1] + R[:, 1:])

        if self.axisym:
            self.i_area = self.i_len * np.abs(self.i_y)      # r-weighted (/2pi)
            self.j_area = self.j_len * np.abs(self.j_y)
            self.vol = self.area * np.abs(self.yc)
            self.i_area_phys = self.i_len * 2.0 * np.pi * np.abs(self.i_y)
            self.j_area_phys = self.j_len * 2.0 * np.pi * np.abs(self.j_y)
            self.vol_phys = self.area * 2.0 * np.pi * np.abs(self.yc)
        else:
            self.i_area = self.i_len.copy()
            self.j_area = self.j_len.copy()
            self.vol = self.area.copy()
            self.i_area_phys = self.i_len.copy()
            self.j_area_phys = self.j_len.copy()
            self.vol_phys = self.area.copy()

        # local cell sizes (used by the shock sensor and the resolution report)
        self.dx_cell = 0.5 * (self.i_len[:, 1:] + self.i_len[:, :-1]) * 0.0 + \
            np.abs(self.Xn[:-1, 1:] - self.Xn[:-1, :-1])
        self.dy_cell = np.abs(self.Yn[1:, :-1] - self.Yn[:-1, :-1])
        self.h_cell = np.sqrt(np.maximum(self.area, 1e-30))

    # ------------------------------------------------------------- numbering
    def _numbering(self):
        ny, nx = self.ny, self.nx
        j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        self.cell_i, self.cell_j = i, j
        self.cell_id = j * nx + i                     # == ravel order, C-order
        zone = np.empty((ny, nx), dtype=object)
        xc = self.xc
        zone[:] = "divergent"
        zone[xc < 0.0] = "convergent"
        zone[:, self.i_throat] = "throat"          # the minimum-area column
        if self.nx_plume > 0:
            zone[:, self.plume_cols] = "plume"
        self.zone = zone
        self.n_cells = ny * nx
        self.n_nodes = (ny + 1) * (nx + 1)
        # Rows that represent the CENTRELINE.  In axisymmetric mode j = 0 is
        # the symmetry axis.  In planar mode the channel spans -h..+h, so j = 0
        # is the LOWER WALL and the centreline y = 0 falls on a cell face: the
        # two rows either side of it must be averaged.  (Reading j = 0 in
        # planar mode samples the wall / outer shear layer instead of the jet.)
        if self.axisym:
            self.centre_rows = [0]
        else:
            jm = ny // 2
            self.centre_rows = ([jm - 1, jm] if (ny % 2 == 0 and jm >= 1)
                                else [jm])

    def area_ratio_cells(self):
        h = 0.5 * (self.hn[:-1] + self.hn[1:])
        return (h / self.h_t) ** 2 if self.axisym else h / self.h_t

    def summary(self):
        return (f"Mesh {self.ny}x{self.nx} ({'axisym' if self.axisym else 'planar'}, "
                f"{self.profile}) | nozzle {self.nx_noz} + plume {self.nx_plume} | "
                f"cells {self.n_cells} | Ae/At={self.Ae_At:.3f} | "
                f"x=[{self.xn[0]:.2f},{self.xn[-1]:.2f}]")


# =============================================================================
# 4.  RECONSTRUCTION  (2nd-order MUSCL, minmod, bounds preserving)
# =============================================================================
def minmod(a, b):
    return np.where(a * b <= 0.0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


def _floor(W):
    W = W.copy()
    W[0] = np.maximum(W[0], 1e-8)
    W[3] = np.maximum(W[3], 1.0)
    return W


def _clamp(WL, WR, lo, hi):
    return (np.minimum(np.maximum(WL, lo), hi),
            np.minimum(np.maximum(WR, lo), hi))


def reconstruct(Wg, axis):
    """MUSCL face states along 'axis' (2 = z, 1 = r) of the padded array."""
    if axis == 2:
        W = Wg[:, 1:-1, :]
        s = minmod(W[:, :, 1:-1] - W[:, :, :-2], W[:, :, 2:] - W[:, :, 1:-1])
        Wc = W[:, :, 1:-1]
        L = np.concatenate([Wg[:, 1:-1, 0][:, :, None],
                            Wc[:, :, :-1] + 0.5 * s[:, :, :-1],
                            Wc[:, :, -1][:, :, None]], axis=2)
        R = np.concatenate([Wc[:, :, 0][:, :, None],
                            Wc[:, :, 1:] - 0.5 * s[:, :, 1:],
                            Wg[:, 1:-1, -1][:, :, None]], axis=2)
        cL, cR = Wg[:, 1:-1, :-1], Wg[:, 1:-1, 1:]
    else:
        W = Wg[:, :, 1:-1]
        s = minmod(W[:, 1:-1, :] - W[:, :-2, :], W[:, 2:, :] - W[:, 1:-1, :])
        Wc = W[:, 1:-1, :]
        L = np.concatenate([Wg[:, 0, 1:-1][:, None, :],
                            Wc[:, :-1, :] + 0.5 * s[:, :-1, :],
                            Wc[:, -1, :][:, None, :]], axis=1)
        R = np.concatenate([Wc[:, 0, :][:, None, :],
                            Wc[:, 1:, :] - 0.5 * s[:, 1:, :],
                            Wg[:, -1, 1:-1][:, None, :]], axis=1)
        cL, cR = Wg[:, :-1, 1:-1], Wg[:, 1:, 1:-1]
    L, R = _clamp(L, R, np.minimum(cL, cR), np.maximum(cL, cR))
    return _floor(L), _floor(R)


# =============================================================================
# 5.  HLLC RIEMANN SOLVER  (real gas, corrected Roe average)
# =============================================================================
def hllc(WL, WR, nx, ny, gas, w_hll=None):
    rL, uL, vL, pL = np.maximum(WL[0], 1e-9), WL[1], WL[2], np.maximum(WL[3], 1e-9)
    rR, uR, vR, pR = np.maximum(WR[0], 1e-9), WR[1], WR[2], np.maximum(WR[3], 1e-9)
    tx, ty = -ny, nx
    unL, utL = uL * nx + vL * ny, uL * tx + vL * ty
    unR, utR = uR * nx + vR * ny, uR * tx + vR * ty

    TL = gas.T_from_rho_p(rL, pL)
    TR = gas.T_from_rho_p(rR, pR)
    aL = gas.sound_speed_Trho(TL, rL)
    aR = gas.sound_speed_Trho(TR, rR)
    EL = rL * gas.e(TL, rL) + 0.5 * rL * (unL ** 2 + utL ** 2)
    ER = rR * gas.e(TR, rR) + 0.5 * rR * (unR ** 2 + utR ** 2)

    # ---- Roe average (the tangential kinetic energy must be included, else
    #      a_roe is over-estimated and the scheme becomes far too dissipative)
    sL, sR = np.sqrt(rL), np.sqrt(rR)
    w = 1.0 / (sL + sR)
    un_roe = (sL * unL + sR * unR) * w
    ut_roe = (sL * utL + sR * utR) * w
    H_roe = (sL * (EL + pL) / rL + sR * (ER + pR) / rR) * w
    g_roe = 0.5 * (gas.gamma(TL, rL) + gas.gamma(TR, rR))
    a_roe = np.sqrt(np.maximum((g_roe - 1.0) *
                               (H_roe - 0.5 * (un_roe ** 2 + ut_roe ** 2)), 1e-12))
    SL = np.minimum(unL - aL, un_roe - a_roe)
    SR = np.maximum(unR + aR, un_roe + a_roe)

    den = rL * (SL - unL) - rR * (SR - unR)
    Sst = (pR - pL + rL * unL * (SL - unL) - rR * unR * (SR - unR)) / \
          np.where(np.abs(den) < 1e-30, 1e-30, den)

    def safe(d):
        return np.where(d >= 0, 1.0, -1.0) * np.maximum(np.abs(d), 1e-12)

    def F(r, un, ut, p, E):
        return np.stack([r * un, r * un * un + p, r * un * ut, (E + p) * un], axis=0)

    def Ustar(r, un, ut, p, E, S):
        c = r * (S - un) / safe(S - Sst)
        return np.stack([c, c * Sst, c * ut,
                         c * (E / r + (Sst - un) *
                              (Sst + p / safe(r * (S - un))))], axis=0)

    UL = np.stack([rL, rL * unL, rL * utL, EL], axis=0)
    UR = np.stack([rR, rR * unR, rR * utR, ER], axis=0)
    FL, FR = F(rL, unL, utL, pL, EL), F(rR, unR, utR, pR, ER)
    FsL = FL + SL * (Ustar(rL, unL, utL, pL, EL, SL) - UL)
    FsR = FR + SR * (Ustar(rR, unR, utR, pR, ER, SR) - UR)

    cond = [SL >= 0, (SL < 0) & (Sst >= 0), (Sst < 0) & (SR > 0), SR <= 0]
    Fn = np.stack([np.select(cond, [FL[k], FsL[k], FsR[k], FR[k]], default=FR[k])
                   for k in range(4)], axis=0)
    if w_hll is not None and np.any(w_hll > 0):
        # HLL (HLLE) flux with the same wave-speed estimates
        dS = np.where(np.abs(SR - SL) < 1e-12, 1e-12, SR - SL)
        Fm = (SR * FL - SL * FR + SL * SR * (UR - UL)) / dS
        Fh = np.where(SL >= 0, FL, np.where(SR <= 0, FR, Fm))
        Fn = (1.0 - w_hll) * Fn + w_hll * Fh
    return np.stack([Fn[0], Fn[1] * nx + Fn[2] * tx,
                     Fn[1] * ny + Fn[2] * ty, Fn[3]], axis=0)


# =============================================================================
# 6.  BOUNDARY CONDITIONS  (characteristic based)
# =============================================================================
def apply_bcs(U, mesh, gas, cfg):
    """Fill the ghost ring of the padded primitive array.

    inlet   : subsonic stagnation inflow, characteristic form.  Exactly one
              characteristic leaves the domain, carrying the Riemann invariant
              R- = u_n - 2a/(gamma-1) out of the interior.  The boundary Mach
              number is the point on the (p0, T0) VARIABLE-gamma isentrope with
              that same R-, capped at M = 1 so the inlet can never choke on
              itself.  (Imposing p_b = p_interior instead over-determines the
              boundary and leaves a permanent residual hard spot at the inlet
              corner -- that is what stalled the original solver at ~1e-1.)
    outlet  : supersonic -> extrapolate;  subsonic outflow -> impose p_b on the
              outgoing characteristic;  reverse flow -> ambient reservoir.
    walls   : inviscid slip, mirrored about the face normal (exact symmetry on
              the axis).
    farfield: 1D Riemann-invariant far field with all four branches
              (supersonic in/out, subsonic in/out).
    """
    W = gas.cons_to_prim(U)
    Wg = np.pad(W, ((0, 0), (1, 1), (1, 1)), mode="edge")
    iso = cfg["iso"]

    # ------------------------------------------------------------ 6a. inlet
    nxi, nyi = mesh.i_nx[:, 0], mesh.i_ny[:, 0]
    rho_i, u_i, v_i, p_i = W[0, :, 0], W[1, :, 0], W[2, :, 0], W[3, :, 0]
    p0, T0 = cfg["p0"], cfg["T0"]
    T_i = gas.T_from_rho_p(rho_i, p_i)
    a_i = gas.sound_speed_Trho(T_i, rho_i)
    g_i = gas.gamma(T_i, rho_i)
    un_i = u_i * nxi + v_i * nyi
    Rm = un_i - 2.0 * a_i / (g_i - 1.0)                 # outgoing invariant
    M_in = np.clip(np.interp(Rm, iso.Rminus, iso.M), 1e-6, 1.0)
    T_in = np.interp(M_in, iso.M, iso.T_T0) * T0
    pb_in = np.interp(M_in, iso.M, iso.p_p0) * p0
    R_in = gas.R_eff(T_in, np.maximum(rho_i, 1e-9))
    rho_in = pb_in / (R_in * T_in)
    V_in = M_in * gas.sound_speed_Trho(T_in, rho_in)
    if cfg.get("inlet_axial", True):
        dx, dy = np.ones_like(nxi), np.zeros_like(nyi)
    else:
        dx, dy = nxi, nyi
    Wg[0, 1:-1, 0] = rho_in
    Wg[1, 1:-1, 0] = V_in * dx
    Wg[2, 1:-1, 0] = V_in * dy
    Wg[3, 1:-1, 0] = pb_in

    # ----------------------------------------------------------- 6b. outlet
    nxo, nyo = mesh.i_nx[:, -1], mesh.i_ny[:, -1]
    rho_o, u_o, v_o, p_o = W[0, :, -1], W[1, :, -1], W[2, :, -1], W[3, :, -1]
    a_o = gas.sound_speed(rho_o, p_o)
    un_o = u_o * nxo + v_o * nyo
    if cfg["outlet"] == "supersonic":
        Wg[:, 1:-1, -1] = W[:, :, -1]
    else:
        pb = cfg["pb"]
        rho_amb = cfg["p_amb"] / (float(np.asarray(gas.R_eff(
            np.array(cfg["T_amb"]), np.array(1.0))).ravel()[0]) * cfg["T_amb"])
        sup = (un_o / a_o) >= 1.0
        rev = un_o < 0.0                                  # reverse (inflow)
        dun = (p_o - pb) / (rho_o * a_o)
        rho_b = np.where(sup, rho_o,
                         np.where(rev, rho_amb,
                                  np.maximum(rho_o + (pb - p_o) / a_o ** 2, 1e-8)))
        un_b = np.where(sup, un_o, un_o + dun)
        ut_o = u_o * (-nyo) + v_o * nxo
        ut_b = np.where(rev, 0.0, ut_o)
        Wg[0, 1:-1, -1] = rho_b
        Wg[1, 1:-1, -1] = np.where(sup, u_o, un_b * nxo + ut_b * (-nyo))
        Wg[2, 1:-1, -1] = np.where(sup, v_o, un_b * nyo + ut_b * nxo)
        Wg[3, 1:-1, -1] = np.where(sup, p_o, pb)

    # ------------------------------------------------- 6c. walls / symmetry
    for side, j, jg, jf in (("bot", 0, 0, 0), ("top", -1, -1, -1)):
        nxw = mesh.j_nx[jf, :]
        nyw = mesh.j_ny[jf, :]
        rho, u, v, p = W[0, j, :], W[1, j, :], W[2, j, :], W[3, j, :]
        un = u * nxw + v * nyw
        Wg[0, jg, 1:-1] = rho
        Wg[1, jg, 1:-1] = u - 2.0 * un * nxw
        Wg[2, jg, 1:-1] = v - 2.0 * un * nyw
        Wg[3, jg, 1:-1] = p

    # ----------------------------------------------- 6d. plume far field
    #  Applied to the top boundary always, and to the BOTTOM boundary as well
    #  in planar mode -- a planar jet is free on both sides.  In axisymmetric
    #  mode the bottom stays the symmetry axis.
    if mesh.nx_plume > 0:
        pl = mesh.plume_cols
        p_a, T_a = cfg["p_amb"], cfg["T_amb"]
        R_a = float(np.asarray(gas.R_eff(np.array(T_a), np.array(1.0))).ravel()[0])
        rho_a = p_a / (R_a * T_a)
        g_a = float(np.asarray(gas.gamma(np.array(T_a), np.array(rho_a))).ravel()[0])
        a_a = math.sqrt(g_a * R_a * T_a)

        sides = [(-1, 1.0)] if mesh.axisym else [(-1, 1.0), (0, -1.0)]
        for jrow, sgn in sides:
            jf = -1 if jrow == -1 else 0
            nxf = sgn * mesh.j_nx[jf, pl]        # outward normal from the fluid
            nyf = sgn * mesh.j_ny[jf, pl]
            rho = W[0, jrow, pl]
            u, v, p = W[1, jrow, pl], W[2, jrow, pl], W[3, jrow, pl]
            T = gas.T_from_rho_p(rho, p)
            g = gas.gamma(T, rho)
            a = gas.sound_speed_Trho(T, rho)
            un = u * nxf + v * nyf
            ut = u * (-nyf) + v * nxf

            if cfg.get("legacy_plume", False):
                gm = 0.5 * (g + g_a)
                Rp = un + 2.0 * a / (gm - 1.0)          # invariant from inside
                Rm = 0.0 - 2.0 * a_a / (gm - 1.0)       # from the ambient (V_amb = 0)
                un_b = 0.5 * (Rp + Rm)
                a_b = np.maximum(0.25 * (gm - 1.0) * (Rp - Rm), 1e-3)
                out = un_b > 0.0
                s_in = p / np.maximum(rho, 1e-12) ** gm
                s_am = p_a / rho_a ** gm
                s_b = np.where(out, s_in, s_am)
                rho_b = (a_b ** 2 / (gm * s_b)) ** (1.0 / (gm - 1.0))
                p_b = rho_b * a_b ** 2 / gm
                ut_b = np.where(out, ut, 0.0)
                sup_out = (un / a) >= 1.0
                sup_in = (un / a) <= -1.0
                rho_f = np.where(sup_out, rho, np.where(sup_in, rho_a, rho_b))
                p_f = np.where(sup_out, p, np.where(sup_in, p_a, p_b))
                un_f = np.where(sup_out, un, np.where(sup_in, 0.0, un_b))
                ut_f = np.where(sup_out, ut, np.where(sup_in, 0.0, ut_b))
            else:
                # PRESSURE OPENING (entrainment) boundary
                #   supersonic outflow : everything from the interior
                #   subsonic outflow   : p = p_amb, rho/velocity from interior
                #   inflow             : ambient p_amb, rho_amb (T_amb),
                #                        normal velocity from the interior,
                #                        no tangential velocity
                sup_out = (un / a) >= 1.0
                outf = un >= 0.0
                rho_f = np.where(outf, rho, rho_a)
                p_f = np.where(sup_out, p, p_a)
                un_f = un
                ut_f = np.where(outf, ut, 0.0)
            u_f = un_f * nxf + ut_f * (-nyf)
            v_f = un_f * nyf + ut_f * nxf
            for k, val in ((0, rho_f), (1, u_f), (2, v_f), (3, p_f)):
                row = Wg[k, jrow, 1:-1]
                row[pl] = val
    return Wg


# =============================================================================
# 7.  RESIDUAL, TIME STEP, DRIVER
# =============================================================================
def residual(U, mesh, gas, cfg, order):
    Wg = apply_bcs(U, mesh, gas, cfg)
    Wint = Wg[:, 1:-1, 1:-1]
    if order == 2:
        WzL, WzR = reconstruct(Wg, 2)
        WrL, WrR = reconstruct(Wg, 1)
    else:
        WzL, WzR = Wg[:, 1:-1, :-1].copy(), Wg[:, 1:-1, 1:].copy()
        WrL, WrR = Wg[:, :-1, 1:-1].copy(), Wg[:, 1:, 1:-1].copy()
    # boundary faces always use the raw interior/ghost pair
    WzL[:, :, 0] = Wg[:, 1:-1, 0]
    WzR[:, :, 0] = Wint[:, :, 0]
    WzL[:, :, -1] = Wint[:, :, -1]
    WzR[:, :, -1] = Wg[:, 1:-1, -1]
    WrL[:, 0, :] = Wg[:, 0, 1:-1]
    WrR[:, 0, :] = Wint[:, 0, :]
    WrL[:, -1, :] = Wint[:, -1, :]
    WrR[:, -1, :] = Wg[:, -1, 1:-1]

    wz = wr = None
    if cfg.get("flux", "hllc") != "hllc":
        if cfg["flux"] == "hll":
            wz = np.ones(mesh.i_nx.shape); wr = np.ones(mesh.j_nx.shape)
        else:
            pg = Wg[3]
            pc = pg[1:-1, 1:-1]
            rat = np.ones_like(pc)
            for nb in (pg[1:-1, :-2], pg[1:-1, 2:], pg[:-2, 1:-1], pg[2:, 1:-1]):
                rat = np.maximum(rat, np.maximum(nb, pc) / np.maximum(np.minimum(nb, pc), 1e-9))
            flag = rat > 1.0 + cfg.get("hll_thr", 0.5)
            for _ in range(int(cfg.get("hll_dilate", 1))):
                fp = np.pad(flag, 1, mode="edge")
                flag = (fp[1:-1, 1:-1] | fp[:-2, 1:-1] | fp[2:, 1:-1] |
                        fp[1:-1, :-2] | fp[1:-1, 2:])
            f = flag.astype(float)
            fz = np.pad(f, ((0, 0), (1, 1)), mode="edge")
            fr = np.pad(f, ((1, 1), (0, 0)), mode="edge")
            wz = np.maximum(fz[:, :-1], fz[:, 1:])
            wr = np.maximum(fr[:-1, :], fr[1:, :])
            cfg["_hll_cells"] = int(flag.sum())
    Fz = hllc(WzL, WzR, mesh.i_nx, mesh.i_ny, gas, wz) * mesh.i_area
    Fr = hllc(WrL, WrR, mesh.j_nx, mesh.j_ny, gas, wr) * mesh.j_area
    R = -((Fz[:, :, 1:] - Fz[:, :, :-1]) + (Fr[:, 1:, :] - Fr[:, :-1, :])) / mesh.vol
    if mesh.axisym:
        # conservative axisymmetric form: the only source is the hoop pressure
        R[2] += Wint[3] * mesh.area / mesh.vol
    return R


def local_dt(U, mesh, gas, cfl):
    """Face-based spectral radius: dt = CFL * V / sum_faces (|u.n| + a) S."""
    W = gas.cons_to_prim(U)
    a = gas.sound_speed(W[0], W[3])
    u, v = W[1], W[2]
    li = ((np.abs(u * mesh.i_nx[:, :-1] + v * mesh.i_ny[:, :-1]) + a) * mesh.i_area[:, :-1] +
          (np.abs(u * mesh.i_nx[:, 1:] + v * mesh.i_ny[:, 1:]) + a) * mesh.i_area[:, 1:])
    lj = ((np.abs(u * mesh.j_nx[:-1, :] + v * mesh.j_ny[:-1, :]) + a) * mesh.j_area[:-1, :] +
          (np.abs(u * mesh.j_nx[1:, :] + v * mesh.j_ny[1:, :]) + a) * mesh.j_area[1:, :])
    return cfl * mesh.vol / (0.5 * (li + lj) + 1e-30)


def smooth(R, eps, passes):
    """Implicit residual smoothing (Jacobi).  Fixed point unchanged: S=0 iff R=0."""
    if eps <= 0:
        return R
    S = R
    for _ in range(passes):
        P = np.pad(S, ((0, 0), (1, 1), (1, 1)), mode="edge")
        nb = P[:, :-2, 1:-1] + P[:, 2:, 1:-1] + P[:, 1:-1, :-2] + P[:, 1:-1, 2:]
        S = (R + eps * nb) / (1.0 + 4.0 * eps)
    return S


def march(U, mesh, gas, cfg, it0=0, res0=-1.0, hist=None, wall_limit=None):
    """SSP-RK2 march to steady state.  Returns (U, it, res0, hist, converged)."""
    hist = list(hist or [])
    conv = False
    t0 = time.time()
    it = it0
    rel = hist[-1][1] if hist else 1.0
    stop = "max_iter"
    while it < cfg["max_iter"]:
        if wall_limit and time.time() - t0 > wall_limit:
            stop = "time_limit"
            break
        it += 1
        c = cfg["cfl"] * min(1.0, 0.1 + 0.9 * it / 400.0)      # gentle ramp-in
        dt = local_dt(U, mesh, gas, c)
        R1 = residual(U, mesh, gas, cfg, cfg["order"])
        U1 = U + dt * smooth(R1, cfg["irs_eps"], cfg["irs_passes"])
        U1 = gas.prim_to_cons(_floor(gas.cons_to_prim(U1)))
        R2 = residual(U1, mesh, gas, cfg, cfg["order"])
        Un = 0.5 * (U + U1 + dt * smooth(R2, cfg["irs_eps"], cfg["irs_passes"]))
        U = gas.prim_to_cons(_floor(gas.cons_to_prim(Un)))

        res = float(np.sqrt(np.mean(R1[0] ** 2)))
        if not np.isfinite(res):
            print(f"  FATAL: NaN at iteration {it}.  Try a lower --cfl or --order 1.")
            sys.exit(2)
        if res0 < 0:
            res0 = res + 1e-30
        rel = res / res0
        hist.append((it, rel))
        if it % cfg["log_every"] == 0 or it == it0 + 1:
            print(f"  iter {it:6d}   res={res:.3e}   rel={rel:.3e}")
        if rel < cfg["tol"]:
            conv, stop = True, "converged"
            print(f"  CONVERGED at iteration {it}: rel = {rel:.3e} < {cfg['tol']:.0e}")
            break
    if stop == "max_iter":
        print(f"  stopped at max_iter={cfg['max_iter']} with rel={rel:.3e}")
    elif stop == "time_limit":
        print(f"  stopped on the wall-clock limit at iteration {it}, rel={rel:.3e}")
    return U, it, res0, hist, conv


# =============================================================================
# 8.  INITIAL CONDITION
# =============================================================================
def initial_state(mesh, gas, cfg, th):
    W = np.zeros((4, mesh.ny, mesh.nx))
    noz = th["noz"]
    rho = np.nan_to_num(th["rho"], nan=cfg["p_amb"] / (gas.R0 * cfg["T_amb"]))
    u = np.nan_to_num(th["u"], nan=0.0)
    p = np.nan_to_num(th["p"], nan=cfg["p_amb"])
    W[0][:, noz], W[1][:, noz], W[3][:, noz] = rho[noz], u[noz], p[noz]
    W[2][:] = 0.0
    if mesh.nx_plume > 0:
        last = int(np.where(noz)[0][-1])
        R_a = float(np.asarray(gas.R_eff(np.array(cfg["T_amb"]), np.array(1.0))).ravel()[0])
        rho_amb = cfg["p_amb"] / (R_a * cfg["T_amb"])
        jet = np.abs(mesh.yc) < mesh.h_e
        for i in np.where(~noz)[0]:
            m = jet[:, i]
            W[0, m, i], W[1, m, i], W[3, m, i] = rho[last], u[last], p[last]
            W[0, ~m, i], W[1, ~m, i], W[3, ~m, i] = rho_amb, 0.0, cfg["p_amb"]
    W[0] = np.maximum(W[0], 1e-8)
    W[3] = np.maximum(W[3], 1.0)
    return gas.prim_to_cons(W)


def interpolate_state(U_old, mesh_old_shape, mesh, gas):
    """Grid sequencing: map a coarse solution onto the current mesh."""
    nr_o, nz_o = mesh_old_shape
    W = gas.cons_to_prim(U_old)
    x_o = np.linspace(0.0, 1.0, nz_o)
    x_n = np.linspace(0.0, 1.0, mesh.nx)
    e_o = (np.arange(nr_o) + 0.5) / nr_o
    e_n = (np.arange(mesh.ny) + 0.5) / mesh.ny
    tmp = np.empty((4, nr_o, mesh.nx))
    for k in range(4):
        for j in range(nr_o):
            tmp[k, j] = np.interp(x_n, x_o, W[k, j])
    out = np.empty((4, mesh.ny, mesh.nx))
    for k in range(4):
        for i in range(mesh.nx):
            out[k, :, i] = np.interp(e_n, e_o, tmp[k, :, i])
    return gas.prim_to_cons(out)


# =============================================================================
# 9.  DERIVED FIELDS  --  every quantity a shock calculation needs
# =============================================================================
def cell_gradient(mesh, f):
    """Green-Gauss cell gradient of a cell-centred field on the planar mesh."""
    fp = np.pad(f, ((1, 1), (1, 1)), mode="edge")
    fi = 0.5 * (fp[1:-1, :-1] + fp[1:-1, 1:])          # (ny, nx+1) on i-faces
    fj = 0.5 * (fp[:-1, 1:-1] + fp[1:, 1:-1])          # (ny+1, nx) on j-faces
    gz = (fi[:, 1:] * mesh.i_nx[:, 1:] * mesh.i_len[:, 1:] -
          fi[:, :-1] * mesh.i_nx[:, :-1] * mesh.i_len[:, :-1] +
          fj[1:, :] * mesh.j_nx[1:, :] * mesh.j_len[1:, :] -
          fj[:-1, :] * mesh.j_nx[:-1, :] * mesh.j_len[:-1, :]) / mesh.area
    gr = (fi[:, 1:] * mesh.i_ny[:, 1:] * mesh.i_len[:, 1:] -
          fi[:, :-1] * mesh.i_ny[:, :-1] * mesh.i_len[:, :-1] +
          fj[1:, :] * mesh.j_ny[1:, :] * mesh.j_len[1:, :] -
          fj[:-1, :] * mesh.j_ny[:-1, :] * mesh.j_len[:-1, :]) / mesh.area
    return gz, gr


def divergence_curl(mesh, u, v):
    uz, ur = cell_gradient(mesh, u)
    vz, vr = cell_gradient(mesh, v)
    div = uz + vr
    if mesh.axisym:
        div = div + v / np.maximum(np.abs(mesh.yc), 1e-12)
    curl = vz - ur
    return div, curl


def prandtl_meyer(M, g):
    """nu(M) for a locally frozen gamma [rad]."""
    M = np.maximum(np.asarray(M, dtype=float), 1.0)
    b = np.sqrt(np.maximum(M * M - 1.0, 0.0))
    k = np.sqrt((g + 1.0) / (g - 1.0))
    return k * np.arctan(b / k) - np.arctan(b)


def derived_fields(mesh, gas, cfg, W):
    """All primitive + derived state, as (ny, nx) arrays."""
    rho, u, v, p = W[0], W[1], W[2], W[3]
    T = gas.T_from_rho_p(rho, p)
    Rg = gas.R_eff(T, rho)
    g = gas.gamma(T, rho)
    cv = gas.cv(T, rho)
    cp = gas.cp(T, rho)
    a = gas.sound_speed_Trho(T, rho)
    V = np.sqrt(u * u + v * v)
    M = V / a
    ang = np.degrees(np.arctan2(v, u))
    T0f, p0f, rho0f = gas.stagnation(T, rho, V)
    s = gas.entropy(T, rho)
    mu = np.degrees(np.arcsin(np.clip(1.0 / np.maximum(M, 1e-9), -1.0, 1.0)))
    mu = np.where(M >= 1.0, mu, np.nan)
    nu = np.degrees(prandtl_meyer(M, g))
    nu = np.where(M >= 1.0, nu, np.nan)
    alpha = gas.alpha(T, rho)
    gz, gr = cell_gradient(mesh, p)
    gmag = np.sqrt(gz * gz + gr * gr)
    div, curl = divergence_curl(mesh, u, v)
    return dict(rho=rho, u=u, v=v, p=p, T=T, R=Rg, gamma=g, cv=cv, cp=cp,
                a=a, V=V, M=M, angle_deg=ang, T0=T0f, p0=p0f, rho0=rho0f,
                entropy=s, mach_angle_deg=mu, prandtl_meyer_deg=nu,
                alpha_dissociation=alpha, grad_p_x=gz, grad_p_y=gr,
                grad_p_mag=gmag, div_V=div, curl_V=curl,
                q_dynamic=0.5 * rho * V * V, mass_flux=rho * V,
                total_enthalpy=gas.h(T, rho) + 0.5 * V * V)


# =============================================================================
# 10.  SHOCK DETECTION AND CHARACTERISATION
# =============================================================================
def shock_sensor(mesh, F, sp_min=0.05, Mn_min=1.02):
    """Flag shock cells.

    A cell is a shock cell when ALL of the following hold:
      1. normalised pressure jump   |grad p| * h_cell / p  >  sp_min
      2. compression                div V < 0
      3. supersonic shock-normal Mach  M_n = (V . grad p/|grad p|) / a > Mn_min
    Criterion 3 is what distinguishes a shock from a strong expansion or a
    contact/shear layer; criterion 2 removes expansion fans.
    """
    p, a, u, v = F["p"], F["a"], F["u"], F["v"]
    gmag = np.maximum(F["grad_p_mag"], 1e-30)
    nxs, nys = F["grad_p_x"] / gmag, F["grad_p_y"] / gmag
    sp = gmag * mesh.h_cell / np.maximum(p, 1e-12)
    Mn = (u * nxs + v * nys) / np.maximum(a, 1e-12)
    ducros = np.abs(F["div_V"]) / (np.abs(F["div_V"]) + np.abs(F["curl_V"]) + 1e-8)
    flag = (sp > sp_min) & (F["div_V"] < 0.0) & (Mn > Mn_min)
    return dict(flag=flag, sensor=sp, Mn=Mn, nx=nxs, ny=nys, ducros=ducros)


def _label_structures(flag):
    """4-connected component labelling of the boolean shock mask."""
    ny, nx = flag.shape
    lab = -np.ones((ny, nx), dtype=int)
    n = 0
    for j in range(ny):
        for i in range(nx):
            if not flag[j, i] or lab[j, i] >= 0:
                continue
            q = deque([(j, i)])
            lab[j, i] = n
            while q:
                jj, ii = q.popleft()
                for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = jj + dj, ii + di
                    if 0 <= a < ny and 0 <= b < nx and flag[a, b] and lab[a, b] < 0:
                        lab[a, b] = n
                        q.append((a, b))
            n += 1
    return lab, n


def _probe(mesh, sen, j, i, k):
    """Index offset k cells along the shock normal (negative = upstream)."""
    dx = max(float(mesh.dx_cell[j, i]), 1e-12)
    dy = max(float(mesh.dy_cell[j, i]), 1e-12)
    sz, sr = sen["nx"][j, i] / dx, sen["ny"][j, i] / dy
    nrm = max(abs(sz), abs(sr), 1e-30)
    di = int(round(k * sz / nrm))
    dj = int(round(k * sr / nrm))
    return (min(max(j + dj, 0), mesh.ny - 1), min(max(i + di, 0), mesh.nx - 1))


def characterise_shocks(mesh, gas, F, sen, probe=3, max_report=40):
    """Group flagged cells and extract shock-wave parameters per structure."""
    lab, n = _label_structures(sen["flag"])
    out = []
    for c in range(n):
        m = lab == c
        cnt = int(m.sum())
        if cnt < 3:
            continue
        js, iss = np.where(m)
        strength = sen["sensor"][m]
        k = int(np.argmax(strength))
        j0, i0 = int(js[k]), int(iss[k])
        ju, iu = _probe(mesh, sen, j0, i0, -probe)
        jd, idn = _probe(mesh, sen, j0, i0, +probe)

        rho1, p1 = float(F["rho"][ju, iu]), float(F["p"][ju, iu])
        rho2, p2 = float(F["rho"][jd, idn]), float(F["p"][jd, idn])
        T1, T2 = float(F["T"][ju, iu]), float(F["T"][jd, idn])
        V1 = float(F["V"][ju, iu])
        M1 = float(F["M"][ju, iu])
        M2 = float(F["M"][jd, idn])
        a1 = float(F["a"][ju, iu])
        nxs, nys = float(sen["nx"][j0, i0]), float(sen["ny"][j0, i0])
        un1 = float(F["u"][ju, iu]) * nxs + float(F["v"][ju, iu]) * nys
        Mn1 = un1 / max(a1, 1e-12)
        sinb = min(max(abs(Mn1) / max(M1, 1e-12), 0.0), 1.0)
        beta = math.asin(sinb) if M1 > 1e-9 else float("nan")
        jump = normal_shock_real(gas, rho1, abs(un1), p1)
        obl = oblique_from_normal(jump, beta) if (jump and beta == beta) else None
        ds = float(F["entropy"][jd, idn] - F["entropy"][ju, iu])

        on_axis = bool(js.min() == 0)
        axial = abs(nxs) > 0.9
        if on_axis and axial:
            kind = "normal shock on the axis (Mach disc / in-nozzle normal shock)"
        elif abs(nxs) > 0.9:
            kind = "quasi-normal shock"
        elif on_axis:
            kind = "reflected / intersecting oblique shock at the axis"
        else:
            kind = "oblique shock (barrel / compression wave)"

        out.append(dict(
            structure_id=c, n_cells=cnt, type=kind,
            centroid_x=sig(float(mesh.xc[m].mean())),
            centroid_y=sig(float(mesh.yc[m].mean())),
            x_range=[sig(float(mesh.xc[m].min())), sig(float(mesh.xc[m].max()))],
            y_range=[sig(float(mesh.yc[m].min())), sig(float(mesh.yc[m].max()))],
            strongest_cell_id=int(mesh.cell_id[j0, i0]),
            strongest_cell_ij=[i0, j0],
            shock_normal_unit=[sig(nxs), sig(nys)],
            wave_angle_beta_deg=sig(math.degrees(beta)) if beta == beta else None,
            deflection_theta_deg=sig(obl["theta_deg"]) if obl else None,
            M1_upstream=sig(M1), Mn1_normal=sig(Mn1), M2_downstream=sig(M2),
            M2_oblique_theory=sig(obl["M2_oblique"]) if obl else None,
            p2_over_p1_solver=sig(p2 / max(p1, 1e-30)),
            p2_over_p1_rankine_hugoniot=sig(jump["p2_p1"]) if jump else None,
            rho2_over_rho1_solver=sig(rho2 / max(rho1, 1e-30)),
            rho2_over_rho1_rankine_hugoniot=sig(jump["rho2_rho1"]) if jump else None,
            T2_over_T1_solver=sig(T2 / max(T1, 1e-30)),
            T2_over_T1_rankine_hugoniot=sig(jump["T2_T1"]) if jump else None,
            p02_over_p01_rankine_hugoniot=sig(jump["p02_p01"]) if jump else None,
            entropy_rise_J_per_kgK=sig(ds),
            entropy_rise_rankine_hugoniot=sig(jump["ds_J_per_kgK"]) if jump else None,
            gamma_upstream=sig(float(F["gamma"][ju, iu])),
            gamma_downstream=sig(float(F["gamma"][jd, idn])),
            R_upstream_J_per_kgK=sig(float(F["R"][ju, iu])),
            R_downstream_J_per_kgK=sig(float(F["R"][jd, idn])),
            upstream_cell_id=int(mesh.cell_id[ju, iu]),
            upstream_cell_ij=[int(ju), int(iu)],
            downstream_cell_id=int(mesh.cell_id[jd, idn]),
            upstream_state=dict(p_Pa=sig(p1), T_K=sig(T1), rho=sig(rho1),
                                V_m_per_s=sig(V1), M=sig(M1)),
            downstream_state=dict(p_Pa=sig(p2), T_K=sig(T2), rho=sig(rho2),
                                  V_m_per_s=sig(float(F["V"][jd, idn])), M=sig(M2)),
            cells_per_shock_thickness=sig(cnt / max(1.0, (js.max() - js.min() + 1.0))),
            deflection_from_velocity_deg=sig(float(F["angle_deg"][jd, idn] -
                                                   F["angle_deg"][ju, iu])),
            deflection_note="deflection_theta_deg comes from the density ratio "
                            "via tan(beta-theta) = (rho1/rho2) tan(beta), so a "
                            "theta-beta-M check against it is circular; "
                            "deflection_from_velocity_deg is measured directly "
                            "from the upstream and downstream flow angles and "
                            "is the independent quantity",
        ))
    out.sort(key=lambda d: -(d["p2_over_p1_solver"] or 0.0))
    return lab, out[:max_report]


def axis_shock_cells(mesh, F, rel_tol=2e-3, prominence=0.12):
    """Shock-cell (diamond) nodes on the jet centreline.

    Turning points of the centreline pressure alone are far too sensitive:
    every small numerical wiggle registers, and the reported cell spacing comes
    out several times too short.  This uses a zig-zag / hysteresis filter --
    an extremum is only accepted once the pressure has since moved by
    `prominence` x (p_max - p_min) in the opposite direction -- so only genuine
    compression peaks and expansion troughs survive.  A shock cell spans peak
    to peak.
    """
    if mesh.nx_plume == 0:
        return [], None
    pl = mesh.plume_cols
    rows = mesh.centre_rows
    x = mesh.xc[rows[0], pl]
    p = np.mean(np.stack([F["p"][j, pl] for j in rows]), axis=0)
    Ma = np.mean(np.stack([F["M"][j, pl] for j in rows]), axis=0)
    n = len(p)
    if n < 5:
        return [], None
    prom = max(prominence * float(np.nanmax(p) - np.nanmin(p)),
               rel_tol * float(np.mean(np.abs(p))))
    cells, last, trend = [], 0, 0      # trend: 0 unknown, +1 rising, -1 falling
    for i in range(1, n):
        if trend >= 0 and p[i] >= p[last]:
            last, trend = i, 1             # extend the rise
            continue
        if trend <= 0 and p[i] <= p[last]:
            last, trend = i, -1            # extend the fall
            continue
        # p has turned against the trend: confirm the extremum only once it has
        # moved back by the prominence band
        if trend > 0 and p[i] < p[last] - prom:
            cells.append((last, "compression_peak"))
            last, trend = i, -1
        elif trend < 0 and p[i] > p[last] + prom:
            cells.append((last, "expansion_trough"))
            last, trend = i, 1
    # accept the still-pending extremum at the end of the domain if it is far
    # enough from the last confirmed one -- otherwise the last cell of a short
    # plume is always discarded for want of the downstream half of its cycle
    if cells:
        k0 = cells[-1][0]
        if abs(p[last] - p[k0]) >= prom and last != k0:
            cells.append((last, "compression_peak" if p[last] > p[k0]
                          else "expansion_trough"))
    out = [dict(x=sig(float(x[k])), mach=sig(float(Ma[k])),
                p_Pa=sig(float(p[k])), type=t) for k, t in cells]
    xp = [c["x"] for c in out if c["type"] == "compression_peak"]
    spacing = sig(float(np.mean(np.diff(xp)))) if len(xp) >= 2 else None
    return out, spacing


def plume_shock_metrics(mesh, gas, cfg, iso, F, cells, spacing):
    """Classical scaling laws for the free-jet shock-cell structure."""
    if mesh.nx_plume == 0:
        return None
    p0, p_a = cfg["p0"], cfg["p_amb"]
    NPR = p0 / max(p_a, 1e-30)
    Mj = iso.M_from_p_ratio(max(p_a / p0, iso.p_p0[-1]))     # fully expanded
    De = 2.0 * mesh.h_e
    Ls = 1.306 * De * math.sqrt(max(Mj ** 2 - 1.0, 0.0))
    Me = iso.M_from_area(mesh.Ae_At, "supersonic")
    pe = iso.p_ratio(Me) * p0
    return dict(
        NPR=sig(NPR),
        fully_expanded_jet_mach_Mj=sig(Mj),
        exit_mach_design_Me=sig(Me),
        exit_pressure_design_Pa=sig(pe),
        pe_over_pamb=sig(pe / max(p_a, 1e-30)),
        jet_state=("underexpanded" if pe > p_a * 1.02 else
                   ("overexpanded" if pe < p_a * 0.98 else "matched")),
        prandtl_shock_cell_spacing_m=sig(Ls),
        prandtl_formula="L = 1.306 * D_e * sqrt(Mj^2 - 1)",
        ashkenas_sherman_mach_disc_z_from_lip_m=sig(0.67 * De * math.sqrt(max(NPR, 1.0))),
        ashkenas_sherman_formula="x_M/D_e = 0.67 sqrt(p0/p_amb)",
        measured_shock_cell_spacing_m=spacing,
        n_axis_turning_points=len(cells),
        axis_turning_points=cells[:20],
        exit_diameter_m=sig(De),
        note="the Prandtl and Ashkenas-Sherman correlations are free-jet "
             "scalings for a sonic/underexpanded round jet; treat them as "
             "order-of-magnitude checks on the computed cell structure",
    )




# =============================================================================
# 10b. WAVE TOOLKIT  --  oblique-shock relations, expansion fans, slip lines,
#      characteristics and shock-shock interactions
#
#      Everything here is POST-PROCESSING: it reads the converged field and the
#      shock sensor, and never touches the solver.  All shock relations go
#      through normal_shock_real(), so they use the same variable-gamma gas as
#      the solution rather than perfect-gas closed forms.
# =============================================================================
def oblique_theta(gas, rho1, V1, p1, beta):
    """Flow deflection produced by a wave at angle beta (real gas).

    The tangential velocity is conserved and only the normal component shocks,
    so mass conservation across the wave gives, for ANY equation of state,
        tan(beta - theta) = (rho1/rho2) tan(beta)
    """
    Vn = V1 * math.sin(beta)
    jump = normal_shock_real(gas, rho1, Vn, p1)
    if jump is None:
        return None, None
    r = jump["rho1"] / jump["rho2"]
    return beta - math.atan(r * math.tan(beta)), jump


def oblique_shock_table(gas, rho1, V1, p1, n=181):
    """theta(beta) over the admissible range -- this IS the shock polar.

    Returns theta_max (the detachment angle), the wave angle at which it
    occurs, and beta_sonic (where the downstream flow becomes sonic).  Above
    theta_max no attached oblique shock exists and the wave detaches into a
    bow shock; between beta(theta_max) and beta_sonic the downstream flow is
    subsonic on the strong branch.
    """
    T1 = float(np.asarray(gas.T_from_rho_p(np.array(rho1), np.array(p1))).ravel()[0])
    a1 = float(np.asarray(gas.sound_speed_Trho(np.array(T1), np.array(rho1))).ravel()[0])
    M1 = V1 / a1
    if M1 <= 1.0 + 1e-9:
        return None
    mu = math.asin(min(1.0 / M1, 1.0))
    betas = np.linspace(mu + 1e-5, 0.5 * math.pi - 1e-5, n)
    th = np.full(n, np.nan)
    M2 = np.full(n, np.nan)
    pr = np.full(n, np.nan)
    for k, b in enumerate(betas):
        t, j = oblique_theta(gas, rho1, V1, p1, b)
        if t is None:
            continue
        th[k] = t
        pr[k] = j["p2_p1"]
        M2[k] = j["M2"] / max(math.sin(b - t), 1e-9)
    if not np.any(np.isfinite(th)):
        return None
    kmax = int(np.nanargmax(th))
    b_sonic = None
    fin = np.where(np.isfinite(M2))[0]
    for a, b in zip(fin[:-1], fin[1:]):
        if (M2[a] - 1.0) * (M2[b] - 1.0) < 0.0:
            w = (1.0 - M2[a]) / (M2[b] - M2[a])
            b_sonic = float(betas[a] + w * (betas[b] - betas[a]))
            break
    return dict(M1=M1, mach_angle=mu, beta=betas, theta=th, M2=M2, p2_p1=pr,
                theta_max=float(th[kmax]), beta_at_theta_max=float(betas[kmax]),
                beta_sonic=b_sonic)


def oblique_solve(gas, rho1, V1, p1, theta, branch="weak", tab=None):
    """Invert theta -> beta on the weak or strong branch (the theta-beta-M
    relation, solved for the real gas).  Returns None if the shock detaches."""
    tab = tab or oblique_shock_table(gas, rho1, V1, p1)
    if tab is None:
        return None
    if theta > tab["theta_max"]:
        return dict(attached=False, theta_deg=math.degrees(theta),
                    theta_max_deg=math.degrees(tab["theta_max"]),
                    note="deflection exceeds theta_max: the shock detaches")
    b, t = tab["beta"], tab["theta"]
    kmax = int(np.nanargmax(t))
    if branch == "weak":
        lo, hi = b[0], b[kmax]
    else:
        lo, hi = b[kmax], b[-1]
    f = lambda bb: (oblique_theta(gas, rho1, V1, p1, bb)[0] or -9.9) - theta
    beta = bisect(f, lo, hi, tol=1e-12)
    th, jump = oblique_theta(gas, rho1, V1, p1, beta)
    M2 = jump["M2"] / max(math.sin(beta - th), 1e-9)
    return dict(attached=True, branch=branch,
                beta_deg=sig(math.degrees(beta)), theta_deg=sig(math.degrees(th)),
                M1=sig(tab["M1"]), M2=sig(M2), Mn1=sig(tab["M1"] * math.sin(beta)),
                p2_p1=sig(jump["p2_p1"]), rho2_rho1=sig(jump["rho2_rho1"]),
                T2_T1=sig(jump["T2_T1"]), p02_p01=sig(jump["p02_p01"]),
                ds_J_per_kgK=sig(jump["ds_J_per_kgK"]),
                theta_max_deg=sig(math.degrees(tab["theta_max"])),
                beta_at_theta_max_deg=sig(math.degrees(tab["beta_at_theta_max"])),
                beta_sonic_deg=(sig(math.degrees(tab["beta_sonic"]))
                                if tab["beta_sonic"] else None),
                mach_angle_deg=sig(math.degrees(tab["mach_angle"])))


# ------------------------------------------------------------ wave geometry
def fit_wave_line(mesh, mask):
    """Total-least-squares line through a flagged structure.

    A wave angle taken from the pressure gradient in ONE cell is noisy on a
    coarse grid; the principal axis of the whole structure is far steadier.
    Returns the centroid, unit direction, angle, length and rms thickness.
    """
    js, iss = np.where(mask)
    X = mesh.xc[js, iss]
    Y = mesh.yc[js, iss]
    x0, y0 = float(X.mean()), float(Y.mean())
    dx, dy = X - x0, Y - y0
    cov = np.array([[float((dx * dx).sum()), float((dx * dy).sum())],
                    [float((dx * dy).sum()), float((dy * dy).sum())]])
    w, V = np.linalg.eigh(cov)
    d = V[:, int(np.argmax(w))]                 # principal direction
    nrm = V[:, int(np.argmin(w))]               # wave normal
    t = dx * d[0] + dy * d[1]
    s = dx * nrm[0] + dy * nrm[1]
    lam = np.sort(np.maximum(w, 1e-30))
    aspect = float(math.sqrt(lam[1] / lam[0]))
    return dict(x0=x0, y0=y0, dx=float(d[0]), dy=float(d[1]), aspect=aspect,
                # a handful of cells gives a meaningless principal axis, so demand a
                # genuinely elongated structure before trusting the direction
                direction_reliable=bool(aspect > 3.0 and mask.sum() >= 12),
                nx=float(nrm[0]), ny=float(nrm[1]),
                angle_deg=float(math.degrees(math.atan2(abs(d[1]), abs(d[0])))),
                length=float(t.max() - t.min()),
                thickness_rms=float(np.sqrt(np.mean(s ** 2))),
                t_min=float(t.min()), t_max=float(t.max()))


def wave_family(fit, u, v):
    """C+ (left running) or C- (right running) relative to the local flow."""
    fx, fy = float(u), float(v)
    n = math.hypot(fx, fy)
    if n < 1e-12:
        return "undetermined"
    fx, fy = fx / n, fy / n
    dx, dy = fit["dx"], fit["dy"]
    if dx * fx + dy * fy < 0.0:                 # orient with the flow
        dx, dy = -dx, -dy
    return "C+ (left-running)" if (fx * dy - fy * dx) > 0 else "C- (right-running)"


def _probe_along(mesh, dirx, diry, j, i, k):
    """Index offset k cells along a physical direction."""
    dx = max(float(mesh.dx_cell[j, i]), 1e-12)
    dy = max(float(mesh.dy_cell[j, i]), 1e-12)
    sx, sy = dirx / dx, diry / dy
    nrm = max(abs(sx), abs(sy), 1e-30)
    return (min(max(j + int(round(k * sy / nrm)), 0), mesh.ny - 1),
            min(max(i + int(round(k * sx / nrm)), 0), mesh.nx - 1))


# ---------------------------------------------------------- expansion fans
def expansion_sensor(mesh, F, sp_min=0.07, Mn_min=1.05):
    """Mirror image of the shock sensor.

    A shock has the flow running INTO rising pressure (V.grad p > 0) and
    div V < 0.  An expansion fan has the flow running into FALLING pressure
    (V.grad p < 0) and div V > 0.  The shock sensor throws expansions away on
    purpose; this keeps them.
    """
    gmag = np.maximum(F["grad_p_mag"], 1e-30)
    nxs, nys = F["grad_p_x"] / gmag, F["grad_p_y"] / gmag
    sp = gmag * mesh.h_cell / np.maximum(F["p"], 1e-12)
    Mn = (F["u"] * nxs + F["v"] * nys) / np.maximum(F["a"], 1e-12)
    flag = (sp > sp_min) & (F["div_V"] > 0.0) & (Mn < -Mn_min)
    return dict(flag=flag, sensor=sp, Mn=Mn, nx=nxs, ny=nys)


def characterise_expansions(mesh, gas, F, exp_sen, probe=3, max_report=20):
    """Group expansion cells and measure the turning they produce.

    For an isentropic simple wave the Prandtl-Meyer angle change must equal
    the flow turning angle.  Reporting both is a direct check that the fan is
    resolved and that the solution is isentropic through it.
    """
    lab, n = _label_structures(exp_sen["flag"])
    out = []
    for c in range(n):
        m = lab == c
        cnt = int(m.sum())
        if cnt < 4:
            continue
        js, iss = np.where(m)
        k = int(np.argmax(exp_sen["sensor"][m]))
        j0, i0 = int(js[k]), int(iss[k])
        ux, uy = float(F["u"][j0, i0]), float(F["v"][j0, i0])
        nv = math.hypot(ux, uy) or 1.0
        ju, iu = _probe_along(mesh, -ux / nv, -uy / nv, j0, i0, probe)
        jd, idn = _probe_along(mesh, ux / nv, uy / nv, j0, i0, probe)
        M1, M2 = float(F["M"][ju, iu]), float(F["M"][jd, idn])
        nu1, nu2 = float(F["prandtl_meyer_deg"][ju, iu]), float(F["prandtl_meyer_deg"][jd, idn])
        a1, a2 = float(F["angle_deg"][ju, iu]), float(F["angle_deg"][jd, idn])
        ds = float(F["entropy"][jd, idn] - F["entropy"][ju, iu])
        fit = fit_wave_line(mesh, m)
        out.append(dict(
            structure_id=c, n_cells=cnt, type="expansion fan (Prandtl-Meyer)",
            centroid_x=sig(fit["x0"]), centroid_y=sig(fit["y0"]),
            wave_angle_deg=sig(fit["angle_deg"]),
            family=wave_family(fit, ux, uy),
            M_upstream=sig(M1), M_downstream=sig(M2),
            prandtl_meyer_upstream_deg=sig(nu1),
            prandtl_meyer_downstream_deg=sig(nu2),
            prandtl_meyer_change_deg=sig(nu2 - nu1),
            flow_turning_deg=sig(a2 - a1),
            simple_wave_residual_deg=sig(abs((nu2 - nu1) - abs(a2 - a1))),
            p_downstream_over_upstream=sig(float(F["p"][jd, idn] / max(F["p"][ju, iu], 1e-30))),
            entropy_rise_J_per_kgK=sig(ds),
            entropy_note="an isentropic fan should give ds ~ 0 and "
                         "d(nu) = |flow turning|",
            upstream_cell_id=int(mesh.cell_id[ju, iu]),
            downstream_cell_id=int(mesh.cell_id[jd, idn]),
        ))
    out.sort(key=lambda d: -(d["prandtl_meyer_change_deg"] or 0.0))
    return lab, out[:max_report]


# -------------------------------------------------------------- slip lines
def slip_line_sensor(mesh, F, gas, s_min=0.04, p_max=0.02):
    """Contact / shear discontinuity: entropy jumps, pressure does not.

    Every triple point sheds one, so finding them confirms a Mach reflection.
    """
    gx, gy = cell_gradient(mesh, F["entropy"])
    gs = np.sqrt(gx * gx + gy * gy) * mesh.h_cell / max(float(gas.R0), 1e-30)
    sp = F["grad_p_mag"] * mesh.h_cell / np.maximum(F["p"], 1e-12)
    flag = (gs > s_min) & (sp < p_max)
    return dict(flag=flag, sensor=gs, pressure_sensor=sp)


def characterise_slip_lines(mesh, F, slip, min_cells=6, max_report=12):
    lab, n = _label_structures(slip["flag"])
    out = []
    for c in range(n):
        m = lab == c
        cnt = int(m.sum())
        if cnt < min_cells:
            continue
        fit = fit_wave_line(mesh, m)
        js, iss = np.where(m)
        out.append(dict(structure_id=c, n_cells=cnt,
                        type="slip line / shear layer (contact discontinuity)",
                        centroid_x=sig(fit["x0"]), centroid_y=sig(fit["y0"]),
                        angle_deg=sig(fit["angle_deg"]),
                        length=sig(fit["length"]),
                        entropy_jump_indicator=sig(float(slip["sensor"][m].max())),
                        pressure_sensor_max=sig(float(slip["pressure_sensor"][m].max())),
                        x_range=[sig(float(mesh.xc[m].min())), sig(float(mesh.xc[m].max()))]))
    out.sort(key=lambda d: -(d["n_cells"]))
    return lab, out[:max_report]


# ------------------------------------------------------- wave interactions
def _seg_intersect(a, b, reach=None):
    """Intersection of two fitted wave lines.

    A captured shock is often flagged only in fragments, so the crossing is
    usually OUTSIDE both fragments and has to be found by extrapolating the
    fitted lines.  `reach` is how far (in physical length) each line may be
    extended; the extrapolation distance is reported so the caller can judge
    how much to trust it."""
    p = np.array([a["x0"], a["y0"]]); r = np.array([a["dx"], a["dy"]])
    q = np.array([b["x0"], b["y0"]]); s = np.array([b["dx"], b["dy"]])
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None                                  # parallel
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
    u = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / den
    ea = reach if reach is not None else 3.0 * (a["t_max"] - a["t_min"]) + 1e-9
    eb = reach if reach is not None else 3.0 * (b["t_max"] - b["t_min"]) + 1e-9
    if not (a["t_min"] - ea <= t <= a["t_max"] + ea):
        return None
    if not (b["t_min"] - eb <= u <= b["t_max"] + eb):
        return None
    extra = max(max(a["t_min"] - t, t - a["t_max"], 0.0),
                max(b["t_min"] - u, u - b["t_max"], 0.0))
    return float(p[0] + t * r[0]), float(p[1] + t * r[1]), float(extra)


def find_interactions(mesh, gas, F, structures, fits, slips, max_report=12,
                      reach_frac=0.25):
    """Pair up wave structures that cross, and classify the interaction.

    For a reflection off the axis or a wall the key question is regular (RR)
    versus Mach reflection (MR).  The detachment criterion is used: the
    reflected shock must turn the flow back through the same angle the
    incident shock turned it, which is only possible if that angle does not
    exceed theta_max evaluated at the POST-incident Mach number.
    """
    out = []
    reach = reach_frac * float(mesh.xc[0, -1] - mesh.xc[0, 0])
    for ia in range(len(structures)):
        for ib in range(ia + 1, len(structures)):
            if not (fits[ia].get("direction_reliable") and
                    fits[ib].get("direction_reliable")):
                continue
            pt = _seg_intersect(fits[ia], fits[ib], reach=reach)
            if pt is None:
                continue
            x, y, extrapolated = pt
            if not (mesh.xc[0, 0] <= x <= mesh.xc[0, -1]):
                continue
            A, B = structures[ia], structures[ib]
            near_axis = abs(y) < 2.0 * float(np.min(mesh.dy_cell))
            kind = "shock-shock intersection (opposite families)"
            if A.get("family") == B.get("family"):
                kind = "same-family shock coalescence"
            if near_axis:
                kind = "reflection at the axis / symmetry plane"
            rr = None
            inc = A if (A["M1_upstream"] or 0) >= (B["M1_upstream"] or 0) else B
            try:
                rho1 = float(inc["upstream_state"]["rho"])
                p1 = float(inc["upstream_state"]["p_Pa"])
                V1 = float(inc["upstream_state"]["V_m_per_s"])
                th = math.radians(abs(float(inc["deflection_theta_deg"])))
                sol = oblique_solve(gas, rho1, V1, p1, th, "weak")
                if sol and sol.get("attached"):
                    st2 = normal_shock_real(gas, rho1, V1 * math.sin(
                        math.radians(sol["beta_deg"])), p1)
                    M2 = sol["M2"]
                    T2 = st2["T2"]
                    rho2 = st2["rho2"]
                    p2 = st2["p2"]
                    a2 = float(np.asarray(gas.sound_speed_Trho(
                        np.array(T2), np.array(rho2))).ravel()[0])
                    tab2 = oblique_shock_table(gas, rho2, M2 * a2, p2)
                    if tab2:
                        possible = th <= tab2["theta_max"]
                        rr = dict(
                            incident_theta_deg=sig(math.degrees(th)),
                            M_behind_incident=sig(M2),
                            theta_max_behind_incident_deg=sig(
                                math.degrees(tab2["theta_max"])),
                            regular_reflection_possible=bool(possible),
                            predicted_type=("regular reflection (RR)" if possible
                                            else "Mach reflection (MR): a Mach "
                                                 "stem and a triple point must "
                                                 "form"),
                            criterion="detachment: RR requires theta_incident "
                                      "<= theta_max(M behind the incident shock)")
            except (TypeError, ValueError, KeyError):
                rr = None
            slip_near = None
            for sl in slips:
                if (abs(sl["centroid_x"] - x) < 0.6 and
                        abs(sl["centroid_y"] - y) < 0.6):
                    slip_near = sl["structure_id"]
                    break
            out.append(dict(
                structures=[A["structure_id"], B["structure_id"]],
                type=kind, x=sig(x), y=sig(y),
                wave_angles_deg=[sig(fits[ia]["angle_deg"]),
                                 sig(fits[ib]["angle_deg"])],
                extrapolation_distance=sig(extrapolated),
                extrapolation_note="0 means the two captured wave segments "
                                   "actually overlap; a large value means the "
                                   "crossing was inferred by extending them",
                families=[A.get("family"), B.get("family")],
                reflection_analysis=rr,
                slip_line_structure_id=slip_near,
                triple_point=bool(slip_near is not None and rr and
                                  not rr["regular_reflection_possible"]),
            ))
    return out[:max_report]


# ------------------------------------------------------------ characteristics
def trace_characteristics(mesh, F, family="+", n_lines=10, max_steps=6000):
    """Integrate Mach lines dy/dx = tan(theta +/- mu) through the field.

    Same-family characteristics that converge are compression waves coalescing
    into a shock -- this finds an internal shock BEFORE the pressure jump is
    large enough for the shock sensor to see it.
    """
    sgn = 1.0 if family == "+" else -1.0
    xn = mesh.xn
    hn = mesh.hn
    x_start = 0.0
    x_end = float(mesh.Ld)
    seeds = np.linspace(0.02, 0.95, n_lines)
    lines = []
    for s in seeds:
        xx = x_start + s * (x_end - x_start)
        hx = float(np.interp(xx, xn, hn))
        yy = 0.985 * hx if mesh.axisym else 0.985 * hx
        path = [(xx, yy)]
        for _ in range(max_steps):
            i = int(np.clip(np.searchsorted(mesh.xc[0], xx) - 1, 0, mesh.nx - 1))
            hx = float(np.interp(xx, xn, hn))
            eta = (yy / hx) if mesh.axisym else (0.5 * (yy / hx + 1.0))
            j = int(np.clip(int(eta * mesh.ny), 0, mesh.ny - 1))
            M = float(F["M"][j, i])
            if not np.isfinite(M) or M <= 1.001:
                break
            mu = math.asin(min(1.0 / M, 1.0))
            th = math.radians(float(F["angle_deg"][j, i]))
            slope = math.tan(th + sgn * mu)
            step = 0.35 * float(mesh.dx_cell[j, i])
            xx += step
            yy += slope * step
            if xx > x_end or abs(yy) > hx * 1.02:
                break
            path.append((xx, yy))
        if len(path) > 3:
            lines.append(np.array(path))
    coalesce = None
    if len(lines) >= 2:
        best = None
        for a, b in zip(lines[:-1], lines[1:]):
            n = min(len(a), len(b))
            if n < 5:
                continue
            d = np.hypot(a[:n, 0] - b[:n, 0], a[:n, 1] - b[:n, 1])
            k = int(np.argmin(d))
            if d[k] < 0.35 * d[0] and (best is None or d[k] < best[0]):
                best = (float(d[k]), float(a[k, 0]), float(a[k, 1]),
                        float(d[0]))
        if best:
            coalesce = dict(x=sig(best[1]), y=sig(best[2]),
                            gap_at_start=sig(best[3]), gap_at_focus=sig(best[0]),
                            note="same-family Mach lines converge here: "
                                 "compression waves are coalescing, which is "
                                 "where an internal shock forms")
    return dict(family=("C+" if family == "+" else "C-"),
                n_lines=len(lines),
                lines=[[[sig(px), sig(py)] for px, py in ln[::max(1, len(ln) // 40)]]
                       for ln in lines],
                coalescence=coalesce)


# =============================================================================
# 11.  MASS FLOW AND THRUST
# =============================================================================
def mass_flow_face(mesh, F, k):
    """Mass flow [kg/s] through i-face column k, using the upwind cell."""
    c = min(max(k - 1, 0), mesh.nx - 1)
    un = F["u"][:, c] * mesh.i_nx[:, k] + F["v"][:, c] * mesh.i_ny[:, k]
    return float(np.sum(F["rho"][:, c] * un * mesh.i_area_phys[:, k]))


def thrust(mesh, gas, cfg, F, iso, th):
    """Thrust from three independent routes.

    (a) exit-plane momentum   F = int [rho u_n u_z + (p - p_a) n_z] dA
    (b) wall pressure integral + inlet momentum influx (control-volume check)
    (c) ideal quasi-1D        F = mdot Ve + (pe - pa) Ae
    For a converged inviscid solution (a) and (b) must agree; the difference is
    a direct measure of how well the momentum equation is satisfied discretely.
    """
    # Reference pressure for the ambient correction: the pressure actually
    # acting on the outside of the nozzle.  With a plume that is p_amb; for a
    # nozzle-only run it is the imposed back pressure.
    pa = cfg.get("p_thrust_ref", cfg["p_amb"])
    ke = mesh.lip + 1                    # i-face index of the exit plane
    ce = mesh.lip
    nz_e, nr_e = mesh.i_nx[:, ke], mesh.i_ny[:, ke]
    dA_e = mesh.i_area_phys[:, ke]
    rho, u, v, p = F["rho"][:, ce], F["u"][:, ce], F["v"][:, ce], F["p"][:, ce]
    un = u * nz_e + v * nr_e
    mdot_e = float(np.sum(rho * un * dA_e))
    Fz_exit = float(np.sum((rho * un * u + (p - pa) * nz_e) * dA_e))
    Fr_exit = float(np.sum((rho * un * v + (p - pa) * nr_e) * dA_e))

    # inlet plane, outward normal = -i_n
    nz_i, nr_i = -mesh.i_nx[:, 0], -mesh.i_ny[:, 0]
    dA_i = mesh.i_area_phys[:, 0]
    rho, u, v, p = F["rho"][:, 0], F["u"][:, 0], F["v"][:, 0], F["p"][:, 0]
    un_i = u * nz_i + v * nr_i
    mdot_i = -float(np.sum(rho * un_i * dA_i))
    Fz_inlet_out = float(np.sum((rho * un_i * u + (p - pa) * nz_i) * dA_i))

    # internal wall, outward normal from the fluid
    sl = slice(0, mesh.lip + 1)
    p_top = F["p"][-1, sl]
    w_top = (p_top - pa) * mesh.j_nx[-1, sl] * mesh.j_area_phys[-1, sl]
    p_bot = F["p"][0, sl]
    w_bot = (p_bot - pa) * (-mesh.j_nx[0, sl]) * mesh.j_area_phys[0, sl]
    wall = w_top + w_bot
    zc_w = mesh.xc[0, sl]
    Fz_wall = float(np.sum(wall))
    Fz_wall_conv = float(np.sum(wall[zc_w < 0.0]))
    Fz_wall_div = float(np.sum(wall[zc_w >= 0.0]))
    thrust_wall_route = -(Fz_wall) - Fz_inlet_out

    # ideal quasi-1D
    p0, T0 = cfg["p0"], cfg["T0"]
    st_t = iso.state(1.0, p0, T0)
    mdot_ideal = st_t["rho"] * st_t["V"] * mesh.A_throat
    Me = iso.M_from_area(mesh.Ae_At, "supersonic")
    st_e = iso.state(Me, p0, T0)
    F_ideal = mdot_ideal * st_e["V"] + (st_e["p"] - pa) * mesh.A_exit

    md = mdot_e if abs(mdot_e) > 1e-30 else float("nan")
    Isp = Fz_exit / (md * G0) if md == md and md != 0 else None
    cstar = p0 * mesh.A_throat / md if md == md and md != 0 else None
    CF = Fz_exit / (p0 * mesh.A_throat)
    # normalise the route-to-route check by a scale that does not vanish for
    # low-thrust (subsonic / near-matched) cases
    denom = max(abs(Fz_exit), abs(thrust_wall_route), 0.05 * p0 * mesh.A_throat, 1e-30)
    return dict(
        exit_plane_index_i=ce,
        exit_plane_x=sig(float(mesh.xc[0, ce])),
        mass_flow_exit_kg_per_s=sig(mdot_e),
        mass_flow_inlet_kg_per_s=sig(mdot_i),
        mass_flow_ideal_choked_kg_per_s=sig(mdot_ideal),
        discharge_coefficient_Cd=sig(mdot_e / mdot_ideal if mdot_ideal else None),
        mass_balance_error_percent=sig(100.0 * abs(mdot_e - mdot_i) /
                                       max(abs(mdot_i), 1e-30)),
        thrust_exit_plane_N=sig(Fz_exit),
        exit_plane_transverse_momentum_flux_N=sig(Fr_exit),
        exit_plane_transverse_note="axisymmetric: this cancels around the "
                               "circumference; it is a symmetry diagnostic only",
        thrust_wall_integral_route_N=sig(thrust_wall_route),
        axial_force_on_wall_N=sig(-Fz_wall),
        axial_force_on_convergent_wall_N=sig(-Fz_wall_conv),
        axial_force_on_divergent_wall_N=sig(-Fz_wall_div),
        inlet_momentum_plus_pressure_N=sig(-Fz_inlet_out),
        inlet_note="with an open inlet plane the convergent wall gives a "
                   "rearward force and the inlet momentum+pressure term "
                   "supplies the rest; in a real motor the chamber closure "
                   "replaces both",
        thrust_route_difference_N=sig(Fz_exit - thrust_wall_route),
        thrust_route_disagreement_percent=sig(100.0 *
                                              abs(Fz_exit - thrust_wall_route) / denom),
        thrust_route_disagreement_reference_N=sig(denom),
        thrust_ideal_1d_N=sig(F_ideal),
        vacuum_thrust_N=sig(Fz_exit + pa * mesh.A_exit),
        thrust_at_p_ambient_N=sig(Fz_exit + (pa - cfg["p_amb"]) * mesh.A_exit),
        ambient_reference_note=(
            "thrust_exit_plane_N is corrected with p_thrust_ref = "
            f"{pa:.6g} Pa.  With --outlet supersonic and no plume the exit BC "
            "does not feel any back pressure, so the reference defaults to 0 "
            "(vacuum thrust); thrust_at_p_ambient_N re-corrects it to --p-amb."),
        specific_impulse_s=sig(Isp),
        characteristic_velocity_cstar_m_per_s=sig(cstar),
        thrust_coefficient_CF=sig(CF),
        throat_area_m2=sig(mesh.A_throat), exit_area_m2=sig(mesh.A_exit),
        ambient_pressure_Pa=sig(pa),
        note=("thrust_exit_plane_N is the ambient-corrected gross axial thrust; "
              "the wall-integral route is an independent control-volume check "
              "(they agree only when the momentum equation is converged)."),
    )


# =============================================================================
# 12.  POST-PROCESSING: plots
# =============================================================================
def mirror_field(mesh, Fld, on):
    """Reflect an axisymmetric half-plane solution about the axis.

    The solver only stores y >= 0 when --axisym is used, so a contour plot of
    the raw arrays shows half the nozzle.  Stacking the field with its mirror
    image gives the full profile.  This is display only -- no solver quantity,
    integral or JSON array is touched.  Planar runs already span -h..+h and are
    returned unchanged.
    """
    if not on:
        return mesh.xc, mesh.yc, Fld
    X = np.vstack([mesh.xc[::-1, :], mesh.xc])
    Y = np.vstack([-mesh.yc[::-1, :], mesh.yc])
    C = np.vstack([Fld[::-1, :], Fld])
    return X, Y, C


def plot_all(mesh, gas, cfg, F, th, sen, hist, path,
             Mc=None, err=None, val=None, mirror=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = mesh.xc[0, :]
    noz_n = mesh.xn <= mesh.Ld + 1e-9
    mir = bool(mirror and mesh.axisym)          # planar already spans -h..+h
    full = mir or not mesh.axisym               # is the lower half drawn?
    fields = [(F["M"], "Mach number M", "turbo", "M [-]"),
              (F["T"], "Temperature T", "inferno", "T [K]"),
              (F["rho"], "Density", "cividis", "rho [kg/m^3]"),
              (F["p"] / cfg["p0"], "Static pressure p/p0", "viridis", "p/p0 [-]"),
              (F["gamma"], "Ratio of specific heats gamma(T)", "coolwarm", "gamma [-]"),
              (F["R"], "Specific gas constant R(T,rho)", "BrBG", "R [J/kg/K]"),
              (np.sqrt(F["u"] ** 2 + F["v"] ** 2), "Velocity magnitude |V|",
               "magma", "|V| [m/s]"),
              (F["angle_deg"], "Velocity direction (flow angle)", "twilight", "deg"),
              (sen["sensor"] * sen["flag"], "Shock sensor (flagged cells only)",
               "hot", "|grad p| h / p")]

    # Size the contour rows from the real data aspect: the panels use
    # set_aspect("equal") (needed for shock angles to be read correctly), so a
    # row that is too short makes matplotlib shrink the panel WIDTH instead,
    # which is what left the plots narrow and surrounded by white space.
    FIG_W, AX_W = 13.5, 10.6
    x_ext = float(mesh.xn[-1] - mesh.xn[0])
    y_ext = float(np.max(mesh.hn)) * (2.0 if full else 1.0)
    ph = min(max(AX_W * y_ext / max(x_ext, 1e-9) + 0.85, 1.6), 5.2)
    extra = 2.4
    fig_h = ph * len(fields) + 2 * extra + 0.8
    fig = plt.figure(figsize=(FIG_W, fig_h))
    # explicit margins in INCHES: the matplotlib defaults are fractions
    # (top=0.88), which on a 45-inch-tall figure leaves several inches blank
    gs = fig.add_gridspec(len(fields) + 2, 2, hspace=0.45, wspace=0.22,
                          left=0.07, right=0.94,
                          top=1.0 - 0.55 / fig_h, bottom=0.45 / fig_h,
                          height_ratios=[ph] * len(fields) + [extra, extra])
    for k, (Fld, title, cmap, cb) in enumerate(fields):
        ax = fig.add_subplot(gs[k, :])
        Xp, Yp, Cp = mirror_field(mesh, Fld, mir)
        pc = ax.pcolormesh(Xp, Yp, Cp, shading="gouraud", cmap=cmap)
        ax.plot(mesh.xn[noz_n], mesh.hn[noz_n], "k", lw=2.2)
        if full:
            ax.plot(mesh.xn[noz_n], -mesh.hn[noz_n], "k", lw=2.2)
        if mesh.nx_plume:
            ax.plot(mesh.xn[~noz_n], mesh.hn[~noz_n], color="gray", lw=0.9, ls="--")
            if full:
                ax.plot(mesh.xn[~noz_n], -mesh.hn[~noz_n],
                        color="gray", lw=0.9, ls="--")
            ax.axvline(mesh.Ld, color="w", ls=":", lw=1.0)
        if mesh.axisym:
            ax.axhline(0.0, color="k", ls="-.", lw=0.8)
        if th["shock"].get("x_shock") is not None:
            ax.axvline(th["shock"]["x_shock"], color="w", ls="--", lw=1.2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("y (= r when axisymmetric)")
        if mir and k == 0:
            ax.text(0.005, 0.02, "lower half mirrored about the axis",
                    transform=ax.transAxes, fontsize=7, color="w",
                    ha="left", va="bottom", alpha=0.85)
        if k == len(fields) - 1:
            ax.set_xlabel("x")
        fig.colorbar(pc, ax=ax, label=cb, pad=0.01)

    ax = fig.add_subplot(gs[len(fields), :])
    cM, cT = centreline(mesh, F["M"]), centreline(mesh, F["T"])
    cR, cP = centreline(mesh, F["rho"]), centreline(mesh, F["p"])
    cV, cG = centreline(mesh, F["V"]), centreline(mesh, F["gamma"])
    for arr_, lab, col in ((cM, f"M / {np.nanmax(cM):.2f}", "r"),
                           (cT, f"T / {np.nanmax(cT):.0f} K", "darkorange"),
                           (cR, f"rho / {np.nanmax(cR):.4g}", "olive"),
                           (cP / cfg["p0"], "p / p0", "b"),
                           (cV, f"|V| / {np.nanmax(cV):.0f} m/s", "purple"),
                           (cG, f"gamma / {np.nanmax(cG):.3f}", "green")):
        ax.plot(x, arr_ / max(np.nanmax(np.abs(arr_)), 1e-30), color=col, lw=1.6, label=lab)
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    if mesh.nx_plume:
        ax.axvline(mesh.Ld, color="gray", ls="--", lw=0.9)
    ax.set_xlabel("x"); ax.set_ylabel("normalized")
    ax.set_title("Centreline profiles (each normalized by its own maximum; p by p0)")
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[len(fields) + 1, 0])
    Msol = Mc if Mc is not None else centreline(mesh, F["M"])
    ax.plot(x, th["M"], "k-", lw=2.0, label=f"quasi-1D theory ({gas.model})")
    ax.plot(x, Msol, "r--", lw=1.6, label="solver (centreline)")
    if th["shock"].get("x_shock") is not None:
        ax.axvline(th["shock"]["x_shock"], color="b", ls="--", lw=1.1,
                   label=f"1D shock x={th['shock']['x_shock']:.2f}")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.axvline(0.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("Mach"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title((f"Centreline validation  (max |dM| = {val['max_abs_delta_M']}, "
                  f"mean = {val['mean_abs_delta_M']})") if val else
                 "Centreline Mach vs quasi-1D theory", fontsize=10)

    ax = fig.add_subplot(gs[len(fields) + 1, 1])
    h = np.array(hist) if hist else np.array([[0, 1.0]])
    ax.semilogy(h[:, 0], h[:, 1], "b-", lw=1.2)
    ax.axhline(cfg["tol"], color="r", ls="--", lw=1.4, label=f"tol={cfg['tol']:.0e}")
    ax.set_xlabel("iteration"); ax.set_ylabel("relative residual")
    ax.set_title(f"Convergence: {h[-1,1]:.2e} @ iter {int(h[-1,0])}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    fig.suptitle(f"{gas.name.upper()} CD nozzle{' + plume' if mesh.nx_plume else ''} | "
                 f"p0={cfg['p0']/1e5:.3g} bar, T0={cfg['T0']:.0f} K | {mesh.profile} | "
                 f"{'axisym (mirrored)' if mir else ('axisym' if mesh.axisym else 'planar')}"
                 f" | gas model {gas.model}",
                 fontsize=13, y=1.0 - 0.12 / fig_h)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# =============================================================================
# 13.  POST-PROCESSING: JSON export with GLOBAL GRID NUMBERING
# =============================================================================
def export_json(mesh, gas, cfg, F, sen, shocks, th, iso, thr, hist, conv, it,
                path, cells_mode="full", nsig=6, waves=None, validation=None, geom=None,
                conversion=None):
    """Write the case.

    ORDERING CONTRACT
    -----------------
    grid.elements.cell_id is the master list.  EVERY array under "fields" is
    the corresponding quantity for the same element, in the same position.
    cell_id = j*nx + i  (numpy C-order ravel of an (ny, nx) array).
    """
    A = lambda a: sig_arr(a, nsig)                       # noqa: E731
    lip = mesh.lip
    cid = mesh.cell_id.ravel()

    grid = {
        "ordering": "cell_id = j*nx + i   (j = transverse index, i = axial index; "
                    "C-order ravel of an (ny, nx) array)",
        "index_ranges": {"i_axial": [0, mesh.nx - 1], "j_transverse": [0, mesh.ny - 1]},
        "n_cells": int(mesh.n_cells), "n_nodes": int(mesh.n_nodes),
        "n_axial_total": int(mesh.nx), "n_axial_nozzle": int(mesh.nx_noz),
        "n_axial_plume": int(mesh.nx_plume), "n_transverse": int(mesh.ny),
        "exit_plane_axial_index_i": int(lip),
        "plume_length": sig(mesh.Lp) if mesh.nx_plume else None,
        "plume_length_over_L_divergent": sig(mesh.Lp / mesh.Ld) if mesh.nx_plume else None,
        "plume_cell_size_first": sig(mesh.plume_dx_first),
        "plume_cell_size_last": sig(mesh.plume_dx_last),
        "plume_cell_growth_per_cell": sig(mesh.plume_growth),
        "throat_axial_index_i": int(mesh.i_throat),
        "axisymmetric": bool(mesh.axisym),
    }
    if cells_mode != "none":
        grid["elements"] = {
            "cell_id": [int(x) for x in cid],
            "i_axial": [int(x) for x in mesh.cell_i.ravel()],
            "j_transverse": [int(x) for x in mesh.cell_j.ravel()],
            "x_center_m": A(mesh.xc), "y_center_m": A(mesh.yc),
            "planar_area_m2": A(mesh.area),
            "cell_volume_m3": A(mesh.vol_phys),
            "dx_m": A(mesh.dx_cell), "dy_m": A(mesh.dy_cell),
            "zone": [str(x) for x in mesh.zone.ravel()],
        }
        grid["node_coordinates"] = {
            "ordering": "node_id = jn*(nx+1) + in",
            "node_id": [int(x) for x in np.arange(mesh.n_nodes)],
            "x_m": A(mesh.Xn), "y_m": A(mesh.Yn),
        }

    fields = None
    if cells_mode == "full":
        fields = {
            "_ordering": "identical to grid.elements.cell_id",
            "cell_id": [int(x) for x in cid],
            "temperature_K": A(F["T"]),
            "pressure_Pa": A(F["p"]),
            "density_kg_per_m3": A(F["rho"]),
            "velocity_magnitude_m_per_s": A(F["V"]),
            "velocity_x_m_per_s": A(F["u"]),
            "velocity_y_m_per_s": A(F["v"]),
            "velocity_direction_deg_from_x_axis": A(F["angle_deg"]),
            "mach_number": A(F["M"]),
            "sound_speed_m_per_s": A(F["a"]),
            "gamma": A(F["gamma"]),
            "R_specific_J_per_kgK": A(F["R"]),
            "cp_J_per_kgK": A(F["cp"]),
            "cv_J_per_kgK": A(F["cv"]),
            "dissociation_fraction_alpha": A(F["alpha_dissociation"]),
            "total_temperature_K": A(F["T0"]),
            "total_pressure_Pa": A(F["p0"]),
            "entropy_J_per_kgK": A(F["entropy"]),
            "mach_angle_deg": A(F["mach_angle_deg"]),
            "prandtl_meyer_angle_deg": A(F["prandtl_meyer_deg"]),
            "dynamic_pressure_Pa": A(F["q_dynamic"]),
            "mass_flux_kg_per_m2s": A(F["mass_flux"]),
            "total_enthalpy_J_per_kg": A(F["total_enthalpy"]),
            "grad_p_z_Pa_per_m": A(F["grad_p_x"]),
            "grad_p_r_Pa_per_m": A(F["grad_p_y"]),
            "divergence_V_per_s": A(F["div_V"]),
            "vorticity_per_s": A(F["curl_V"]),
            "shock_sensor": A(sen["sensor"]),
            "shock_normal_mach": A(sen["Mn"]),
            "ducros_sensor": A(sen["ducros"]),
            "is_shock_cell": [int(x) for x in sen["flag"].ravel()],
            "shock_structure_id": [int(x) for x in shocks["labels"].ravel()],
            "is_expansion_cell": ([int(x) for x in waves["expansion_flag"].ravel()]
                                  if waves else None),
            "expansion_structure_id": ([int(x) for x in waves["expansion_labels"].ravel()]
                                       if waves else None),
            "is_slip_line_cell": ([int(x) for x in waves["slip_flag"].ravel()]
                                  if waves else None),
        }

    md_in = thr["mass_flow_inlet_kg_per_s"]
    md_ex = thr["mass_flow_exit_kg_per_s"]
    case = {
        "description": "Converging-diverging nozzle Euler solution (cd_nozzle_solver_axiconv_openplume_hybrid.py)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_ordering_contract":
            "grid.elements.cell_id is the master element list; every array in "
            "'fields' is in exactly that order (cell_id = j*nx + i).",
        "solver": {
            "scheme": f"2D finite volume, flux={cfg.get('flux','hllc')} "
                      f"(HLL at strong shocks, HLLC elsewhere when hybrid; "
                      f"thr={cfg.get('hll_thr')}, dilate={cfg.get('hll_dilate')}), "
                      "MUSCL-minmod, SSP-RK2, local dt, IRS",
            "hll_cells_last_iteration": cfg.get("_hll_cells"),
            "axisymmetric_formulation":
                "conservative r-weighted (d/dt(r U A) + sum r_f F.n L = p A e_r)"
                if mesh.axisym else "planar",
            "order": cfg["order"], "cfl": cfg["cfl"], "tolerance": cfg["tol"],
            "iterations": int(it), "converged": bool(conv),
            "final_relative_residual": sig(hist[-1][1], 4) if hist else None,
            "residual_smoothing_eps": cfg["irs_eps"],
            "residual_smoothing_passes": cfg["irs_passes"],
            "limitations": "inviscid: no boundary layers, no separation, no "
                           "turbulence quantities; single species; steady",
        },
        "gas_model": gas.describe(),
        "gas_model_field_ranges": {
            "gamma_min_max": [sig(float(np.nanmin(F["gamma"]))),
                              sig(float(np.nanmax(F["gamma"])))],
            "R_min_max_J_per_kgK": [sig(float(np.nanmin(F["R"]))),
                                    sig(float(np.nanmax(F["R"])))],
            "T_min_max_K": [sig(float(np.nanmin(F["T"]))),
                            sig(float(np.nanmax(F["T"])))],
            "alpha_min_max": [sig(float(np.nanmin(F["alpha_dissociation"]))),
                              sig(float(np.nanmax(F["alpha_dissociation"])))],
        },
        "operating_conditions": {
            "p0_Pa": cfg["p0"], "T0_K": cfg["T0"],
            "outlet_type": cfg["outlet"], "pb_Pa": cfg.get("pb"),
            "p_ambient_Pa": cfg.get("p_amb"), "T_ambient_K": cfg.get("T_amb"),
            "NPR_p0_over_pamb": sig(cfg["p0"] / cfg["p_amb"]),
        },
        "geometry": {
            "profile": mesh.profile, "axial_clustering": mesh.cluster_mode,
            "axial_clustering_strength": mesh.cluster, "L_convergent": mesh.Lc,
            "L_divergent": mesh.Ld, "L_plume": mesh.Lp,
            "h_inlet": mesh.h_in, "h_throat": mesh.h_t, "h_exit": mesh.h_e,
            "h_farfield": mesh.Hmax,
            "super_ellipse_radial_exponent_n": sig(mesh.se_n),
            "super_ellipse_axial_exponent_p": sig(mesh.se_p),
            "super_ellipse_truncation": sig(mesh.se_trunc),
            "super_ellipse_exit_wall_angle_deg": sig(mesh.se_exit_angle_deg),
            "requested_exit_wall_angle_deg": mesh.theta_e,
            "achieved_exit_wall_angle_deg": sig(mesh.theta_e_actual),
            "bell_initial_wall_angle_deg_requested": mesh.theta_n,
            "bell_initial_wall_angle_deg_achieved": sig(mesh.theta_n_actual),
            "mean_wall_angle_deg": sig(math.degrees(math.atan(
                (mesh.h_e - mesh.h_t) / mesh.Ld))),
            "exit_wall_angle_note":
                "for --profile ellipse the smallest exit angle compatible with "
                "a zero-slope throat is atan(2*(h_e-h_t)/L_div); a smaller "
                "request is clamped to it",
            "planar_or_axisymmetric": ("axisymmetric" if mesh.axisym
                                       else "planar (per unit depth)"),
            "D_throat": sig(2.0 * mesh.h_t), "D_exit": sig(2.0 * mesh.h_e),
            "D_inlet": sig(2.0 * mesh.h_in),
            "D_farfield": sig(2.0 * mesh.Hmax) if mesh.nx_plume else None,
            "area_ratio_Ae_At": sig(mesh.Ae_At),
            "throat_area_m2": sig(mesh.A_throat), "exit_area_m2": sig(mesh.A_exit),
        },
        "grid": grid,
        "boundary_conditions": {
            "inlet": "subsonic stagnation reservoir (p0, T0); static pressure "
                     "extrapolated from the interior, remaining state from the "
                     "variable-gamma isentrope; velocity along the face normal",
            "outlet": cfg["outlet"] + " (characteristic; reverse flow treated as "
                                      "an ambient reservoir)",
            "lower_boundary": "symmetry axis (exact for the r-weighted "
                              "conservative form: the axis face area is zero)"
                              if mesh.axisym else "inviscid slip wall",
            "nozzle_wall": "inviscid slip wall, mirrored about the face normal",
            "farfield": (("1D Riemann-invariant far field (legacy plume)"
                          if cfg.get("legacy_plume") else
                          "pressure opening: outflow p = p_amb (supersonic "
                          "outflow extrapolated); inflow ambient p_amb, T_amb")
                         if mesh.nx_plume else "n/a"),
            "plume_outer_boundary_ramp_length": (
                sig(mesh.plume_ramp) if mesh.plume_ramp is not None
                else sig(0.5 * mesh.Lp)),
        },
        "shock_initiation_parameters": th["regimes"],
        "shock_analysis": {
            "detector": {
                "criteria": ["|grad p| * h_cell / p > sp_min",
                             "div V < 0 (compression)",
                             "normal Mach (V . grad p/|grad p|)/a > Mn_min"],
                "sp_min": cfg["shock_sp_min"], "Mn_min": cfg["shock_mn_min"],
                "probe_offset_cells": cfg["shock_probe"],
            },
            "n_shock_cells": int(sen["flag"].sum()),
            "n_shock_structures": len(shocks["structures"]),
            "structures": shocks["structures"],
            "quasi_1d_normal_shock": {
                k: (sig(v) if isinstance(v, (int, float)) else v)
                for k, v in th["shock"].items() if k != "jump"
            },
            "quasi_1d_normal_shock_jump": (
                {k: sig(v) for k, v in th["shock"]["jump"].items()}
                if th["shock"].get("jump") else None),
            "plume": shocks["plume"],
        },
        "wave_analysis": ({k: v for k, v in waves.items()
                           if k not in ("expansion_labels", "slip_labels",
                                        "expansion_flag", "slip_flag")}
                          if waves else None),
        "thrust_and_performance": thr,
        "centreline_validation": validation,
        "key_results": {
            "exit_axis_mach": sig(float(F["M"][0, lip]), 4),
            "exit_axis_T_K": sig(float(F["T"][0, lip]), 4),
            "exit_axis_p_Pa": sig(float(F["p"][0, lip]), 5),
            "exit_axis_velocity_m_per_s": sig(float(F["V"][0, lip]), 5),
            "exit_mass_averaged_mach": sig(float(
                np.sum(F["rho"][:, lip] * F["u"][:, lip] * F["M"][:, lip] *
                       mesh.i_area_phys[:, lip + 1]) /
                max(np.sum(F["rho"][:, lip] * F["u"][:, lip] *
                           mesh.i_area_phys[:, lip + 1]), 1e-30)), 4),
            "peak_mach": sig(float(np.nanmax(F["M"])), 4),
            "peak_speed_m_per_s": sig(float(np.nanmax(F["V"])), 5),
            "T_range_K": [sig(float(np.nanmin(F["T"])), 4),
                          sig(float(np.nanmax(F["T"])), 4)],
            "p_range_Pa": [sig(float(np.nanmin(F["p"])), 4),
                           sig(float(np.nanmax(F["p"])), 4)],
            "rho_range_kg_m3": [sig(float(np.nanmin(F["rho"])), 4),
                                sig(float(np.nanmax(F["rho"])), 4)],
            "mass_flow_inlet_kg_s": md_in, "mass_flow_exit_kg_s": md_ex,
            "mass_balance_error_percent": thr["mass_balance_error_percent"],
            "thrust_N": thr["thrust_exit_plane_N"],
            "specific_impulse_s": thr["specific_impulse_s"],
        },
        "nozzle_wall_contour": {
            "x": A(mesh.xn[mesh.xn <= mesh.Ld + 1e-9]),
            "h_wall": A(mesh.hn[mesh.xn <= mesh.Ld + 1e-9]),
        },
        "centreline_profiles": {
            "_note": "jet centreline: row j = 0 when axisymmetric, and the mean "
                     "of the two rows straddling y = 0 in planar mode (where "
                     "j = 0 is the LOWER WALL, not the centreline)",
            "j_rows_averaged": [int(j) for j in mesh.centre_rows],
            "cell_id_of_first_row": [int(v) for v in mesh.cell_id[mesh.centre_rows[0]]],
            "x": A(mesh.xc[mesh.centre_rows[0]]),
            "mach": A(centreline(mesh, F["M"])),
            "temperature_K": A(centreline(mesh, F["T"])),
            "density_kg_m3": A(centreline(mesh, F["rho"])),
            "pressure_Pa": A(centreline(mesh, F["p"])),
            "velocity_m_per_s": A(centreline(mesh, F["V"])),
            "velocity_direction_deg": A(centreline(mesh, F["angle_deg"])),
            "gamma": A(centreline(mesh, F["gamma"])),
            "R_J_per_kgK": A(centreline(mesh, F["R"])),
            "total_pressure_Pa": A(centreline(mesh, F["p0"])),
            "entropy_J_per_kgK": A(centreline(mesh, F["entropy"])),
            "shock_sensor": A(centreline(mesh, sen["sensor"])),
        },
        "exit_plane_profiles": {
            "_note": "i = exit index, j = 0..ny-1 (bottom wall/axis to top wall)",
            "cell_id": [int(x) for x in mesh.cell_id[:, lip]],
            "y": A(mesh.yc[:, lip]), "mach": A(F["M"][:, lip]),
            "temperature_K": A(F["T"][:, lip]), "density_kg_m3": A(F["rho"][:, lip]),
            "pressure_Pa": A(F["p"][:, lip]),
            "velocity_x_m_per_s": A(F["u"][:, lip]),
            "velocity_y_m_per_s": A(F["v"][:, lip]),
            "velocity_direction_deg": A(F["angle_deg"][:, lip]),
            "gamma": A(F["gamma"][:, lip]), "R_J_per_kgK": A(F["R"][:, lip]),
        },
        "theory_1d_axis": {
            "x": A(mesh.xc[0]), "mach": A(th["M"]), "pressure_Pa": A(th["p"]),
            "temperature_K": A(th["T"]), "density_kg_m3": A(th["rho"]),
            "velocity_m_per_s": A(th["u"]), "area_ratio_A_over_At": A(th["AR"]),
        },
        "isentrope_reference": {
            "gamma_at_throat": sig(iso.gamma_throat),
            "p_throat_over_p0": sig(iso.p_p0_throat),
            "T_throat_over_T0": sig(iso.T_T0_throat),
            "note": "variable-gamma sonic ratios; compare with the constant-gamma "
                    "values (2/(g+1))^(g/(g-1)) and 2/(g+1)",
        },
        "convergence_history": {"iteration": [int(h[0]) for h in hist],
                                "relative_residual": sig_arr([h[1] for h in hist], 4)},
    }
    if fields is not None:
        case["fields"] = fields
    if conversion is not None:
        case["solver"]["axisymmetric_formulation"] = (
            "planar 2D Euler solve, converted to axisymmetric after convergence "
            "(see planar_to_axisymmetric_conversion)")
        case["planar_to_axisymmetric_conversion"] = conversion

    with open(path, "w") as f:
        json.dump(case, f, indent=1, allow_nan=False)
    print(f"  saved {path}  ({os.path.getsize(path)/1e6:.2f} MB, "
          f"{mesh.n_cells} numbered elements)")
    return case

# =============================================================================
# 13b. PLANAR -> AXISYMMETRIC CONVERTER  (post-processing only)
# =============================================================================
def planar_to_axisym(mesh, U):
    """Stitch a converged PLANAR solution onto an axisymmetric configuration.

    The solver itself is untouched: it marches the planar full channel
    y in [-h, +h] exactly as before.  This runs afterwards, on the converged
    state, and only changes how that data is laid out and integrated:

      mesh  The planar node rows j = ny/2 .. ny (y = 0 .. +h) are, node for
            node, the rows of an axisymmetric Mesh with ny/2 transverse cells
            (r = eta * h).  That Mesh is built from the same geometry with
            axisym=True, so every metric switches to the axisymmetric one:
            r-weighted face areas, cell volumes 2 pi r A, A = pi r^2, j = 0 is
            the symmetry axis, cell_id = j*nx + i renumbered over the half.
      data  Each upper-half cell (y > 0) is averaged with its mirror image in
            the lower half (y < 0) with the transverse momentum sign flipped,
            so the stitched field is exactly symmetric about the axis and
            v = 0 in the mean on it.  A converged planar solution is already
            symmetric; the residual asymmetry is reported.

    The returned field is the planar solution: it is NOT re-solved with the
    axisymmetric (r-weighted) equations, and the planar and axisymmetric
    area-Mach relations differ (A ~ h vs A ~ r^2).  Integrals such as mass
    flow and thrust are recomputed on the axisymmetric geometry.
    """
    if mesh.axisym:
        raise ValueError("planar_to_axisym() needs a planar mesh")
    if mesh.ny % 2:
        raise ValueError(f"planar_to_axisym() needs an even --ny (got {mesh.ny}) "
                         "so that y = 0 is a node line")
    jm = mesh.ny // 2
    ax = Mesh(nx_noz=mesh.nx_noz, ny=jm, nx_plume=mesh.nx_plume,
              Lc=mesh.Lc, Ld=mesh.Ld, Lp=mesh.Lp,
              h_in=mesh.h_in, h_t=mesh.h_t, h_e=mesh.h_e, Hmax=mesh.Hmax,
              profile=mesh.profile, cluster=mesh.cluster, axisym=True,
              theta_n=mesh.theta_n, theta_e=mesh.theta_e,
              cluster_mode=mesh.cluster_mode,
              se_n=mesh.se_n, se_p=mesh.se_p, se_trunc=mesh.se_trunc,
              plume_ramp=mesh.plume_ramp)
    scale = max(float(np.max(np.abs(mesh.Yn))), 1e-30)
    node_err = max(float(np.max(np.abs(ax.Xn - mesh.Xn[jm:]))),
                   float(np.max(np.abs(ax.Yn - mesh.Yn[jm:])))) / scale
    if node_err > 1e-9:
        raise RuntimeError(f"planar upper half and axisymmetric nodes differ "
                           f"(rel {node_err:.2e})")

    upper = U[:, jm:, :]
    lower = U[:, jm - 1::-1, :].copy()          # row jm-1-k mirrors row jm+k
    lower[2] *= -1.0                            # rho*v changes sign
    U_ax = 0.5 * (upper + lower)

    ref = np.maximum(np.max(np.abs(U_ax), axis=(1, 2)), 1e-30)
    asym = np.max(np.abs(upper - lower), axis=(1, 2)) / ref
    info = {
        "source": "planar full-channel solution (y in [-h, +h]) stitched onto "
                  "an axisymmetric half-plane (r = y >= 0)",
        "method": "upper-half planar nodes reused as the axisymmetric mesh; "
                  "each r > 0 cell = mean of the planar cells at +y and -y "
                  "(transverse momentum sign flipped for -y); mesh metrics, "
                  "areas, volumes, mass flow, thrust and every derived "
                  "quantity recomputed on the axisymmetric geometry",
        "governing_equations_of_the_field": "planar 2D Euler (the field was not "
                                            "re-solved in r-weighted form)",
        "planar_mesh_ny_nx": [int(mesh.ny), int(mesh.nx)],
        "axisymmetric_mesh_ny_nx": [int(ax.ny), int(ax.nx)],
        "planar_row_for_axisym_row_j": f"j_planar = {jm} + j  (mirror: {jm - 1} - j)",
        "node_match_max_rel_error": sig(node_err, 3),
        "planar_asymmetry_max_rel": {
            k: sig(float(a), 3) for k, a in
            zip(["rho", "rho_u", "rho_v", "rho_E"], asym)},
        "planar_areas_for_reference": {
            "throat_area_per_unit_depth_m": sig(mesh.A_throat),
            "exit_area_per_unit_depth_m": sig(mesh.A_exit),
            "area_ratio_Ae_At": sig(mesh.Ae_At)},
    }
    return ax, U_ax, info


# =============================================================================
# 14.  CENTRELINE VALIDATION AGAINST QUASI-1D THEORY
# =============================================================================
def centreline_index(mesh):
    """Rows to average to get the centreline (see Mesh.centre_rows).

    Planar full channel: y = 0 is a NODE line, so it lies on a cell FACE when
    ny is even.  The original code sampled Mf[ny//2], the first cell ABOVE the
    centreline, which is a systematic off-axis bias.  Axisymmetric: row 0.
    """
    return list(mesh.centre_rows)


def centreline(mesh, field):
    rows = centreline_index(mesh)
    return np.mean(np.stack([field[j, :] for j in rows], axis=0), axis=0)


def validate_centreline(mesh, gas, cfg, F, th):
    """Compare the solver centreline with the quasi-1D reference."""
    Mc = centreline(mesh, F["M"])
    M1d = th["M"]
    ok = np.isfinite(M1d)
    err = np.full_like(Mc, np.nan)
    err[ok] = np.abs(Mc[ok] - M1d[ok])
    it = mesh.i_throat
    lip = mesh.lip
    AR = mesh.area_ratio_cells()
    trans = ok & (np.abs(M1d - 1.0) < 0.2)
    err_ns = np.where(ok & ~trans, err, np.nan)
    if not np.any(np.isfinite(err_ns)):
        err_ns = err
    out = dict(
        throat_axial_index_i=int(it),
        throat_cell_x=sig(float(mesh.xc[0, it]), 5),
        throat_cell_area_ratio_A_over_At=sig(float(AR[it]), 8),
        throat_mach_solver=sig(float(Mc[it]), 5),
        throat_mach_ideal=1.0,
        throat_mach_theory_at_cell=sig(float(M1d[it]) if ok[it] else None, 5),
        throat_comparison_note=(
            "dM/d(A/A*) is infinite at M = 1, so evaluating the 1D branch at "
            "the throat CELL CENTRE (A/At = "
            f"{AR[it]:.6f}, not exactly 1) is ill-conditioned: a 1% area error "
            "there is a ~15% Mach error.  Compare throat_mach_solver with "
            "throat_mach_ideal = 1, and judge the solution on the exit Mach "
            "and on mean_abs_delta_M instead."),
        exit_axial_index_i=int(lip),
        exit_mach_solver=sig(float(Mc[lip]), 5),
        exit_mach_theory=sig(float(M1d[lip]) if ok[lip] else None, 5),
        max_abs_delta_M=sig(float(np.nanmax(err)), 5),
        mean_abs_delta_M=sig(float(np.nanmean(err)), 5),
        max_abs_delta_M_outside_transonic=sig(float(np.nanmax(err_ns)), 5),
        mean_abs_delta_M_outside_transonic=sig(float(np.nanmean(err_ns)), 5),
        transonic_band_excluded="|M_1D - 1| < 0.2",
        transonic_note=(
            "quasi-1D is not valid through the transonic region: in a real 2D "
            "or axisymmetric throat the sonic line is CURVED, so the centreline "
            "Mach at the geometric throat is below 1 and the wall Mach is above "
            "it.  That is also why the discharge coefficient is below 1.  Judge "
            "the solver on the values outside the transonic band."),
        centreline_rows_averaged=[int(j) for j in centreline_index(mesh)],
        note="the quasi-1D reference uses the SAME variable-gamma gas as the "
             "solver, so the comparison is not contaminated by a gamma "
             "mismatch; it is only meaningful where the nozzle is choked and "
             "shock free",
    )
    return Mc, err, out


# =============================================================================
# 15.  PRESETS + CLI + MAIN
# =============================================================================
PRESETS = {
    # name                     : argument overrides
    "validated-choked": dict(gas="air", gas_model="cpg", gamma=1.4,
                             p0=1.0e5, T0=300.0, outlet="supersonic",
                             plume=False, profile="ellipse"),
    "shock-in-nozzle":  dict(gas="air", p0=1.0e5, T0=300.0, pb=0.9e5,
                             outlet="pressure", plume=False, profile="cosine"),
    # p_exit/p_amb ~ 0.7: mildly overexpanded, oblique shock train (diamonds).
    # p0 = 3 bar (the earlier value) gives p_exit/p_amb = 0.33, which is so far
    # off design that the inviscid jet shocks down through a single strong Mach
    # disc and never recovers a supersonic core -- no cell structure at all.
    "overexpanded-jet": dict(gas="air", p0=6.46e5, T0=300.0, p_amb=1.01325e5,
                             plume=True, profile="bell"),
    # p_exit/p_amb ~ 2.0: underexpanded, expansion fan + barrel shock + Mach disc
    "underexpanded-jet": dict(gas="air", p0=1.846e6, T0=300.0, p_amb=1.01325e5,
                              plume=True, profile="bell"),
    "strongly-overexpanded": dict(gas="air", p0=3.0e5, T0=300.0, p_amb=1.01325e5,
                                  plume=True, profile="bell"),
    "hot-h2-plume":     dict(gas="h2", gas_model="tpg", p0=1.8e6, T0=2500.0,
                             plume=True, profile="cosine", axisym=True),
    "h2-dissociating":  dict(gas="h2", gas_model="idg", p0=2.2e6, T0=4000.0,
                             plume=True, profile="bell", axisym=True),
    "axisym-design":    dict(gas="air", p0=2.575e6, T0=300.0, axisym=True,
                             outlet="supersonic", plume=False, profile="bell"),
}

_MODEL_ALIAS = {"thermal": "tpg", "calorific": "cpg", "perfect": "cpg",
                "tpg": "tpg", "cpg": "cpg", "idg": "idg"}

# -----------------------------------------------------------------------------
# NOZZLE GEOMETRY CONFIGURATION  <-- EDIT HERE to change the nozzle
# -----------------------------------------------------------------------------
# All lengths are in the same (arbitrary, consistent) unit.  For the planar 2D
# nozzle the "diameter" is the full channel height (2 x half-height); in
# --axisym mode it is the true diameter (2 x radius).
#
#   D_throat : throat diameter        (minimum-area cross section)
#   D_exit   : exit diameter          (end of the diverging section)
#   D_inlet  : inlet diameter         (start of the converging section)
#   L_div    : diverging-section length (throat -> exit)
#   L_conv   : converging-section length (inlet -> throat)
#   profile  : diverging wall contour -- one of
#                "ellipse"        quarter ellipse, parallel at the exit lip
#                "ellipse-legacy" the ORIGINAL contour (vertical at the lip)
#                "conical"        straight wall, constant divergence angle
#                "cosine"         cosine bell, parallel at throat AND exit
#                "bell"           quadratic-Bezier Rao-type bell
#   cluster  : axial stretching strength (larger = finer at the throat)
DEFAULT_GEOM = {
    "D_throat": 1.2,
    "D_exit":   2.2,
    "D_inlet":  2.4,
    "L_div":    2.0,
    "L_conv":   1.0,
    "profile":  "ellipse",
    "cluster":  2.2,
}


def analyse_state(mesh, gas, cfg, iso, U, args):
    """Post-processing of a converged state U on `mesh`: derived fields,
    shock detection / characterisation and the Tier-1 wave analysis.
    Moved verbatim out of main() so that it can be re-run on the
    axisymmetric mesh produced by planar_to_axisym()."""
    W = gas.cons_to_prim(U)
    F = derived_fields(mesh, gas, cfg, W)
    sen = shock_sensor(mesh, F, args.shock_sp_min, args.shock_mn_min)
    lab, structs = characterise_shocks(mesh, gas, F, sen, args.shock_probe)
    cells, spacing = axis_shock_cells(mesh, F)
    shocks = dict(labels=lab, structures=structs,
                  plume=plume_shock_metrics(mesh, gas, cfg, iso, F,
                                            cells, spacing))
    # ---- Tier-1 wave analysis: families, fans, slip lines, interactions ----
    fits = [fit_wave_line(mesh, lab == st["structure_id"]) for st in structs]
    for st, fv in zip(structs, fits):
        i0, j0 = st["strongest_cell_ij"]
        st["fitted_wave_angle_deg"] = sig(fv["angle_deg"])
        st["fitted_length"] = sig(fv["length"])
        st["fitted_thickness_rms"] = sig(fv["thickness_rms"])
        st["family"] = wave_family(fv, F["u"][j0, i0], F["v"][j0, i0])
        try:
            ju_, iu_ = st["upstream_cell_ij"]
            fx, fy = float(F["u"][ju_, iu_]), float(F["v"][ju_, iu_])
        except (KeyError, TypeError):
            fx, fy = float(F["u"][j0, i0]), float(F["v"][j0, i0])
        nf = math.hypot(fx, fy) or 1.0
        cosb = abs((fv["dx"] * fx + fv["dy"] * fy) / nf)
        st["beta_from_fitted_line_deg"] = (
            sig(math.degrees(math.acos(min(cosb, 1.0))))
            if fv.get("direction_reliable") else None)
        st["beta_note"] = ("beta_from_fitted_line_deg is the GEOMETRIC angle "
                           "between the fitted wave and the upstream velocity; "
                           "wave_angle_beta_deg comes from the local pressure "
                           "gradient.  The two are independent, so when they "
                           "disagree the wave is not resolved well enough for "
                           "either to be trusted.")
        bg = st.get("wave_angle_beta_deg")
        bf = st.get("beta_from_fitted_line_deg")
        st["wave_geometry_consistent"] = (
            bool(abs(float(bg) - float(bf)) < 15.0)
            if (bg is not None and bf is not None) else None)
        st["wave_resolution_warning"] = (
            None if st["wave_geometry_consistent"] in (True, None) else
            "the gradient-based and geometry-based wave angles differ by more "
            "than 15 deg: this structure is under-resolved, so beta, theta and "
            "the jump ratios taken from it are not reliable")
        try:
            th_defl = math.radians(abs(float(
                st.get("deflection_from_velocity_deg")
                if st.get("deflection_from_velocity_deg") is not None
                else st["deflection_theta_deg"])))
            rho1 = float(st["upstream_state"]["rho"])
            V1 = float(st["upstream_state"]["V_m_per_s"])
            p1 = float(st["upstream_state"]["p_Pa"])
            weak = oblique_solve(gas, rho1, V1, p1, th_defl, "weak")
            strong = oblique_solve(gas, rho1, V1, p1, th_defl, "strong")
            st["theta_beta_M_weak"] = weak
            st["theta_beta_M_strong"] = strong
            bm = st.get("wave_angle_beta_deg")
            if bm is not None and weak and weak.get("attached"):
                dw = abs(float(bm) - float(weak["beta_deg"]))
                ds_ = (abs(float(bm) - float(strong["beta_deg"]))
                       if strong and strong.get("attached") else 1e9)
                st["branch_matched"] = "weak" if dw <= ds_ else "strong"
                st["beta_measured_minus_theory_deg"] = sig(min(dw, ds_))
                st["theta_beta_M_note"] = (
                    "a Mach disc / normal shock sits on the STRONG branch, so a "
                    "weak-branch comparison there is meaningless")
        except (TypeError, ValueError, KeyError):
            st["theta_beta_M_check"] = None
    exp_sen = expansion_sensor(mesh, F)
    elab, expansions = characterise_expansions(mesh, gas, F, exp_sen)
    slip = slip_line_sensor(mesh, F, gas)
    slab, slips = characterise_slip_lines(mesh, F, slip)
    interactions = find_interactions(mesh, gas, F, structs, fits, slips)
    chars = [trace_characteristics(mesh, F, "+"), trace_characteristics(mesh, F, "-")]
    waves = dict(expansion_fans=expansions, slip_lines=slips,
                 interactions=interactions, characteristics=chars,
                 expansion_labels=elab, slip_labels=slab,
                 n_expansion_cells=int(exp_sen["flag"].sum()),
                 n_slip_cells=int(slip["flag"].sum()),
                 expansion_flag=exp_sen["flag"], slip_flag=slip["flag"])
    return (F, sen, shocks, structs, waves, expansions, slips,
            interactions, chars)


def build_parser():
    ap = argparse.ArgumentParser(
        description="2D Euler CD-nozzle solver (single file) with real-gas "
                    "thermodynamics, shock diagnostics, thrust accounting and "
                    "numbered-grid JSON output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS),
                    help="parameter bundle (see --list-presets)")
    ap.add_argument("--list-presets", action="store_true")

    n = ap.add_argument_group("numerics")
    n.add_argument("--nx", type=int, default=160, help="cells in x (nozzle)")
    n.add_argument("--ny", type=int, default=64, help="cells in y")
    n.add_argument("--npl", type=int, default=64,
                   help="extra axial cells in the external plume block")
    n.add_argument("--order", type=int, default=1, choices=[1, 2],
                   help="reconstruction order")
    n.add_argument("--cfl", type=float, default=0.4)
    n.add_argument("--flux", default="hybrid", choices=["hybrid", "hllc", "hll"],
                   help="hybrid = HLL at strong shocks, HLLC elsewhere (anti-carbuncle)")
    n.add_argument("--hll-thr", type=float, default=0.5,
                   help="hybrid: neighbour pressure ratio above 1+thr marks a shock cell")
    n.add_argument("--hll-dilate", type=int, default=1,
                   help="hybrid: grow the HLL region by this many cells")
    n.add_argument("--max-iter", type=int, default=12000)
    n.add_argument("--tol", type=float, default=1e-6)
    n.add_argument("--irs-eps", type=float, default=0.6,
                   help="implicit residual smoothing coefficient (0 = off)")
    n.add_argument("--irs-passes", type=int, default=2)
    n.add_argument("--log-every", type=int, default=1000)
    n.add_argument("--time-limit", type=float, default=None,
                   help="stop after N seconds and checkpoint")

    g = ap.add_argument_group("gas")
    g.add_argument("--gas", default="air", choices=sorted(Gas.LIB))
    g.add_argument("--gas-model", default="tpg", choices=sorted(set(_MODEL_ALIAS)),
                   help="cpg = constant gamma; tpg = gamma(T); "
                        "idg = gamma(T,rho) and R(T,rho)")
    g.add_argument("--gamma", type=float, default=None,
                   help="force a constant gamma (implies --gas-model cpg)")
    g.add_argument("--R-gas", type=float, default=None,
                   help="override the specific gas constant [J/kg/K]")
    g.add_argument("--theta-d", type=float, default=None,
                   help="dissociation temperature [K] for --gas-model idg")
    g.add_argument("--rho-d", type=float, default=None,
                   help="dissociation density [kg/m^3] for --gas-model idg")

    o = ap.add_argument_group("operating conditions")
    o.add_argument("--p0", type=float, default=1.0e5,
                   help="stagnation (chamber) pressure [Pa]")
    o.add_argument("--T0", type=float, default=300.0,
                   help="stagnation (chamber) temperature [K]")
    o.add_argument("--outlet", default="supersonic",
                   choices=["supersonic", "pressure"],
                   help="'supersonic' extrapolates (no back pressure is felt, "
                        "so no shock can form); 'pressure' imposes --pb")
    o.add_argument("--pb", type=float, default=None,
                   help="back pressure [Pa] for --outlet pressure "
                        "(default = --p-amb)")
    o.add_argument("--p-amb", type=float, default=1.01325e5,
                   help="ambient pressure [Pa] (plume and thrust reference)")
    o.add_argument("--T-amb", type=float, default=300.0,
                   help="ambient temperature [K]")

    gd = DEFAULT_GEOM
    geo = ap.add_argument_group("nozzle geometry")
    geo.add_argument("--d-throat", type=float, default=gd["D_throat"],
                     help="throat diameter (full height in planar mode)")
    geo.add_argument("--d-exit", type=float, default=gd["D_exit"],
                     help="exit diameter")
    geo.add_argument("--d-inlet", type=float, default=gd["D_inlet"],
                     help="inlet diameter")
    geo.add_argument("--l-div", type=float, default=gd["L_div"],
                     help="diverging-section length")
    geo.add_argument("--l-conv", type=float, default=gd["L_conv"],
                     help="converging-section length")
    geo.add_argument("--profile", type=str, default=gd["profile"],
                     choices=["ellipse", "ellipse-legacy", "conical",
                              "cosine", "bell", "super-ellipse"],
                     help="diverging wall profile")
    geo.add_argument("--theta-n", type=float, default=22.0,
                     help="bell profile: wall angle just after the throat [deg]")
    geo.add_argument("--theta-e", type=float, default=8.0,
                     help="bell profile: wall angle at the exit [deg]")
    geo.add_argument("--se-n", type=float, default=3.0,
                     help="super-ellipse RADIAL exponent n "
                          "(3 = the MATLAB profile)")
    geo.add_argument("--se-p", type=float, default=2.0,
                     help="super-ellipse AXIAL exponent p (2 = the MATLAB profile)")
    geo.add_argument("--se-trunc", type=float, default=1.0,
                     help="super-ellipse: fraction of L_div the curve is "
                          "evaluated over.  1.0 is the exact MATLAB equation "
                          "and is vertical at the lip; 0.998 matches the "
                          "MATLAB truncation but keeps both end radii exact "
                          "and leaves a finite exit angle")
    geo.add_argument("--cluster", type=float, default=gd["cluster"],
                     help="axial stretching strength (larger = finer at throat)")
    geo.add_argument("--cluster-mode", default="throat",
                     choices=["throat", "ends"],
                     help="'throat' refines the throat (correct); 'ends' "
                          "reproduces the original distribution")
    geo.add_argument("--d-farfield", type=float, default=None,
                     help="plume outer diameter (default grows with the plume "
                          "length: max(2 x D_exit, D_exit + 0.24 x L_plume))")
    geo.add_argument("--plume-factor", type=float, default=5.0,
                     help="plume length = factor x L_div (the DIVERGING length)")
    geo.add_argument("--axisym", action="store_true",
                     help="axisymmetric nozzle (default is planar 2D)")
    pl = geo.add_mutually_exclusive_group()
    pl.add_argument("--plume", dest="plume", action="store_true",
                    help="add the external jet block (needed for shock diamonds)")
    pl.add_argument("--no-plume", dest="plume", action="store_false")
    geo.add_argument("--plume-ramp", type=float, default=None,
                     help="OPEN-PLUME: length after the lip over which the plume "
                          "outer boundary opens to its full radius "
                          "(default 0.5 x D_exit)")
    geo.add_argument("--legacy-plume", action="store_true",
                     help="restore the original plume (ramp over 0.5 x L_plume "
                          "and Riemann-invariant lateral far field)")
    ap.set_defaults(plume=False)

    s = ap.add_argument_group("shock detection")
    s.add_argument("--shock-sp-min", type=float, default=0.05,
                   help="minimum |grad p| h / p for a cell to be a shock cell")
    s.add_argument("--shock-mn-min", type=float, default=1.02,
                   help="minimum shock-normal Mach number")
    s.add_argument("--shock-probe", type=int, default=3,
                   help="cells up/downstream used to sample the shock jump")

    io = ap.add_argument_group("output / restart")
    io.add_argument("--out", type=str, default="cd_nozzle_validation.png",
                    help="figure path (.png)")
    io.add_argument("--json", type=str, default=None,
                    help="JSON path (default: same stem as --out, .json)")
    io.add_argument("--npz", type=str, default=None,
                    help="checkpoint path (default: same stem as --out, .npz)")
    io.add_argument("--restart", action="store_true",
                    help="resume from the checkpoint (interpolates if the grid "
                         "changed)")
    io.add_argument("--no-plot", action="store_true", help="skip the figure")
    io.add_argument("--no-mirror", action="store_true",
                    help="axisymmetric runs: plot only the computed upper half "
                         "instead of mirroring it about the axis")
    io.add_argument("--no-json", action="store_true")
    io.add_argument("--planar-json", action="store_true",
                    help="write the planar JSON as solved instead of converting "
                         "it to an axisymmetric configuration (the default "
                         "for planar runs; --axisym runs are written as solved)")
    io.add_argument("--json-cells", default="full",
                    choices=["full", "grid", "none"],
                    help="full = grid + every field per element; grid = element "
                         "list only; none = summary output only")
    io.add_argument("--json-sigfig", type=int, default=6)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.list_presets:
        for k, v in PRESETS.items():
            print(f"  {k:20s} {v}")
        return
    if args.preset:
        for k, v in PRESETS[args.preset].items():
            setattr(args, k.replace("-", "_"), v)

    geom = {"D_throat": args.d_throat, "D_exit": args.d_exit,
            "D_inlet": args.d_inlet, "L_div": args.l_div,
            "L_conv": args.l_conv, "profile": args.profile,
            "cluster": args.cluster}
    if not (geom["D_throat"] < geom["D_exit"]):
        ap.error("D_exit must be larger than D_throat (diverging section).")
    if not (geom["D_throat"] < geom["D_inlet"]):
        ap.error("D_inlet must be larger than D_throat (converging section).")
    if min(geom["D_throat"], geom["L_div"], geom["L_conv"]) <= 0:
        ap.error("diameters and lengths must be positive.")
    if args.p0 <= 0 or args.T0 <= 0:
        ap.error("p0 and T0 must be positive.")
    if (not args.axisym and not args.no_json and not args.planar_json
            and args.ny % 2):
        ap.error("the planar -> axisymmetric JSON conversion needs an even --ny "
                 "(y = 0 must be a node line); or pass --planar-json.")

    stem = os.path.splitext(args.out)[0]
    json_path = args.json or (stem + ".json")
    npz_path = args.npz or (stem + ".npz")

    gas = Gas(args.gas, _MODEL_ALIAS[args.gas_model], args.gamma,
              R_override=args.R_gas, T0_ref=args.T0,
              theta_d=args.theta_d, rho_d=args.rho_d)

    plume_cells = args.npl if args.plume else 0
    Lp = args.plume_factor * geom["L_div"] if args.plume else 0.0
    # A long plume needs radial room as well: a supersonic jet spreads, and a
    # far field placed too close clips the shock-cell train even though the
    # characteristic BC itself does not reflect.
    h_far = (0.5 * args.d_farfield if args.d_farfield else
             max(geom["D_exit"], 0.5 * geom["D_exit"] + 0.12 * Lp))
    mesh = Mesh(nx_noz=args.nx, ny=args.ny, nx_plume=plume_cells,
                Lc=geom["L_conv"], Ld=geom["L_div"], Lp=Lp,
                h_in=0.5 * geom["D_inlet"], h_t=0.5 * geom["D_throat"],
                h_e=0.5 * geom["D_exit"], Hmax=h_far,
                profile=geom["profile"], cluster=geom["cluster"],
                axisym=args.axisym, theta_n=args.theta_n,
                theta_e=args.theta_e, cluster_mode=args.cluster_mode,
                se_n=args.se_n, se_p=args.se_p, se_trunc=args.se_trunc,
                plume_ramp=(None if args.legacy_plume else
                            (args.plume_ramp if args.plume_ramp is not None
                             else 0.5 * geom["D_exit"])))
    iso = Isentrope(gas, args.T0, args.p0)

    outlet = "pressure" if args.plume else args.outlet
    pb = (args.p_amb if args.plume else
          (args.pb if args.pb is not None else args.p_amb))
    cfg = dict(p0=args.p0, T0=args.T0, p_amb=args.p_amb, T_amb=args.T_amb,
               outlet=outlet, pb=pb,
               # With --outlet supersonic the exit plane feels no back
               # pressure, so the physically consistent ambient reference for
               # the pressure-area term is vacuum.
               p_thrust_ref=(args.p_amb if args.plume else
                             (0.0 if outlet == "supersonic" else pb)),
               order=args.order, cfl=args.cfl, tol=args.tol,
               max_iter=args.max_iter, irs_eps=args.irs_eps,
               irs_passes=args.irs_passes, log_every=args.log_every,
               iso=iso, inlet_axial=True, legacy_plume=args.legacy_plume,
               flux=args.flux, hll_thr=args.hll_thr, hll_dilate=args.hll_dilate,
               shock_sp_min=args.shock_sp_min, shock_mn_min=args.shock_mn_min,
               shock_probe=args.shock_probe)

    print("=" * 78)
    print(" 2D compressible Euler solver -- converging-diverging nozzle")
    print(" HLLC finite volume, body-fitted mesh, SSP-RK2 to steady state")
    print("=" * 78)
    er = geom["D_exit"] / geom["D_throat"]
    print(f" Geometry : D_throat={geom['D_throat']:.4g}  D_exit={geom['D_exit']:.4g}"
          f"  (area ratio {mesh.Ae_At:.3f}, diameter ratio {er:.3f})")
    print(f"            D_inlet={geom['D_inlet']:.4g}  L_conv={geom['L_conv']:.4g}"
          f"  L_div={geom['L_div']:.4g}  profile={geom['profile']}")
    print(f" Gas      : {gas.name}  model={gas.model}  R0={gas.R0:.1f} J/kg/K  "
          f"gamma(T0)={gas.gamma_ref:.4f}")
    print(f" {mesh.summary()}")
    if mesh.nx_plume > 0:
        print(f" Plume    : L_plume={mesh.Lp:.4g} = {mesh.Lp/mesh.Ld:.3g} x L_div"
              f"  |  {mesh.nx_plume} cells, dx {mesh.plume_dx_first:.4g} -> "
              f"{mesh.plume_dx_last:.4g} (growth {mesh.plume_growth:.4f} per cell)")
        if mesh.plume_growth > 1.10:
            print("   WARNING: the plume cells grow more than 10% per cell, so "
                  "the far\n            downstream shock cells are smeared "
                  "rather than resolved.\n            Raise the plume cell "
                  "count to fix it.")

    print(f" Operating: p0={cfg['p0']/1e5:.4g} bar  T0={cfg['T0']:.0f} K  "
          f"outlet={cfg['outlet']}  pb={cfg['pb']/1e5:.4g} bar  "
          f"p_amb={cfg['p_amb']/1e5:.4g} bar")
    print(f"            p0/pb={cfg['p0']/cfg['pb']:.4g}   "
          f"p0/p_amb={cfg['p0']/cfg['p_amb']:.4g}   "
          f"thrust ambient reference {cfg['p_thrust_ref']/1e5:.4g} bar")
    print(f" Order    : {args.order}   CFL {args.cfl}   tol {args.tol:.0e}")
    if cfg["outlet"] == "supersonic":
        print(" NOTE: --outlet supersonic extrapolates at the exit, so the back")
        print("       pressure is NOT felt and NO shock can form in the nozzle.")
        print("       Use --outlet pressure --pb <Pa>, or --plume, for shocks.")
    print("=" * 78)

    th = theory_1d(mesh, gas, cfg, iso)
    reg = th["regimes"]
    print(f" 1D regime  : {reg['regime']}")
    print(f" 1D critical: NPR={reg['NPR']:.4g} | design NPR={reg['NPR_design']:.4g}"
          f" | 1st crit={reg['NPR_first_critical']:.4g}"
          f" | 2nd crit={reg['NPR_second_critical']:.4g}")
    if mesh.nx_plume > 0:
        _r = reg["p_exit_design_Pa"] / max(cfg["p_amb"], 1e-30)
        print(f" Jet      : p_exit(design)/p_amb = {_r:.3f}"
              f"  ({'under' if _r > 1.0 else 'over'}expanded)")
        if _r < 0.45 or _r > 4.0:
            print("   WARNING: the jet is far off design.  An INVISCID solution at "
                  "this pressure\n            ratio shocks down through a single "
                  "strong Mach disc and keeps no\n            supersonic core, so "
                  "no shock-cell (diamond) train develops.  A real\n            "
                  "nozzle would separate internally instead, which is viscous and"
                  "\n            outside what this solver models.  For a developed "
                  "cell train aim\n            for p_exit/p_amb roughly 0.6 - 2.5.")
    lo, hi = reg["shock_in_nozzle_window_pb_p0"]
    if lo is not None and reg.get("back_pressure_is_felt", True):
        print(f" 1D: a normal shock sits inside the nozzle for "
              f"{lo:.4f} < pb/p0 < {hi:.4f}  (here {cfg['pb']/cfg['p0']:.4f})")
    elif lo is not None:
        print(f" 1D: with a pressure outlet a normal shock would sit inside the "
              f"nozzle for {lo:.4f} < pb/p0 < {hi:.4f}"
              f"\n     (not applicable here -- the outlet BC ignores pb)")
    if th["shock"].get("x_shock") is not None:
        print(f" 1D theory: normal shock at x={th['shock']['x_shock']:.4f}, "
              f"M1={th['shock']['M1']:.4f} -> M2={th['shock']['M2']:.4f}, "
              f"p02/p01={th['shock']['p02_p01']:.4f}")

    it, res0, hist, conv = 0, -1.0, [], False
    if args.restart and os.path.exists(npz_path):
        d = np.load(npz_path)
        U_old = d["U"]
        if U_old.shape[1:] == (mesh.ny, mesh.nx):
            U = U_old
            it, res0 = int(d["it"]), float(d["res0"])
            hist = [tuple(r) for r in d["hist"]]
            conv = bool(d["conv"])
            print(f" restart: resumed at iteration {it}")
        else:
            U = interpolate_state(U_old, U_old.shape[1:], mesh, gas)
            print(f" restart: grid sequencing {U_old.shape[1:]} -> "
                  f"{(mesh.ny, mesh.nx)} (warm start)")
    else:
        U = initial_state(mesh, gas, cfg, th)
        print(" initial condition: quasi-1D theory" +
              (" + ambient plume" if args.plume else ""))

    if not conv:
        print("\nMarching to steady state ...")
        U, it, res0, hist, conv = march(U, mesh, gas, cfg, it, res0, hist,
                                        args.time_limit)
        np.savez(npz_path, U=U, it=it, res0=res0, hist=np.array(hist), conv=conv)
        print(f"  checkpoint -> {npz_path}")

    # ------------------------------------------------------------- analysis
    (F, sen, shocks, structs, waves, expansions, slips, interactions,
     chars) = analyse_state(mesh, gas, cfg, iso, U, args)

    print(f"\nWave analysis (Tier 1):")
    print(f"  expansion fans : {len(expansions)} structure(s), "
          f"{waves['n_expansion_cells']} cells")
    for e in expansions[:3]:
        print(f"    - at x={e['centroid_x']}, y={e['centroid_y']}: "
              f"M {e['M_upstream']} -> {e['M_downstream']}, "
              f"d(nu)={e['prandtl_meyer_change_deg']} deg vs turning "
              f"{e['flow_turning_deg']} deg  (simple-wave residual "
              f"{e['simple_wave_residual_deg']} deg, ds="
              f"{e['entropy_rise_J_per_kgK']} J/kg/K)")
    print(f"  slip lines     : {len(slips)} structure(s), "
          f"{waves['n_slip_cells']} cells")
    for sl in slips[:2]:
        print(f"    - at x={sl['centroid_x']}, y={sl['centroid_y']}, "
              f"angle {sl['angle_deg']} deg, length {sl['length']}")
    _bad = [st for st in structs if st.get("wave_resolution_warning")]
    _thin = [st for st in structs
             if st.get("beta_from_fitted_line_deg") is None]
    if _bad or _thin:
        print(f"  RESOLUTION     : {len(_bad)} structure(s) with inconsistent "
              f"wave angles, {len(_thin)} too fragmented to fit a line")
        print( "                   -> the captured waves are too coarse for "
               "reliable beta/theta;")
        print( "                      refine, or use --order 2, before trusting "
               "the wave geometry.")
    print(f"  interactions   : {len(interactions)}")
    for iv in interactions[:4]:
        print(f"    - {iv['type']} at x={iv['x']}, y={iv['y']}")
        if iv["reflection_analysis"]:
            ra = iv["reflection_analysis"]
            print(f"        theta_inc={ra['incident_theta_deg']} deg, "
                  f"M behind={ra['M_behind_incident']}, "
                  f"theta_max={ra['theta_max_behind_incident_deg']} deg "
                  f"-> {ra['predicted_type']}")
    for st in structs[:3]:
        w = st.get("theta_beta_M_weak")
        if w and w.get("attached"):
            sgm = st.get("theta_beta_M_strong") or {}
            print(f"  theta-beta-M   : {st['type'][:36]:36s}")
            print(f"        theta measured from the velocity turning = "
                  f"{st.get('deflection_from_velocity_deg')} deg")
            print(f"        measured beta={st['wave_angle_beta_deg']} deg | "
                  f"theory weak={w['beta_deg']} strong={sgm.get('beta_deg')} | "
                  f"best match: {st.get('branch_matched')} "
                  f"(theta_max {w['theta_max_deg']} deg)")
    for ch in chars:
        if ch["coalescence"]:
            c = ch["coalescence"]
            print(f"  {ch['family']} characteristics converge at x={c['x']}, "
                  f"y={c['y']} (gap {c['gap_at_start']} -> {c['gap_at_focus']})"
                  f"  <- incipient internal shock")

    thr = thrust(mesh, gas, cfg, F, iso, th)
    Mc, err, val = validate_centreline(mesh, gas, cfg, F, th)

    lip = mesh.lip
    print("\nResults:")
    print(f"  exit centreline: M={Mc[lip]:.4f}  "
          f"T={centreline(mesh, F['T'])[lip]:.1f} K  "
          f"p={centreline(mesh, F['p'])[lip]/1e5:.5g} bar  "
          f"|V|={centreline(mesh, F['V'])[lip]:.1f} m/s")
    print(f"  gamma(T)       : {np.nanmin(F['gamma']):.4f} .. "
          f"{np.nanmax(F['gamma']):.4f}    R(T,rho): "
          f"{np.nanmin(F['R']):.1f} .. {np.nanmax(F['R']):.1f} J/kg/K")
    print(f"  T range        : {np.nanmin(F['T']):.1f} .. {np.nanmax(F['T']):.1f} K"
          f"    p range: {np.nanmin(F['p'])/1e5:.4g} .. "
          f"{np.nanmax(F['p'])/1e5:.4g} bar")
    print(f"  rho range      : {np.nanmin(F['rho']):.4g} .. "
          f"{np.nanmax(F['rho']):.4g} kg/m^3    peak |V|: "
          f"{np.nanmax(F['V']):.1f} m/s")
    print(f"  mass balance   : {thr['mass_balance_error_percent']}%   "
          f"Cd={thr['discharge_coefficient_Cd']}")
    print(f"  thrust         : {thr['thrust_exit_plane_N']:.5g} N (exit plane), "
          f"{thr['thrust_wall_integral_route_N']:.5g} N (wall+inlet route), "
          f"disagreement {thr['thrust_route_disagreement_percent']}%")
    print(f"  Isp={thr['specific_impulse_s']} s   "
          f"c*={thr['characteristic_velocity_cstar_m_per_s']} m/s   "
          f"CF={thr['thrust_coefficient_CF']}")

    print("\nCentreline validation vs quasi-1D theory "
          f"(gamma model '{gas.model}'):")
    print(f"  throat Mach (solver) = {val['throat_mach_solver']}   "
          f"(ideal = 1.0; theory at the throat CELL "
          f"[A/At={val['throat_cell_area_ratio_A_over_At']}] = "
          f"{val['throat_mach_theory_at_cell']})")
    print(f"  exit   Mach (solver) = {val['exit_mach_solver']}   "
          f"(theory = {val['exit_mach_theory']})")
    print(f"  max |dM| = {val['max_abs_delta_M']}   "
          f"mean |dM| = {val['mean_abs_delta_M']}   (whole nozzle)")
    print(f"  max |dM| = {val['max_abs_delta_M_outside_transonic']}   "
          f"mean |dM| = {val['mean_abs_delta_M_outside_transonic']}   "
          f"(excluding the transonic band |M-1| < 0.2, where quasi-1D is "
          f"not valid)")

    print(f"\nShocks: {int(sen['flag'].sum())} flagged cells in "
          f"{len(structs)} structure(s)")
    for st in structs[:5]:
        print(f"   - {st['type']}")
        print(f"     x~{st['centroid_x']}, y~{st['centroid_y']}, "
              f"M1={st['M1_upstream']}, Mn1={st['Mn1_normal']}, "
              f"beta={st['wave_angle_beta_deg']} deg, "
              f"theta={st['deflection_theta_deg']} deg")
        print(f"     p2/p1={st['p2_over_p1_solver']} "
              f"(RH {st['p2_over_p1_rankine_hugoniot']}), "
              f"T2/T1={st['T2_over_T1_solver']}, "
              f"ds={st['entropy_rise_J_per_kgK']} J/kg/K, "
              f"p02/p01={st['p02_over_p01_rankine_hugoniot']}")
    if shocks["plume"]:
        pm = shocks["plume"]
        print(f"   plume: {pm['jet_state']}, Mj={pm['fully_expanded_jet_mach_Mj']},"
              f" cell spacing measured={pm['measured_shock_cell_spacing_m']} "
              f"vs Prandtl={pm['prandtl_shock_cell_spacing_m']}")

    if not args.no_plot:
        plot_all(mesh, gas, cfg, F, th, sen, hist, args.out, Mc, err, val,
                 mirror=not args.no_mirror)
    if not args.no_json and not mesh.axisym and not args.planar_json:
        # Planar data -> axisymmetric configuration.  Post-processing only:
        # the planar march above is untouched, the converged field is stitched
        # onto the axisymmetric half-plane and everything written to the JSON
        # (grid, fields, integrals, theory, shocks, thrust) is recomputed on it.
        print("\nConverting the planar solution to an axisymmetric "
              "configuration for the JSON ...")
        mesh_ax, U_ax, cinfo = planar_to_axisym(mesh, U)
        th_ax = theory_1d(mesh_ax, gas, cfg, iso)
        (F_ax, sen_ax, shocks_ax, _, waves_ax, _, _, _,
         _) = analyse_state(mesh_ax, gas, cfg, iso, U_ax, args)
        thr_ax = thrust(mesh_ax, gas, cfg, F_ax, iso, th_ax)
        _, _, val_ax = validate_centreline(mesh_ax, gas, cfg, F_ax, th_ax)
        print(f"  {mesh_ax.summary()}")
        print(f"  max planar asymmetry (rel): "
              f"{max(cinfo['planar_asymmetry_max_rel'].values()):.3g}")
        print(f"  axisym: mass flow {thr_ax['mass_flow_exit_kg_per_s']} kg/s, "
              f"thrust {thr_ax['thrust_exit_plane_N']:.5g} N, exit axis "
              f"M={F_ax['M'][0, mesh_ax.lip]:.4f} "
              f"(axisym 1D theory {val_ax['exit_mach_theory']})")
        export_json(mesh_ax, gas, cfg, F_ax, sen_ax, shocks_ax, th_ax, iso,
                    thr_ax, hist, conv, it, json_path,
                    cells_mode=args.json_cells, nsig=args.json_sigfig,
                    validation=val_ax, geom=geom, waves=waves_ax,
                    conversion=cinfo)
    elif not args.no_json:
        export_json(mesh, gas, cfg, F, sen, shocks, th, iso, thr, hist, conv, it,
                    json_path, cells_mode=args.json_cells, nsig=args.json_sigfig,
                    validation=val, geom=geom, waves=waves)
    if not conv:
        print("\n  NOTE: not converged to tolerance.  Re-run with --restart to "
              "continue, or see the limitations note in the file header.")


if __name__ == "__main__":
    main()


# =============================================================================
# CHANGELOG / DEFECT LIST  (what was wrong in cd_nozzle_solver.py)
# =============================================================================
# SHOCK WAVES
#  S1  A shock could not exist.  The outlet was hard-wired to supersonic
#      extrapolation (Wg[:,1:-1,-1] = Wint[:,:,-1], and residual() further set
#      BOTH sides of the outlet face to the interior state), so no back
#      pressure was ever felt.  There was also no external domain.  The only
#      reachable steady state was the perfectly expanded one; every other
#      nozzle pressure ratio was unrepresentable.  Added a characteristic
#      pressure outlet (--outlet pressure --pb) and an external plume block
#      (--plume) with a Riemann-invariant far field.
#  S2  No shock detection, and none of the parameters that govern shock
#      initiation: design / 1st / 2nd / 3rd critical pressure ratios, the
#      pb/p0 window for an in-nozzle shock, or the operating regime.  Added.
#  S3  No propagation parameters: wave angle beta, deflection theta, normal
#      Mach, p2/p1, rho2/rho1, T2/T1, entropy rise, total-pressure loss, Mach
#      angle, Prandtl-Meyer angle, shock-cell spacing, Mach-disc position.
#      All added, with a real-gas Rankine-Hugoniot solve for cross-checking.
#  S4  HLLC Roe average bug: a_roe = sqrt((g-1)(H_roe - 0.5 un_roe^2)) omits
#      the TANGENTIAL kinetic energy, which is present in H.  a_roe was
#      therefore over-estimated, SL/SR too wide, and every shock more smeared
#      than the scheme should give.  Fixed by including ut_roe.
#
# THERMODYNAMICS
#  T1  gamma was the module constant 1.4 and R_GAS the module constant 287,
#      with no temperature dependence at all -- unusable for hot gas, where
#      gamma falls towards 1.3 as the vibrational modes activate.  Added the
#      multi-mode vibrational model gamma(T) with correct monatomic / linear /
#      non-linear baselines, for ten gases.
#  T2  R could not vary.  Added the Lighthill / Freeman ideal dissociating gas,
#      alpha^2/(1-alpha) = (rho_d/rho) exp(-theta_d/T), giving
#      R_eff(T,rho) = R0 (1+alpha) -- the physical mechanism by which the gas
#      constant changes with temperature (the mean molar mass drops).
#  T3  The 1D validation curve and the inlet BC used closed-form constant-gamma
#      isentropic relations.  Replaced by a tabulated isentrope built by
#      integrating d ln p / d ln T = cp(T)/R, so the reference curve uses the
#      same gas as the solver and the comparison is meaningful.
#  T4  cons_to_prim divided by U[0] with no floor, so a single negative-density
#      cell produced inf/NaN rather than a recoverable state.
#
# MESH / GEOMETRY
#  M1  _x_nodes used s2 = (1 + tanh(c(2s-1))/tanh c)/2 and was documented as
#      "clustering toward the throat".  It actually refines the INLET and the
#      OUTLET: at Nx=160, Lc=1, Ld=2 the throat cell is 5.9x the inlet cell and
#      the coarsest cells sit at x ~ 0.48.  The throat is exactly where the
#      flow chokes and where the wall curvature is largest, so this is the
#      worst possible distribution; the entropy it generates can stop the
#      nozzle choking at back-pressure ratios where 1D theory says it must.
#      Replaced by two one-sided tanh stretchings that both refine towards
#      x = 0 (--cluster-mode throat); --cluster-mode ends restores the old one.
#  M2  The DEFAULT profile "ellipse", h = h_e - dh*sqrt(1-xr^2), is smooth at
#      the throat as documented but VERTICAL at the exit lip: the wall angle
#      passes 60 deg at xr = 0.99 and 90 deg at xr = 1, firing a spurious
#      expansion fan into the exit plane and corrupting the exit state that
#      thrust and Mach validation are read from.  "ellipse" is now the
#      parallel-at-exit quarter ellipse; the original is "ellipse-legacy".
#      A proper Bezier bell profile was also added.
#  M3  Cell centres were the arithmetic mean of the four corners; now the exact
#      polygon centroid, which matters on the stretched cells.
#  M4  Only planar geometry existed.  Real nozzles are axisymmetric and thrust
#      differs by the area weighting, so --axisym was added, using the strictly
#      conservative y-weighted formulation (no 1/y source, exact on the axis).
#
# BOUNDARY CONDITIONS
#  B1  The stagnation inlet copied the interior Mach number and imposed the
#      full isentropic state from it.  That over-determines a subsonic inflow
#      (two conditions are admissible, three were applied) and leaves a
#      permanent residual hard spot in the inlet corner cells.  Replaced by the
#      characteristic form: the interior supplies R- = u_n - 2a/(gamma-1) and
#      the boundary state is the point on the variable-gamma (p0,T0) isentrope
#      with that same R-, capped at M = 1.
#  B2  The inlet used the velocity MAGNITUDE sqrt(u^2+v^2) rather than the
#      face-normal component, so any transverse velocity inflated the inferred
#      inlet Mach number.
#  B3  The "mach" inlet mode specified rho, u and p simultaneously for a
#      subsonic inflow -- fully reflective.  Removed in favour of the
#      characteristic stagnation inlet; supersonic inflow is handled by the
#      supersonic branch.
#  B4  The outlet had no subsonic branch and no reverse-flow branch, so the
#      solver was undefined whenever the exit was not supersonic.  Both added.
#  B5  local_dt used cfl*sqrt(vol)/(|V|+a).  On stretched, high-aspect-ratio
#      cells -- exactly what the throat clustering produces -- sqrt(vol) is not
#      the limiting length and the step is unsafe.  Replaced by the face-based
#      spectral radius dt = CFL V / sum_faces (|u.n| + a) S.
#
# THRUST AND INTEGRALS
#  P1  There was NO thrust calculation, and no mass flow.  Added the exit-plane
#      momentum integral, an independent wall-pressure + inlet-momentum
#      control-volume route, the ideal quasi-1D value, Isp, c*, C_F, discharge
#      coefficient and vacuum thrust.  The difference between the two computed
#      routes is reported as a convergence measure.
#
# VALIDATION / OUTPUT
#  V1  The centreline was sampled as Mf[ny//2].  With the full channel
#      y in [-h, +h] and ny even, y = 0 falls on a cell FACE, so that row is
#      the first cell ABOVE the centreline -- a systematic off-axis bias.  Now
#      the two rows straddling y = 0 are averaged.
#  V2  The 1D reference always took the subsonic branch for x<0 and the
#      supersonic branch for x>0, i.e. it assumed the nozzle was choked and
#      shock free no matter what was actually being solved.  The reference now
#      follows the computed regime, including a normal shock in the divergent.
#  V3  No JSON output and no grid numbering.  Every cell now carries
#      cell_id = j*nx + i, the element list is written first, and every field
#      array follows exactly that order.
#  V4  Only the Mach number was ever reported.  Temperature, pressure, density,
#      velocity magnitude AND direction, plus ~25 further quantities, are now
#      exported per element.
# =============================================================================
