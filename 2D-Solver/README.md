# 2-D Euler solver for a converging-diverging nozzle

A single Python file (`cd_nozzle_solver_axiconv_openplume_hybrid.py`, about 3,900 lines, numpy + matplotlib) that solves the inviscid compressible flow through a converging-diverging nozzle and, if asked, the jet behind it. The wall-profile check in `../Profile-Check` is compared against its results.

Part of the MachViz team project at Otto von Guericke University Magdeburg (team of 6). A teammate and I wrote it with an AI coding assistant; I ran it and checked its results against 1-D theory. My focus was the HLLC/HLL flux scheme, the plume and shock handling, and the warm start (restart onto a finer grid).

## What it solves

The 2-D Euler equations in conservative form, for mass, x- and y-momentum and total energy:

∂U/∂t + ∂F/∂x + ∂G/∂y = 0,  U = (ρ, ρu, ρv, ρE)

There are two geometries:

- **Planar** (default): a full channel, y from −h(x) to +h(x). Area ratio A/A* = h/h_t.
- **Axisymmetric** (`--axisym`): r from 0 to h(x), the lower boundary is the axis. The equations are multiplied by r before integrating over a cell, so the only source term left is the hoop pressure p·A in the radial momentum equation and the axis needs no 1/r term. A/A* = (r/r_t)².

The solver marches in pseudo-time until the residual stops changing, so it only gives steady solutions.

## How one run works

```
inputs (geometry, gas, p0, T0, back pressure)
  -> Gas model            gamma(T), R(T, rho), e(T), h(T), T from e
  -> Isentrope table      1-D reference through (p0, T0) for that gas
  -> 1-D theory           regime, critical pressure ratios, shock position
  -> Mesh                 body-fitted structured grid, metrics, cell numbering
  -> Initial field        1-D solution in the nozzle, ambient air in the plume
  -> March to steady state
       boundary conditions -> MUSCL faces -> HLLC/HLL flux -> residual
       -> local time step -> SSP-RK2 update -> residual smoothing
  -> Post-processing      derived fields, shock/expansion/slip-line detection,
                          Rankine-Hugoniot and oblique-shock checks, thrust,
                          centreline vs 1-D theory
  -> Output               PNG figure, JSON (every cell numbered), NPZ checkpoint
```

The file follows that order. Section numbers below are the ones in the code.

## 1. Gas model (`class Gas`)

Three closures, chosen with `--gas-model`:

| Model | γ | R | Use |
|---|---|---|---|
| `cpg` | constant | constant | cold air, comparison with textbook tables |
| `tpg` (default) | γ(T) | constant | hot gas where vibration switches on |
| `idg` | γ(T, ρ) | R(T, ρ) | very hot H₂/N₂/O₂ where molecules split |

For `tpg` every vibrational mode i (temperature θᵢ, weight gᵢ) adds an Einstein term to the heat capacity:

c_v/R = n_base + Σ gᵢ · xᵢ² eˣⁱ / (eˣⁱ − 1)²,  xᵢ = θᵢ/T

n_base is 3/2, 5/2 or 3 for monatomic, linear and non-linear molecules. Ten gases are built in (H₂, N₂, O₂, air, CO, CO₂, H₂O, CH₄, He, Ar).

For `idg` (Lighthill–Freeman ideal dissociating gas) the dissociated fraction α comes from α²/(1−α) = (ρ_d/ρ)·exp(−θ_d/T). Then R_eff = R₀(1 + α), e = R₀(3T + α θ_d) and the frozen γ = (4 + α)/3.

Temperature is never a state variable. The solver carries U and recovers T from the internal energy by Newton iteration on e(T) − e = 0, using the equilibrium c_v (with dα/dT) as the derivative, because the frozen c_v makes the Newton step overshoot when the gas dissociates.

## 2. 1-D reference (`class Isentrope`, `theory_1d`, `npr_regimes`)

With variable γ the closed-form isentropic formulas don't apply, so the code builds a table instead. Along an isentrope ds = 0 gives

d ln p / d ln T = c_p(T) / R

which is integrated (trapezoid rule) from T0 down to 0.2 % of T0 on 4,000 log-spaced points. From the energy equation V = √(2(h0 − h)), then M = V/a. The area ratio comes from mass conservation, A/A* = (ρV)* / (ρV), and the Prandtl–Meyer angle from ν = ∫ √(M²−1) dV/V. The table also stores the outgoing Riemann invariant V − 2a/(γ−1), which the inlet boundary needs.

From this table `npr_regimes` finds the three critical back pressures:

1. **1st critical:** throat just sonic, exit subsonic (subsonic branch at A_e/A_t).
2. **2nd critical:** normal shock standing exactly at the exit plane.
3. **3rd critical (design):** shock-free supersonic exit (supersonic branch).

For a back pressure between the 1st and 2nd critical, `find_shock_area` bisects on the shock area A_s: supersonic state at A_s → real-gas normal shock → new A* behind it (A*₂ = A_s / (A/A*)(M₂)) → subsonic exit Mach on the post-shock isentrope → exit pressure. The A_s whose exit pressure equals p_b is where the 1-D shock sits.

The normal shock (`normal_shock_real`) is solved for any equation of state. With v = ρ1/ρ2 the momentum and energy jumps give p2 = p1 + ρ1u1²(1 − v) and h2 = h1 + u1²(1 − v²)/2; v is found by bisection so that p2 = ρ2 R(T2) T2 with T2 from h2.

## 3. Mesh (`class Mesh`)

A structured, body-fitted grid: nx axial × ny transverse cells, every column scaled to the local wall height h(x).

- **Convergent wall:** cosine blend from the inlet radius to the throat.
- **Divergent wall** (`--profile`): `conical`, `cosine`, `bell` (quadratic Bézier from θ_n at the throat to θ_e at the lip), `ellipse` (part of an ellipse arc with a finite exit angle), `ellipse-legacy` and `super-ellipse` (the contour from an earlier MATLAB script).
- **Axial spacing:** two one-sided tanh stretchings that both put the finest cells at the throat (`--cluster-mode throat`).
- **Plume block** (`--plume`): extra columns behind the lip, 5 × L_div long by default, whose outer edge opens to the far-field radius over 0.5 × D_exit.
- **Metrics:** exact polygon centroids, face normals and lengths, cell areas. In axisymmetric mode face areas and volumes are weighted by r.
- **Numbering:** cell_id = j·nx + i (j transverse, i axial). That is numpy's C-order ravel, so every field in the JSON is in the same order as the element list.

## 4. Fluxes (`reconstruct`, `hllc`)

Cell-centred finite volume. At each face:

1. **Reconstruction.** First order (cell values) or, with `--order 2`, MUSCL with the minmod limiter. Face values are clamped between the two neighbouring cell values so the scheme can't create new extrema.
2. **Rotation.** The two states are rotated into the face normal/tangent frame.
3. **Riemann solver.** HLLC (Toro). Wave speeds S_L, S_R from Einfeldt-type bounds (the smaller/larger of the cell and Roe-average values u_n ∓ a); the Roe sound speed includes the tangential kinetic energy. The contact speed S* follows from the jump conditions, and the flux is picked from F_L, F*_L, F*_R or F_R by the signs of S_L, S*, S_R.
4. **Anti-carbuncle switch.** On a fine grid HLLC can grow the carbuncle instability at a strong normal shock lined up with the grid (a Mach disc on the axis). With `--flux hybrid` (default) any cell whose pressure ratio to a neighbour exceeds 1.5 is flagged, the flag is grown by one cell, and faces next to it use HLL, which has no contact wave and is carbuncle-free. Everywhere else stays HLLC.

The residual of a cell is minus the sum of face fluxes × face area, divided by the cell volume, plus the hoop-pressure term in axisymmetric mode.

## 5. Boundary conditions (`apply_bcs`)

Ghost cells around the domain, filled each stage:

- **Inlet:** subsonic stagnation inflow. One characteristic leaves the domain, carrying R⁻ = u_n − 2a/(γ−1) from the interior. The inlet Mach number is the point on the (p0, T0) isentrope with that same R⁻, capped at M = 1; T, p and ρ follow from the table. So p0, T0 and the flow direction are imposed and the Mach number comes from the interior, which keeps the boundary from reflecting waves back in.
- **Outlet:** `supersonic` copies the interior (no back pressure is felt, so no shock can form); `pressure` imposes p_b on subsonic outflow, extrapolates supersonic outflow and treats reverse flow as an ambient reservoir.
- **Walls and axis:** inviscid slip, the ghost velocity mirrored about the face normal.
- **Plume edge:** a pressure opening. Outflow gets p = p_amb with density and velocity from inside; inflow gets ambient p and T with no tangential velocity. A constant-pressure edge is the free-jet condition, and it stops the region past the lip from acting like more nozzle. `--legacy-plume` restores the older Riemann-invariant far field.

## 6. Time marching (`march`, `local_dt`, `smooth`)

- **SSP-RK2 (Heun):** U¹ = Uⁿ + Δt R(Uⁿ), then Uⁿ⁺¹ = ½(Uⁿ + U¹ + Δt R(U¹)).
- **Local time stepping:** each cell uses its own Δt = CFL · V / Σ_faces(|u·n| + a)·S, since only the steady state matters. CFL ramps from 10 % to the set value over the first 400 iterations.
- **Implicit residual smoothing:** two Jacobi passes of (1 + 4ε)R̄ − ε Σ R̄_nb = R (ε = 0.6). This lets a larger CFL run stably and leaves the steady solution unchanged, since R̄ = 0 only when R = 0.
- **Floors:** density and pressure are kept positive after each stage.
- **Convergence:** stop when the RMS density residual has dropped by `--tol` (default 10⁻⁶) or at `--max-iter`. The state is saved to `.npz`; `--restart` continues it, also onto a finer grid (grid sequencing).

## 7. Post-processing (sections 9–14)

None of this changes the solution.

- **Derived fields:** T, a, M, flow angle, total T and p, entropy, Mach angle, Prandtl–Meyer angle, Green–Gauss pressure gradient, divergence and vorticity.
- **Shock detection:** a cell is a shock cell when |∇p|·h/p > 0.05, ∇·V < 0 and the Mach number normal to the shock (V·∇p/|∇p|)/a > 1.02. Flagged cells are grouped into structures, and each structure is sampled three cells upstream and downstream along its normal. The code reports β, θ, M₁, M_n1, M₂, p₂/p₁, ρ₂/ρ₁, T₂/T₁, entropy rise and p₀₂/p₀₁, both from the solution and from the real-gas Rankine–Hugoniot solve, so each shock comes with its own check.
- **Wave toolkit:** a line fitted through each structure (gives the wave angle independent of the local gradient), weak/strong θ–β–M solutions and θ_max for the real gas, expansion fans (Prandtl–Meyer change vs flow turning), slip lines (entropy jump without a pressure jump), shock–shock crossings with a regular vs Mach reflection test, and Mach lines traced through the field to find where compression waves bunch into a shock.
- **Jet structure:** shock-cell spacing from the centreline pressure, compared with Prandtl's L = 1.306 D_e √(M_j² − 1) and the Ashkenas–Sherman Mach-disc distance x/D_e = 0.67 √(p0/p_amb).
- **Thrust, three ways:** exit-plane momentum + pressure integral; wall pressure + inlet momentum (a control-volume check); and ideal 1-D ṁV_e + (p_e − p_a)A_e. Also mass balance, discharge coefficient, Isp, c* and C_F. The gap between the first two is reported as a measure of how well the momentum equation has converged.
- **Centreline check:** solver Mach vs the 1-D table for the same gas, with the transonic band |M − 1| < 0.2 left out, where 1-D theory doesn't hold (the sonic line is curved).

## Planar runs and the JSON

By default the solve is planar, and the JSON is then written for an axisymmetric layout: the upper half of the planar grid is reused as r ≥ 0, each cell is averaged with its mirror cell, and mass flow, thrust and the shock analysis are recomputed on the axisymmetric geometry. **The field itself is not re-solved.** A planar channel and a round nozzle with the same diameters have different area ratios (h/h_t vs (r/r_t)²), so the Mach numbers in that JSON belong to the planar nozzle. The comparison in `../Profile-Check/crossing_check_1.png` was made from such a planar run. For a true axisymmetric solution use `--axisym`; `--planar-json` writes the planar result as solved.

## Running it

```
python cd_nozzle_solver_axiconv_openplume_hybrid.py --list-presets
python cd_nozzle_solver_axiconv_openplume_hybrid.py --preset validated-choked
python cd_nozzle_solver_axiconv_openplume_hybrid.py --preset shock-in-nozzle
python cd_nozzle_solver_axiconv_openplume_hybrid.py --axisym --plume --profile cosine \
       --d-throat 1.2 --d-exit 2.5 --l-div 2.5 --p0 6.46e5 --p-amb 1.01325e5
python cd_nozzle_solver_axiconv_openplume_hybrid.py --help
```

Defaults: air with `tpg`, p0 = 1 bar, T0 = 300 K, D_throat 1.2, D_exit 2.2, D_inlet 2.4, L_div 2.0, L_conv 1.0 (consistent units), ellipse profile, 160 × 64 cells, first order, CFL 0.4. Presets cover a choked nozzle, a shock inside the nozzle, over- and underexpanded jets, a hot H₂ plume and a dissociating H₂ case.

Each run writes a PNG (Mach, T, ρ, p, γ, R, |V|, flow angle and shock-sensor contours, centreline profiles, Mach vs 1-D theory, convergence history), a JSON with every cell numbered and more than 30 fields per cell, and an `.npz` checkpoint.

## Limits

- Inviscid: no boundary layer, no separation, no turbulence. Real nozzles separate at strong overexpansion; this solver can't show that.
- Single species. The dissociation model is a two-parameter fit, not finite-rate chemistry.
- Steady only. A shock near the exit at low pressure ratio is unsteady in reality and may not converge here.
- First order smears shocks over several cells; second order is sharper but can stall around a 10⁻³ residual because of limiter switching.
