# MachViz – nozzle flow solver (MATLAB)

Solver side of **MachViz**, a team project in the course *DE-Projekt: Digital Engineering of Process Engineering Applications*, Otto von Guericke University Magdeburg, 2026.

The goal of MachViz is an interactive Unity application that shows Mach number, pressure, temperature and density inside a convergent-divergent nozzle. The team has six members:

- **Chemical and Energy Engineering (CEE), solver development:** Jaideep Buksagarmath (subsonic flow), Theresa Kasthuri Dinah Samuel (supersonic flow), Shashank Suresh Srinivasan (flow with shocks)
- **Digital Engineering (DE), visualisation:** Hemanth Nagaraj, Rakshan Sujan Shetty, Mantasha Khanam

This repository holds the MATLAB part. The Unity application is not included.

## What's here

Quasi-1D isentropic flow of air (γ = 1.4), based on the governing equations in Prof. D. Thévenin's Advanced Fluid Dynamics script (OVGU):

| File | What it does |
|---|---|
| `pressure_ratio.m`, `temperature_ratio.m`, `density_ratio.m` | Isentropic p/p₀, T/T₀ and ρ/ρ₀ as functions of Mach number (0–5.5), plotted and written to CSV |
| `mach_subsonic_every_step.m` | Solves the area–Mach relation for the subsonic branch with Newton-Raphson over A*/A from 0.00001 to 1 |
| `mach_supersonic_every_step.m` | Same for the supersonic branch (Ma > 1) |
| `*.xlsx` | Tabulated area–Mach and property data handed to the visualisation team |

The area–Mach relation solved in both scripts:

$$\frac{A}{A^*} = \frac{1}{Ma}\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}Ma^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}$$

Newton-Raphson tolerance is 10⁻⁹ with at most 200 iterations; each output row records the iteration count and residual.

## Status

Sprint 1 of 10 (05/2026): 1D equations and solvers. Planned next: nozzle wall contours (method of characteristics, Rao bell contour), 2D flow, shocks, and integration with the Unity front end.
