# MachViz – nozzle flow analysis

Flow through a converging-diverging nozzle, worked out three ways: 1-D theory, a quick check of wall shapes, and a 2-D solver with its own mesh study. MachViz team project at Otto von Guericke University Magdeburg (team of 6). Each folder has its own README with the method and the results.

| Folder | What it is | Tools | Written by |
|---|---|---|---|
| [1D-Solution](1D-Solution) | 1-D isentropic solution: Newton-Raphson on the area–Mach relation for the subsonic and supersonic branches, with T, p and ρ ratios for 99,901 area ratios (γ = 1.4) | MATLAB | me |
| [Profile-Check](Profile-Check) | Screens divergent wall shapes: where the 1-D answer stops holding at the wall (wall angle vs Prandtl-Meyer angle) and whether waves from the wall form a shock | Python (NumPy, Matplotlib) | me with teammates |
| [2D-Solver](2D-Solver) | 2-D Euler finite-volume solver (HLLC/HLL fluxes, MUSCL, SSP-RK2), planar or axisymmetric, with temperature-dependent gas properties, shock detection and thrust | Python (NumPy, Matplotlib) | me, with an AI coding assistant |
| [Grid-Independence](Grid-Independence) | Mesh study of the 2-D solver: four wall profiles on 15 meshes each (5k–45k cells) plus a bell case up to 500k cells; nozzle mass flow and thrust settle to about 1 %, the plume does not | Python, Excel | me, with an AI coding assistant |

The 1-D solution gives the reference numbers, the profile check sorts out which wall shapes are worth a closer look, the 2-D solver gives the full flow field for those, and the grid independence study shows how far its numbers can be trusted.
