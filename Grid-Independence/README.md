# Grid independence study – 2-D Euler nozzle solver

How much do the results of the solver in [`../2D-Solver`](../2D-Solver) change when the mesh gets finer? I ran the same nozzle and jet on 15 meshes for each of four wall profiles, plus three much finer meshes for the bell, and compared mass flow, thrust and the flow field from one mesh to the next.

Part of the MachViz team project at Otto von Guericke University Magdeburg (team of 6). I ran all meshes with my solver; the post-processing scripts and the Excel workbooks were made with an AI coding assistant.

## Case

| | |
|---|---|
| Gas | air, temperature-dependent γ (`--gas-model tpg`) |
| Chamber | p0 = 30 bar, T0 = 3,000 K |
| Ambient | 101,325 Pa (nozzle pressure ratio 29.6) |
| Nozzle | axisymmetric, throat diameter 1.2 m, exit diameter 3.0 m (A_e/A_t = 6.25), divergent length 3 m (lip at x = 3 m) |
| Plume | 15 m long block behind the lip |
| Numerics | first order, hybrid HLL/HLLC flux, CFL 0.35, 40,000 iterations per run |
| Profiles | cosine, bell, conical, super-ellipse |

Mesh names read `F_nx_ny_npl`: nx axial cells in the nozzle, ny radial cells, npl axial cells in the plume. Cells = ny × (nx + npl).

- **Coarse study:** 7 meshes, 105×28×77 to 195×52×143 (5,096 to 17,576 cells), i.e. −30 % to +30 % of 150×40×110.
- **Fine study:** 8 meshes, 180×45×139 to 320×80×246 (14,355 to 45,280 cells), i.e. −10 % to +60 % of 200×50×154.

Each mesh is compared with the next coarser one, not with a fixed base mesh.

## Convergence first

Only one run (bell, 135×36×99) reached the 10⁻⁵ residual target. In all the others the relative residual drops fast, then levels off between 10⁻⁴ and 8 × 10⁻² and cycles (see the convergence plot at the bottom of each `*_mesh_comparison_mach.png`). The workbooks trace the cycle to the second shock cell of the jet, which moves back and forth in pseudo-time. Everything below should be read with that in mind: the nozzle values are steady, the plume beyond the first shock cell is not.

## Results – fine study

Change of each value from the 300×75×231 to the 320×80×246 mesh (last step) and over the whole fine study (14,355 → 45,280 cells):

| Profile | Inlet mass flow, last step / total | Wall-pressure thrust, last step / total | Exit-axis Mach, first → last mesh |
|---|---|---|---|
| Cosine | +0.09 % / +0.95 % | +0.08 % / +0.94 % | 3.250 → 3.256 |
| Bell | +0.08 % / +0.93 % | +0.08 % / +0.96 % | 3.718 → 3.880 |
| Conical | +0.08 % / +0.94 % | +0.08 % / +0.87 % | 3.055 → 2.962 |
| Super-ellipse | +0.09 % / +0.95 % | +0.29 % / +2.63 % | 2.368 → 2.351 |

- **Nozzle integrals settle.** Mass flow and wall thrust change by less than 0.1 % per step on the fine meshes (super-ellipse thrust: 0.3 %), and the step size keeps falling. They still drift in one direction, so the values are mesh-converged to about 1 %, not exactly mesh-independent. For a Richardson extrapolation see the bell meshes below.
- **Centreline values drift more.** The exit-axis Mach number of the bell still rises 0.4 % per step and that of the cone falls 0.3 %. A single cell on the axis is more sensitive than an integral over the exit plane.
- **The plume isn't grid-independent.** The peak Mach number in the first shock cell keeps rising (cosine 4.14 → 4.51), the pressure ratio across the shock that closes that cell grows from 21 to 59, and the whole-field difference in the plume between neighbouring meshes reaches 27 % (cosine) and 39 % (bell) of the field range. More cells resolve a stronger shock, and the cycling second cell adds to it.
- **Thrust check.** The exit-plane and wall-pressure thrust routes differ by 0.6–0.7 % on the finest mesh (super-ellipse 3.1 %), down from 1.1–3.3 % on the coarsest fine mesh.

The recommended values for comparing meshes are inlet mass flow and wall-pressure thrust; the exit-plane values pick up the unsteady flow at the lip.

## Profiles on the same mesh

| Profile | Wall thrust on 320×80×246 | vs cosine | Isp |
|---|---|---|---|
| Bell | 4.62 MN | +2.5 % | 198.7 s |
| Conical | 4.60 MN | +2.0 % | 197.4 s |
| Cosine | 4.51 MN | – | 194.0 s |
| Super-ellipse | 4.39 MN | −2.6 % | 179.5 s |

The gaps between profiles (2–2.6 %) are larger than the remaining change per mesh step (below 0.3 %), and the order bell > conical > cosine > super-ellipse is the same on the 200×50×154 mesh. So the ranking holds even though the plume is not converged. Data: `Super-Ellipse/profiles_same_mesh_comparison.csv`.

## Bell on 100k–500k cells

Three much finer bell meshes:

| Stage | Mesh | Cells | Iterations | Final relative residual | Inlet mass flow | Wall thrust | Exit-axis Mach |
|---|---|---|---|---|---|---|---|
| 1 | 475×119×366 | 100,079 | 12,000 | 0.32 | 2,370 kg/s | 4.643 MN | 3.96 |
| 2 | 752×188×579 | 250,228 | 12,000 | 0.31 | 2,378 kg/s | 4.659 MN | 4.01 |
| 3 | 1060×265×815 | 496,875 | 20,000 | 2.03 | 2,382 kg/s | 4.668 MN | 4.03 |

None of these converged: on the 500k mesh the residual swings between about 0.4 and 1.9 for the whole run. The nozzle values carry on the trend of the fine study (2,360 kg/s and 4.62 MN at 45,280 cells) and mass flow and wall thrust change by less than 0.2 % from stage 2 to stage 3. Stage 3 also has a summary taken at iteration 18,666, in a quieter part of the cycle (residual 0.75). Its nozzle values match the iteration-20,000 values to four digits, while the plume differs (10 vs 12 shock structures, peak Mach 4.39 vs 4.51), which shows again that only the nozzle part is steady.

**Richardson extrapolation.** Extrapolating the wall thrust of these three bell meshes puts the 500k value within about 0.5–0.7 % of the extrapolated thrust; the exact figure depends on how the refinement ratio is taken for these unevenly refined meshes. The runs are not converged to tolerance, so this is an estimate, not proof of mesh independence. The calculation file will be added to this folder.

## Files

Per profile folder (`Cosine/`, `Bell/`, `Conical/`, `Super-Ellipse/`):

| File | What it is |
|---|---|
| `coarse_summary_all_levels.csv`, `fine_summary_all_levels.csv` | One row per mesh: about 80 scalars read from each run's JSON (mass flow, thrust routes, Isp, C_d, exit and throat values, residual, plume and shock counts) |
| `coarse_grid_independence_comparison.xlsx`, `fine_…` | Per study: scalars, centreline profiles (Mach, T, ρ, p, velocity, γ, total pressure, entropy) and field differences, each against the base mesh |
| `grid_independence_successive_<profile>.xlsx` | Values per mesh, % change against the previous mesh, and whole-field change against the previous mesh (finer field interpolated onto the coarser cell centres) |
| `coarse_mesh_comparison_mach.png`, `fine_…` | Mach field on every mesh, centreline Mach of all meshes, convergence histories |

`Super-Ellipse/profiles_same_mesh_comparison.csv` compares the four profiles mesh by mesh. `Bell-500k-Cells/` has the three stage summaries and the Mach and schlieren images of the 500k run.

Not uploaded: the solver checkpoints (`.npz`, 105 MB) and the per-mesh field plots (150 MB). Both come out of the solver again with the mesh sizes above. The workbooks mention `analysis/second_cell_cycle.png`, which isn't in this folder.

## Limits

- Inviscid and first order. The study shows how the Euler solution depends on the mesh, not how close it is to a real nozzle.
- Unconverged runs: the plume numbers are snapshots of a cycling solution, and only the nozzle integrals should be quoted as mesh-converged (to about 1 %).
