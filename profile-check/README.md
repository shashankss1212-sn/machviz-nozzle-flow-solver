# Nozzle profile check

A quick check of divergent-nozzle wall shapes before running CFD. For a given throat diameter, exit diameter and length, the script builds the wall contour and asks two questions along it:

1. **Is the 1-D answer still valid at the wall?** It compares the wall angle θ with the Prandtl-Meyer angle ν of the 1-D Mach number. Where θ > ν, the wall turns away faster than the 1-D flow has expanded, so the real wall Mach number is higher and the wall pressure lower than 1-D predicts. This isn't a shock prediction. It only marks where the 1-D numbers stop describing the wall flow, which in a viscous nozzle means a separation risk.
2. **Does the wall make a shock?** Where the wall turns back toward the axis, it sends compression waves into the flow. The script traces these C⁻ Mach lines and takes the first crossing of two neighbouring lines as the point where a shock starts. This is a planar simple-wave estimate; in a round nozzle the waves focus toward the axis and meet a little earlier.

Part of the MachViz team project at Otto von Guericke University Magdeburg (team of 6). The code was written together with teammates.

## Files

| File | What it is |
|---|---|
| `nozzle_profile_check.py` | The tool. Seven profiles: conical, cosine, ellipse, cubic, quadratic, Rao-type bell, super-ellipse. Prints a report per profile, writes station data (`.csv`) and a 4-panel figure (`.png`). |
| `pm_bench_plot.py` | Plots θ against ν for four profiles from a CSV written by the tool, with the θ > ν zones shaded. |
| `profile_check_4profiles.csv` | Station data, 4,001 points per profile, for bell, super-ellipse, cosine and conical. γ = 1.4. |
| `profile_check_4profiles.png` | The tool's own figure for the same run (contour, angles, ν − θ margin, 1-D Mach). γ = 1.4. |
| `pm_bench_4profiles.png` | Output of `pm_bench_plot.py` for the γ = 1.25 run. |
| `crossing_check_1.png` | The shock check compared with the team's 2-D axisymmetric Euler solver for the four profiles, γ = 1.25. The solver itself is not in this folder. |

## Test case

All files use D_t = 1.2 m, D_e = 2.5 m, L = 2.5 m, axisymmetric (A_e/A_t = 4.34). The bell here has no throat arc: it starts at θ_n = 22° and ends at 8°.

To reproduce the CSV and `profile_check_4profiles.png`:

```
python nozzle_profile_check.py --profile bell,super-ellipse,cosine,conical --dt 1.2 --de 2.5 --L 2.5 --gamma 1.4 --theta-n 22 --bell-arc 0 --out profile_check_4profiles
```

`pm_bench_4profiles.png` and `crossing_check_1.png` come from the same command with `--gamma 1.25`. Note that `pm_bench_plot.py` has "γ = 1.25" written into the figure title, so check the title if you run it on the γ = 1.4 CSV in this folder.

Without `--theta-n 22 --bell-arc 0` the bell gets the default Rao throat arc (0.382 r_t, θ_n = 25°) and its numbers change.

## Results (γ = 1.25, 1-D exit Mach 2.76)

| Profile | Max wall angle | Exit angle | θ > ν zone (x/L) | Shock check |
|---|---|---|---|---|
| Bell | 22.0° | 8.0° | 0 – 0.11 | Mach lines reach the axis before they cross |
| Super-ellipse | 82.7° | 82.7° | 0.95 – 1.00 (and a transonic band at the throat) | No compression, no shock |
| Cosine | 22.2° | 0.0° | 0 – 0.18 | Shock starts at x/L = 1.21, in the plume |
| Conical | 14.6° | 14.6° | 0 – 0.10 | No compression, no shock |

The cosine profile is the only one that turns back hard enough to form a shock from the wall waves. The solver agrees. The solver puts that shock at x/L = 1.15, a little upstream of the tool's 1.21. For the super-ellipse and the cone the solver also finds no wall-generated shock; its first shocks there (x/L 1.76 and 1.81) are Mach discs from over-expansion. For the bell the tool makes no prediction, and the solver shows a shock at x/L = 1.03 after the waves reflect off the axis.

With γ = 1.4 (the CSV) the exit Mach number is 3.03, the cosine shock moves to x/L = 1.18 and the θ > ν zones grow by 0.01–0.02 in x/L. The shock check gives the same answer for the other three profiles.

## Limits

1-D theory is poor near the throat (|M − 1| < 0.2), and the tool doesn't model the convergent section, viscosity or back pressure. Use it to screen shapes, then run the 2-D solver on the ones you keep.

## Running it

Python 3 with numpy and matplotlib. Run `python nozzle_profile_check.py` with no arguments to be prompted for the inputs, or `--help` for all options.
