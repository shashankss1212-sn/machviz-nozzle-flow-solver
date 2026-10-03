# 1-D isentropic nozzle solution

A MATLAB program that turns an area ratio into the 1-D isentropic flow state. For every A*/A from 0.001 to 1 (step 0.00001, 99,901 points) it finds both Mach numbers, the subsonic and the supersonic one, and then the temperature, pressure and density ratios for each. These are the 1-D reference numbers that the rest of the nozzle work is checked against.

Part of the MachViz team project at Otto von Guericke University Magdeburg (team of 6). The program puts five earlier project scripts into one: the subsonic and supersonic area–Mach solvers and the temperature, pressure and density ratio scripts. It uses only their equations.

## Files

| File | What it is |
|---|---|
| `isentropic_area_ratio_combined.m` | The program. Solves both branches, writes the CSV, prints a table for chosen A/A* values and draws five plots. |
| `combined_isentropic_results.zip` | Holds `combined_isentropic_results.csv` (20 MB, 99,901 rows, γ = 1.4). Zipped to fit the upload; unzip before use. |

CSV columns: `AstarA`, `AR` (= A/A*), then for each branch the Mach number, T/T_R, p/p_R, ρ/ρ_R, the Newton-Raphson iteration count and the final residual. The subscript R means reservoir (stagnation) value.

## Method

The area–Mach relation

A/A* = (1/Ma) · [1 + α(Ma² − 1)]ⁿ, with n = (k+1)/(2(k−1)) and α = (k−1)/(k+1),

has two roots for every A/A* > 1. The program solves it by Newton-Raphson with an analytic derivative, once from a subsonic start (Ma₀ = min(0.95, A*/A)) and once from a supersonic start (Ma₀ = 1 + 0.9(A/A* − 1) + 0.05). The iteration stops when the Mach step is below 10⁻⁹. At the throat (A/A* = 1) both roots are set to Ma = 1.

Then

- T/T_R = 1 / (1 + (k−1)/2 · Ma²)
- p/p_R = (T/T_R)^(k/(k−1))
- ρ/ρ_R = (T/T_R)^(1/(k−1))

All points are solved at once (vectorised), not one by one.

## Check

I put every Mach number in the CSV back into the area–Mach relation. The largest relative error in A/A* is 2 × 10⁻¹⁴, and no point needed more than 25 iterations. A few rows (γ = 1.4):

| A/A* | Ma subsonic | Ma supersonic | p/p_R supersonic |
|---|---|---|---|
| 2 | 0.306 | 2.197 | 0.0939 |
| 4.34 | 0.135 | 3.026 | 0.0262 |
| 10 | 0.058 | 3.923 | 0.0073 |
| 25 | 0.023 | 5.000 | 0.0019 |

A/A* = 4.34 is the test nozzle in `../profile-check` (D_t 1.2 m, D_e 2.5 m), and its supersonic Mach of 3.026 matches the exit Mach number there.

## Running it

Open the script in MATLAB and set the inputs at the top: `k` (γ), the A*/A range and step, the tolerance and `AR_query`, a list of A/A* values to print. Run it. It writes `combined_isentropic_results.csv` to the working folder and opens five figures: Mach against A/A*, the three ratios against A/A* for each branch, the ratios against Mach, and A/A* against Mach.
