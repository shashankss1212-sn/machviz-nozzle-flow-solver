#!/usr/bin/env python3
"""
Richardson extrapolation of the nozzle grid study - short, step-by-step version.

Input : input_data.csv  (3 meshes per study: fine, medium, coarse; values taken from the solver results)
Output: results.csv, richardson_bell_thrust.png, and a printed worked example (bell thrust)

Run   : python richardson.py
Needs : numpy, pandas, matplotlib

The calculation (f1 = fine, f2 = medium, f3 = coarse mesh)
----------------------------------------------------------
1. Mesh spacing     h_i  = 1 / nx_i            (nx = axial cells in the nozzle; all directions are refined together)
2. Refinement ratio r21  = h2/h1 = nx1/nx2 ,  r32 = h3/h2 = nx2/nx3
3. Differences      e21  = f2 - f1 ,  e32 = f3 - f2
4. Convergence      R    = e21 / e32
                    0 < R < 1  -> monotonic convergence (the differences shrink): Richardson can be used
                    R < 0      -> oscillating ;  R >= 1 -> diverging   : Richardson is not reliable
5. Order p          equal ratios (r21 = r32 = r): p = ln(e32/e21) / ln r          (slide 26, bend-pipe exercise)
                    unequal ratios: solve  p = | ln|e32/e21| + q(p) | / ln r21,
                                           q(p) = ln( (r21^p - s) / (r32^p - s) ),  s = sign(e32/e21)
                                    by repeating it until p stops changing        (Celik et al. 2008)
6. Extrapolation    f(h=0) = f1 + (f1 - f2) / (r21^p - 1)
7. Fine-mesh error  |f1 - f(h=0)| / |f(h=0)|
8. GCI (uncertainty band of the fine mesh, Celik et al. 2008):
                    GCI = 1.25 * |(f1 - f2)/f1| / (r21^p - 1)
9. Reliability      trusted when the convergence is monotonic AND 0.5 <= p <= 3
                    (the solver is first order, so p should be close to 1)
"""
import math
import numpy as np
import pandas as pd

QUANTITIES = {                      # column in input_data.csv : (name, unit, scale)
    "thrust_wall_N": ("Thrust (wall route)", "MN", 1e-6),
    "thrust_exit_plane_N": ("Thrust (exit plane)", "MN", 1e-6),
    "mass_flow_kg_s": ("Mass flow", "kg/s", 1.0),
    "isp_s": ("Specific impulse", "s", 1.0),
    "Cd": ("Discharge coefficient Cd", "-", 1.0),
    "exit_mach_mass_avg": ("Mass-averaged exit Mach", "-", 1.0),
}
FS = 1.25                           # safety factor of the GCI
P_OK = (0.5, 3.0)                   # credible range of the observed order


def order_of_convergence(e21, e32, r21, r32):
    """Step 5. Exact slide-26 formula when the ratios are equal, otherwise Celik's iteration."""
    if abs(r21 - r32) < 1e-9:
        return math.log(e32 / e21) / math.log(r21)
    s = np.sign(e32 / e21)
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)          # first guess (q = 0)
    for _ in range(200):
        a, b = r21 ** p - s, r32 ** p - s
        if a <= 0 or b <= 0:
            return float("nan")
        p_new = abs(math.log(abs(e32 / e21)) + math.log(a / b)) / math.log(r21)
        if abs(p_new - p) < 1e-12:
            break
        p = p_new
    return p_new


def richardson(f1, f2, f3, nx1, nx2, nx3):
    r21, r32 = nx1 / nx2, nx2 / nx3                              # step 2
    e21, e32 = f2 - f1, f3 - f2                                  # step 3
    out = dict(r21=r21, r32=r32, e21=e21, e32=e32, R=np.nan, convergence="no change",
               p=np.nan, f_extrapolated=np.nan, fine_mesh_error_pct=np.nan, GCI_pct=np.nan, reliable=False)
    if e21 == 0 or e32 == 0:
        return out
    R = e21 / e32                                                # step 4
    out["R"] = R
    out["convergence"] = ("monotonic" if 0 < R < 1 else "oscillating" if R < 0 else "diverging")
    if out["convergence"] != "monotonic":
        return out
    p = order_of_convergence(e21, e32, r21, r32)                 # step 5
    rp = r21 ** p
    f_ext = f1 + (f1 - f2) / (rp - 1)                            # step 6
    out.update(p=p, f_extrapolated=f_ext,
               fine_mesh_error_pct=100 * abs(f1 - f_ext) / abs(f_ext),          # step 7
               GCI_pct=100 * FS * abs((f1 - f2) / f1) / (rp - 1),             # step 8
               reliable=bool(P_OK[0] <= p <= P_OK[1]))                         # step 9
    return out


def main():
    data = pd.read_csv("input_data.csv")
    rows = []
    for study, d in data.groupby("study", sort=False):
        d = d.sort_values("nx", ascending=False)                 # fine, medium, coarse
        nx = d.nx.values.astype(float)
        for col, (name, unit, sc) in QUANTITIES.items():
            f = d[col].values.astype(float) * sc
            res = richardson(*f, *nx)
            rows.append(dict(study=study, quantity=name, unit=unit, fine=f[0], medium=f[1], coarse=f[2], **res))
    R = pd.DataFrame(rows)
    R.to_csv("results.csv", index=False)

    # ---- worked example: bell thrust, 5 / 2.5 / 1 lakh meshes
    ex = R[(R.study.str.startswith("bell grid")) & (R.quantity == "Thrust (wall route)")].iloc[0]
    print("Worked example - bell, wall-route thrust, 5 / 2.5 / 1 lakh meshes")
    print(f"  f1, f2, f3      = {ex.fine:.5f}, {ex.medium:.5f}, {ex.coarse:.5f} MN")
    print(f"  r21, r32        = {ex.r21:.4f}, {ex.r32:.4f}   (unequal -> iterative p)")
    print(f"  e21, e32        = {ex.e21:+.5f}, {ex.e32:+.5f} MN,  R = {ex.R:.3f} -> {ex.convergence}")
    print(f"  p               = {ex.p:.3f}")
    print(f"  f(h=0)          = {ex.f_extrapolated:.4f} MN")
    print(f"  fine-mesh error = {ex.fine_mesh_error_pct:.2f} %,  GCI = {ex.GCI_pct:.2f} %\n")

    show = R.copy()
    show["result"] = [f"{v:.4f}" if np.isfinite(v) else "-" for v in show.f_extrapolated]
    print(show[["study", "quantity", "fine", "convergence", "p", "result", "fine_mesh_error_pct", "GCI_pct", "reliable"]]
          .round(3).to_string(index=False))

    # ---- one figure: bell thrust against mesh spacing
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for study, c, m in (("bell (180/240/320 mesh)", "#eb6834", "s"), ("bell grid sequence (1/2.5/5 lakh)", "#1baf7a", "o")):
        d = data[data.study == study]; r = R[(R.study == study) & (R.quantity == "Thrust (wall route)")].iloc[0]
        h = 1000 / d.nx.values
        ax.plot(h, d.thrust_wall_N * 1e-6, m, color=c, ms=8, mec="white", label=study)
        ax.errorbar([0], [r.f_extrapolated], yerr=[r.fine * r.GCI_pct / 100], fmt="D", color=c, ms=8, mec="white", capsize=5)
        hh = np.linspace(0, h.max(), 100)
        ax.plot(hh, r.f_extrapolated + (r.fine - r.f_extrapolated) * (hh / h.min()) ** r.p, "-", color=c, lw=1.2, alpha=0.7)
    ax.set_xlabel("mesh spacing h = 1000 / nx   (0 = infinitely fine mesh)")
    ax.set_ylabel("Thrust, wall route [MN]")
    ax.set_title("Bell nozzle: Richardson extrapolation (diamond = f(h=0), bar = GCI)", loc="left", fontsize=11)
    ax.grid(color="#e6e5e0"); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("richardson_bell_thrust.png", dpi=150)


if __name__ == "__main__":
    main()
