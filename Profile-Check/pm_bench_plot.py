"""Profile check only: wall angle theta vs Prandtl-Meyer angle nu(M_1D) for each profile."""
import csv, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "profile_check_4profiles.csv"
order = ["bell", "super-ellipse", "cosine", "conical"]
D = {}
for r in csv.DictReader(open(src)):
    D.setdefault(r["profile"], []).append(r)

def zones(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))

fig, axs = plt.subplots(2, 2, figsize=(15, 9.5))
print(f"{'profile':14s} {'max theta':>9s} {'exit theta':>10s} {'exit nu':>8s} {'min margin':>10s}  theta > nu zones (x/L, M range)")
for ax, p in zip(axs.flat, order):
    R = D[p]
    xL = np.array([float(r["x_over_L"]) for r in R]); th = np.array([float(r["theta_deg"]) for r in R])
    nu = np.array([float(r["nu_deg"]) for r in R]); M = np.array([float(r["Mach_1D"]) for r in R])
    mg = nu - th
    ax.plot(xL, th, "k-", lw=2.5, label="θ_wall (wall angle)")
    ax.plot(xL, nu, "g--", lw=2.2, label="ν(M_1D) (Prandtl-Meyer angle)")
    ax.fill_between(xL, th, nu, where=mg >= 0, color="g", alpha=0.10, label="θ ≤ ν  (1-D valid)")
    ax.fill_between(xL, th, nu, where=mg < 0, color="r", alpha=0.30, label="θ > ν  (wall over-turns vs 1-D)")
    txt = []
    for i0, i1 in zones(mg < 0):
        tag = "transonic" if M[i1] < 1.2 else "supersonic"
        k = i0 + np.argmin(mg[i0:i1 + 1])
        txt.append(f"x/L {xL[i0]:.3f}–{xL[i1]:.3f}  (M {M[i0]:.2f}–{M[i1]:.2f}, {tag})\n"
                   f"   worst: θ {th[k]:.1f}° vs ν {nu[k]:.1f}°")
        ax.axvline(xL[i0], color="r", lw=0.8, ls=":"); ax.axvline(xL[i1], color="r", lw=0.8, ls=":")
    zs = "; ".join(t.replace("\n   ", ", ") for t in txt) or "none"
    print(f"{p:14s} {th.max():9.2f} {th[-1]:10.2f} {nu[-1]:8.2f} {mg.min():10.2f}  {zs}")
    box = (f"max θ {th.max():.1f}°, exit θ {th[-1]:.1f}°, exit ν {nu[-1]:.1f}°\n"
           "θ > ν zones:\n" + ("\n".join(txt) if txt else "none"))
    ax.text(0.98, 0.04, box, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(fc="white", ec="0.5", alpha=0.92))
    ax.set_title(p, fontweight="bold"); ax.set_xlabel("x / L"); ax.set_ylabel("angle (deg)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 90 if th.max() > 60 else 60); ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Profile check: wall angle θ vs Prandtl-Meyer angle ν(M_1D)   |   D_t 1.2 m, D_e 2.5 m, L 2.5 m, "
             "axisymmetric, γ = 1.25", fontweight="bold")
fig.tight_layout(); fig.savefig("pm_bench_4profiles.png", dpi=110)
print("figure -> pm_bench_4profiles.png")
