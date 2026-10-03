#!/usr/bin/env python3
"""
nozzle_profile_check.py: divergent-nozzle wall profiles and a quick 1-D / Mach-wave check.

Profiles (divergent section only, throat at x = 0, exit at x = L):
  conical        straight wall, constant half-angle
  cosine         r = r_t + dr * (1 - cos(pi*xi))/2, flat at both ends
  ellipse        ellipse arc, flat at the throat, exit angle set by --theta-e
                 (--theta-e 90 gives the quarter ellipse, vertical at the exit)
  cubic          cubic Hermite with start angle --theta-i and exit angle --theta-e
                 (0/0 gives the S-curve 3xi^2 - 2xi^3)
  quadratic      parabola r = r_t + tan(theta_i) x + a x^2
  bell           Rao-type parabola: throat arc of 0.382 r_t up to --theta-n,
                 then a quadratic Bezier down to --theta-e at the exit
  super-ellipse  r = H - B (1 - (tr*xi)^p)^(1/n), same curve as the MATLAB
                 super-ellipse scripts (--se-n, --se-p, --se-trunc)
  all            all of the above

Check 1, 1-D validity: theta_wall <= nu(M_1D)
  M_1D comes from the area ratio and nu is the Prandtl-Meyer angle. If the
  check fails, the wall turns away from the flow faster than the 1-D flow
  has expanded. An inviscid flow still follows the wall through a
  Prandtl-Meyer fan, so the wall Mach is higher than M_1D and the wall
  pressure is lower. This is therefore not a shock and not a detachment
  prediction: it shows where the 1-D numbers stop describing the wall flow.
  In a viscous nozzle a large over-expansion there means separation risk.
  The 2-D Euler run of the super-ellipse showed no shock in the flagged
  exit zone (wall Mach 3.6 against 2.6 from 1-D).

Check 2, compression / shock: Mach-wave coalescence
  Where the wall turns back toward the axis (d theta/dx < 0) it sends
  compression waves into the flow along C- Mach lines at angle theta - mu.
  The first crossing of two neighbouring lines is taken as the point where
  a shock starts. This is a planar simple-wave estimate; in a round nozzle
  the waves focus toward the axis and meet a little earlier.

Not covered: the transonic region at the throat (1-D is poor for |M-1| < 0.2),
the convergent section, viscosity and back pressure. Use the 2-D Euler solver
(cd_nozzle_solver_axiconv.py) for those.

Usage:
  python nozzle_profile_check.py                       (prompts for the inputs)
  python nozzle_profile_check.py --profile bell --dt 1.2 --de 2.5 --L 2.5
  python nozzle_profile_check.py --profile all  --dt 1.2 --de 2.5 --L 2.5
  python nozzle_profile_check.py --profile cubic --dt 1.2 --de 2.5 --L 2.5 --theta-e 10
  python nozzle_profile_check.py --profile super-ellipse --dt 1.2 --de 2.5 --L 2.5 --se-trunc 0.998
  python nozzle_profile_check.py --help

Writes a console summary, <out>.csv (station data) and <out>.png (figure).
Needs numpy and matplotlib.
"""
import argparse
import math
import sys

import numpy as np

PROFILES = ["conical", "cosine", "ellipse", "cubic", "quadratic", "bell", "super-ellipse"]


# Gas dynamics (calorically perfect gas)
def area_ratio(M, g):
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** (
        (g + 1.0) / (2.0 * (g - 1.0)))


def mach_from_area(AR, g):
    """Supersonic root of the area-Mach relation (vectorised bisection)."""
    AR = np.atleast_1d(np.asarray(AR, float))
    lo = np.ones_like(AR)
    hi = np.full_like(AR, 60.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        big = area_ratio(mid, g) < AR
        lo = np.where(big, mid, lo)
        hi = np.where(big, hi, mid)
    M = 0.5 * (lo + hi)
    M[AR <= 1.0] = 1.0
    return M


def pm_nu_deg(M, g):
    M = np.asarray(M, float)
    k = math.sqrt((g + 1.0) / (g - 1.0))
    s = np.sqrt(np.maximum(M * M - 1.0, 0.0))
    return np.degrees(k * np.arctan(s / k) - np.arctan(s))


def p_over_p0(M, g):
    return (1.0 + 0.5 * (g - 1.0) * M * M) ** (-g / (g - 1.0))


# Wall profiles: r(x) for 0 <= x <= L
def _ellipse_k(dr, L, theta_e_deg):
    """Ellipse h = H - B sqrt(1-(x/A)^2), A = L/k.  Solve k for the exit angle.
    tan(theta_e) = dr k^2 / (L c (1-c)),  c = sqrt(1-k^2);  k -> 0 is the
    parabola limit (smallest angle atan(2 dr/L)), k -> 1 the quarter ellipse."""
    t_min = math.degrees(math.atan(2.0 * dr / L))
    if theta_e_deg >= 89.999:
        return 1.0, 90.0, False
    clamped = theta_e_deg < t_min
    te = max(theta_e_deg, t_min + 1e-6)

    def f(k):
        c = math.sqrt(1.0 - k * k)
        return math.degrees(math.atan(dr * k * k / (L * c * (1.0 - c)))) - te
    lo, hi = 1e-6, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), te, clamped


def build_profile(name, r_t, r_e, L, opt, N):
    """Return x, r, info-dict for one profile."""
    x = np.linspace(0.0, L, N)
    xi = x / L
    dr = r_e - r_t
    info = {}
    ti = math.radians(opt.theta_i)
    te = math.radians(opt.theta_e)

    if name == "conical":
        r = r_t + dr * xi
        info["half_angle_deg"] = math.degrees(math.atan(dr / L))

    elif name == "cosine":
        r = r_t + dr * 0.5 * (1.0 - np.cos(np.pi * xi))

    elif name == "ellipse":
        k, te_used, clamped = _ellipse_k(dr, L, opt.theta_e)
        if k >= 1.0:                                  # quarter ellipse
            r = r_e - dr * np.sqrt(np.clip(1.0 - xi * xi, 0.0, 1.0))
        else:
            A = L / k
            c = math.sqrt(1.0 - k * k)
            B = dr / (1.0 - c)
            r = (r_t + B) - B * np.sqrt(np.clip(1.0 - (x / A) ** 2, 0.0, 1.0))
        info["exit_angle_requested_deg"] = opt.theta_e
        if clamped:
            info["note"] = (f"theta_e clamped to the minimum {te_used:.2f} deg "
                            f"(= atan(2 dr / L)) for a flat-throat ellipse")

    elif name == "cubic":
        # Hermite: r(0)=r_t, r(L)=r_e, r'(0)=tan(ti), r'(L)=tan(te)
        h00 = 2 * xi**3 - 3 * xi**2 + 1
        h10 = xi**3 - 2 * xi**2 + xi
        h01 = -2 * xi**3 + 3 * xi**2
        h11 = xi**3 - xi**2
        r = h00 * r_t + h10 * L * math.tan(ti) + h01 * r_e + h11 * L * math.tan(te)

    elif name == "quadratic":
        a = (dr - math.tan(ti) * L) / L**2
        r = r_t + math.tan(ti) * x + a * x**2
        if a < 0:
            info["note"] = "concave parabola (theta_i too large for this length)"

    elif name == "bell":
        tn = math.radians(opt.theta_n)
        Rd = opt.bell_arc * r_t                       # downstream throat arc
        xN = Rd * math.sin(tn)
        rN = r_t + Rd * (1.0 - math.cos(tn))
        if xN >= L or rN >= r_e:
            raise ValueError("bell: throat arc already reaches the exit -- "
                             "lower --theta-n or increase L / D_exit")
        # control point: intersection of the tangents at N (theta_n) and E (theta_e)
        m1, m2 = math.tan(tn), math.tan(te)
        xQ = (r_e - rN + m1 * xN - m2 * L) / (m1 - m2)
        rQ = rN + m1 * (xQ - xN)
        if not (xN < xQ < L):
            raise ValueError(f"bell: no valid Bezier for theta_n={opt.theta_n} / "
                             f"theta_e={opt.theta_e} with this L and D_exit "
                             f"(tangent intersection at x={xQ:.3f}). Adjust the angles "
                             f"or the length.")
        # arc part
        na = max(int(N * xN / L), 20)
        th = np.linspace(0.0, tn, na)
        xa, ra = Rd * np.sin(th), r_t + Rd * (1.0 - np.cos(th))
        # Bezier part, sampled densely then put on the common x grid
        t = np.linspace(0.0, 1.0, 4 * N)
        xb = (1 - t)**2 * xN + 2 * (1 - t) * t * xQ + t**2 * L
        rb = (1 - t)**2 * rN + 2 * (1 - t) * t * rQ + t**2 * r_e
        xx = np.concatenate([xa, xb[1:]])
        rr = np.concatenate([ra, rb[1:]])
        r = np.interp(x, xx, rr)
        info["arc_end_x"] = xN
        info["theta_n_deg"] = opt.theta_n

    elif name == "super-ellipse":
        p, n, tr = opt.se_p, opt.se_n, min(max(opt.se_trunc, 1e-6), 1.0)
        c = (1.0 - tr**p) ** (1.0 / n)
        B = dr / max(1.0 - c, 1e-12)
        r = (r_t + B) - B * (1.0 - np.clip(tr * xi, 0.0, 1.0) ** p) ** (1.0 / n)
        if tr >= 1.0:
            info["note"] = "se-trunc = 1: wall is vertical at the exit (infinite slope)"

    else:
        raise ValueError(f"unknown profile '{name}'")
    return x, r, info


# Analysis
def zones(mask):
    """Contiguous True runs -> list of (i0, i1)."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def analyse(name, r_t, r_e, L, opt):
    x, r, info = build_profile(name, r_t, r_e, L, opt, opt.N)
    g = opt.gamma
    drdx = np.gradient(r, x)
    theta = np.degrees(np.arctan(drdx))
    AR = (r / r_t) if opt.planar else (r / r_t) ** 2
    M = mach_from_area(AR, g)
    nu = pm_nu_deg(M, g)
    mu = np.degrees(np.arcsin(1.0 / np.maximum(M, 1.0 + 1e-12)))
    margin = nu - theta

    res = dict(name=name, x=x, r=r, theta=theta, M=M, nu=nu, mu=mu, margin=margin,
               info=info, L=L, r_t=r_t, r_e=r_e)

    # check 1: 1-D validity (theta_wall <= nu)
    att = []
    for i0, i1 in zones(margin < 0):
        k = i0 + int(np.argmin(margin[i0:i1 + 1]))
        transonic = M[i1] < 1.0 + opt.transonic
        att.append(dict(i0=i0, i1=i1, x0=x[i0], x1=x[i1], r0=r[i0], r1=r[i1],
                        M0=M[i0], M1=M[i1], worst=margin[k], xw=x[k],
                        theta=theta[k], nu=nu[k], transonic=transonic))
    res["attach"] = att
    res["attach_real"] = [z for z in att if not z["transonic"]]

    # check 2: compression waves and where they cross
    dth = np.gradient(theta, x)
    comp = dth < -1.0 / L                     # turning back faster than 1 deg per L
    for i0, i1 in zones(comp):                # drop numerical noise (< 0.05 deg)
        if theta[i0] - theta[i1] < 0.05:
            comp[i0:i1 + 1] = False
    res["comp_mask"] = comp
    shock = None
    lines = []
    if comp.any():
        stride = max(opt.N // 800, 1)
        idx = np.where(comp)[0][::stride]
        slope = np.tan(np.radians(theta[idx] - mu[idx]))          # C- Mach lines
        xs, rs = x[idx], r[idx]
        lines = list(zip(xs, rs, slope))
        best = None
        for a in range(len(idx) - 1):
            b = a + 1
            if idx[b] - idx[a] > 2 * stride:                       # different zone
                continue
            ds = slope[a] - slope[b]
            if abs(ds) < 1e-14:
                continue
            xc = (rs[b] - rs[a] + slope[a] * xs[a] - slope[b] * xs[b]) / ds
            rc = rs[a] + slope[a] * (xc - xs[a])
            if xc > xs[b] and rc > 0.0:
                if best is None or xc < best[0]:
                    best = (xc, rc, xs[a], M[idx[a]])
        first = idx[0]
        x_axis = x[first] - r[first] / np.tan(np.radians(theta[first] - mu[first]))
        cz = zones(comp)
        shock = dict(zones=[(x[i0], x[i1], theta[i0], theta[i1]) for i0, i1 in cz],
                     x_axis_first=x_axis,
                     x=None if best is None else best[0],
                     r=None if best is None else best[1],
                     x_src=None if best is None else best[2],
                     M_src=None if best is None else best[3])
    res["shock"] = shock
    res["comp_lines"] = lines
    return res


# Reporting
def report(res, opt):
    L, x = res["L"], res["x"]
    th, M, nu = res["theta"], res["M"], res["nu"]
    print("-" * 78)
    print(f" PROFILE: {res['name']}")
    for k, v in res["info"].items():
        if k == "note":
            print(f"   note: {v}")
        else:
            print(f"   {k:28s}: {v:.4g}" if isinstance(v, float) else f"   {k}: {v}")
    print(f"   max wall angle          : {th.max():8.3f} deg at x/L = {x[np.argmax(th)]/L:.3f}")
    print(f"   exit wall angle         : {th[-1]:8.3f} deg")
    print(f"   exit Mach (1-D)         : {M[-1]:8.4f}     p_e/p0 = {p_over_p0(M[-1], opt.gamma):.5f}")
    lam = 0.5 * (1.0 + math.cos(math.radians(th[-1])))
    print(f"   divergence factor       : {lam:8.4f}     ((1+cos theta_e)/2, thrust loss {100*(1-lam):.2f} %)")

    print("   [1] 1-D VALIDITY  theta_wall <= nu(M_1D)  (fail = wall over-expands vs 1-D; not a shock):")
    if not res["attach"]:
        print("       PASS everywhere")
    for z in res["attach"]:
        tag = "transonic band, 1-D not valid anyway" if z["transonic"] else "supersonic"
        print(f"       over-turn x/L {z['x0']/L:.4f} -> {z['x1']/L:.4f}  (r {z['r0']:.3f} -> {z['r1']:.3f} m,"
              f" M {z['M0']:.3f} -> {z['M1']:.3f})  [{tag}]")
        print(f"            worst: theta {z['theta']:.2f} deg vs nu {z['nu']:.2f} deg"
              f" (margin {z['worst']:.2f}) at x/L {z['xw']/L:.4f}")

    print("   [2] COMPRESSION / SHOCK (Mach-wave coalescence):")
    s = res["shock"]
    if s is None:
        print("       wall never turns back toward the axis -> no compression waves,"
              " no wall-generated shock")
    else:
        for (a, b, t0, t1) in s["zones"]:
            print(f"       wall turns back x/L {a/L:.4f} -> {b/L:.4f}"
                  f"  (theta {t0:.2f} -> {t1:.2f} deg)")
        if s["x"] is None:
            xa = s['x_axis_first']
            print("       Mach lines reach the axis before crossing -> no wall-shock coalescence"
                  f" (first wave hits the axis at x = {xa:.3f} m, x/L = {xa/L:.3f}"
                  + (", beyond the exit)" if xa > L else ", inside the nozzle;"
                     " axisymmetric focusing there can still steepen it into a weak shock)"))
        else:
            where = "INSIDE the nozzle" if s["x"] <= L else "OUTSIDE (in the plume)"
            print(f"       SHOCK INITIATION at x = {s['x']:.4f} m (x/L = {s['x']/L:.4f}),"
                  f" r = {s['r']:.4f} m  -> {where}")
            print(f"       generated by the wall at x/L = {s['x_src']/L:.4f} (M = {s['M_src']:.3f})")


def summary_table(results, opt):
    print("=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    print(f" {'profile':14s} {'max th':>7s} {'exit th':>8s} {'M_exit':>7s} "
          f"{'1-D valid? (x/L)':>18s} {'shock start (x/L)':>20s}")
    for r in results:
        L = r["L"]
        att = r["attach_real"]
        a = "yes" if not att else "no " + ", ".join(f"{z['x0']/L:.2f}-{z['x1']/L:.2f}" for z in att)
        s = r["shock"]
        if s is None:
            sh = "none"
        elif s["x"] is None:
            sh = "waves reach axis"
        else:
            sh = f"{s['x']/L:.3f}" + ("" if s["x"] <= L else " (plume)")
        print(f" {r['name']:14s} {r['theta'].max():7.2f} {r['theta'][-1]:8.2f} {r['M'][-1]:7.3f} "
              f"{a:>18s} {sh:>20s}")
    print("=" * 78)


def write_csv(results, path, opt):
    with open(path, "w") as f:
        f.write("profile,x_m,r_m,x_over_L,theta_deg,Mach_1D,nu_deg,mu_deg,margin_deg,"
                "p_over_p0,compression\n")
        for r in results:
            pp = p_over_p0(r["M"], opt.gamma)
            for i in range(len(r["x"])):
                f.write(f"{r['name']},{r['x'][i]:.7f},{r['r'][i]:.7f},{r['x'][i]/r['L']:.6f},"
                        f"{r['theta'][i]:.6f},{r['M'][i]:.6f},{r['nu'][i]:.6f},"
                        f"{r['mu'][i]:.6f},{r['margin'][i]:.6f},{pp[i]:.7f},"
                        f"{int(r['comp_mask'][i])}\n")


def plot(results, path, opt, title):
    import matplotlib
    if not opt.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
    single = len(results) == 1
    for k, r in enumerate(results):
        c = cols[k % len(cols)]
        L, x = r["L"], r["x"]
        xn = x / L
        # contour
        A = ax[0, 0]
        A.plot(x, r["r"], "-", color=c, lw=2, label=r["name"])
        for z in r["attach_real"]:
            A.plot(x[z["i0"]:z["i1"] + 1], r["r"][z["i0"]:z["i1"] + 1], "-", color="red", lw=5,
                   alpha=0.5, solid_capstyle="butt")
        if r["comp_mask"].any():
            A.plot(x[r["comp_mask"]], r["r"][r["comp_mask"]], ".", color="k", ms=1.2, alpha=0.35)
        if single:
            for (x0, r0, s) in r["comp_lines"][::max(len(r["comp_lines"]) // 25, 1)]:
                xe = x0 - r0 / s if s < 0 else L * 1.3
                xx = np.array([x0, min(xe, L * 1.3)])
                A.plot(xx, r0 + s * (xx - x0), "-", color="orange", lw=0.5, alpha=0.7)
        s = r["shock"]
        if s is not None and s["x"] is not None:
            A.plot([s["x"]], [s["r"]], "*", color=c, ms=16, mec="k")
        # angles
        A = ax[0, 1]
        A.plot(xn, r["theta"], "-", color=c, lw=2, label=f"{r['name']}: θ_wall")
        A.plot(xn, r["nu"], "--", color=c, lw=1.3, label=f"{r['name']}: ν(M_1D)" if single else None)
        # margin
        A = ax[1, 0]
        A.plot(xn, r["margin"], "-", color=c, lw=2, label=r["name"])
        # Mach
        A = ax[1, 1]
        A.plot(xn, r["M"], "-", color=c, lw=2, label=r["name"])

    A = ax[0, 0]
    A.set_xlabel("x (m)"); A.set_ylabel("r (m)"); A.set_ylim(bottom=0); A.grid(alpha=.3)
    A.set_title("Wall contour  (red = wall over-turns vs 1-D, black dots = wall turning back,\n"
                "★ = predicted shock initiation)", fontsize=10)
    A.legend(fontsize=8, loc="upper left")
    A = ax[0, 1]
    if not single:
        A.plot([], [], "k--", lw=1.3, label="dashed: ν(M_1D) of each profile")
        for k, r in enumerate(results):
            A.plot(r["x"] / r["L"], r["nu"], "--", color=cols[k % len(cols)], lw=1.0)
    A.set_xlabel("x / L"); A.set_ylabel("angle (deg)"); A.grid(alpha=.3)
    A.set_ylim(0, max(90, 1.05 * max(r["theta"].max() for r in results)) if
               max(r["theta"].max() for r in results) > 60 else None)
    A.legend(fontsize=7, loc="upper left"); A.set_title("Wall angle vs Prandtl-Meyer limit")
    A = ax[1, 0]
    A.axhline(0, color="k", ls="--", lw=1)
    A.set_xlabel("x / L"); A.set_ylabel("ν − θ (deg)"); A.grid(alpha=.3)
    A.legend(fontsize=8); A.set_title("ν − θ margin (− = wall over-turns vs 1-D: 1-D not valid there)")
    A = ax[1, 1]
    A.set_xlabel("x / L"); A.set_ylabel("Mach (1-D)"); A.grid(alpha=.3)
    A.legend(fontsize=8); A.set_title("1-D Mach number")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    if opt.show:
        plt.show()


# Command line and interactive input
def build_parser():
    ap = argparse.ArgumentParser(
        description="Divergent nozzle profiles + 1-D validity check + Mach-wave shock check",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--profile", default=None,
                    help="one of: " + ", ".join(PROFILES + ["all"]) +
                         " (or a comma list, e.g. bell,cubic)")
    ap.add_argument("--dt", type=float, default=None, help="throat DIAMETER (m)")
    ap.add_argument("--de", type=float, default=None, help="exit DIAMETER (m)")
    ap.add_argument("--L", type=float, default=None, help="divergent length (m)")
    ap.add_argument("--theta-i", type=float, default=0.0,
                    help="initial wall angle at the throat, deg (cubic, quadratic)")
    ap.add_argument("--theta-e", type=float, default=8.0,
                    help="exit wall angle, deg (bell, ellipse, cubic)")
    ap.add_argument("--theta-n", type=float, default=25.0,
                    help="bell: wall angle at the end of the throat arc, deg")
    ap.add_argument("--bell-arc", type=float, default=0.382,
                    help="bell: downstream throat-arc radius / r_t (Rao: 0.382)")
    ap.add_argument("--se-n", type=float, default=3.0, help="super-ellipse radial exponent n")
    ap.add_argument("--se-p", type=float, default=2.0, help="super-ellipse axial exponent p")
    ap.add_argument("--se-trunc", type=float, default=0.998,
                    help="super-ellipse truncation (1 = vertical exit wall)")
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--planar", action="store_true", help="planar 2-D (A/A* = r/r_t)")
    ap.add_argument("--transonic", type=float, default=0.2,
                    help="over-turn zones ending below M = 1 + this are "
                         "reported as transonic (1-D not valid there)")
    ap.add_argument("--N", type=int, default=4001, help="stations along the wall")
    ap.add_argument("--out", default=None, help="output stem (default: auto)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--show", action="store_true", help="open the figure window")
    return ap


def ask(prompt, cast, default):
    raw = input(f"  {prompt} [{default}]: ").strip()
    return cast(raw) if raw else default


def main(argv=None):
    ap = build_parser()
    opt = ap.parse_args(argv)

    if opt.profile is None or opt.dt is None or opt.de is None or opt.L is None:
        print("Interactive input (press Enter for the default in brackets)")
        print("  profiles: " + ", ".join(PROFILES) + ", all")
        opt.profile = opt.profile or ask("profile", str, "all")
        opt.dt = opt.dt or ask("throat diameter D_t (m)", float, 1.2)
        opt.de = opt.de or ask("exit diameter   D_e (m)", float, 2.5)
        opt.L = opt.L or ask("divergent length L (m)", float, 2.5)
        if any(p in opt.profile for p in ("bell", "ellipse", "cubic", "all")):
            opt.theta_e = ask("exit wall angle theta_e (deg)", float, opt.theta_e)
        if any(p in opt.profile for p in ("bell", "all")):
            opt.theta_n = ask("bell initial angle theta_n (deg)", float, opt.theta_n)

    if not (0 < opt.dt < opt.de):
        ap.error("need 0 < D_t < D_e")
    if opt.L <= 0:
        ap.error("L must be positive")
    names = PROFILES if opt.profile.strip().lower() == "all" else \
        [p.strip().lower() for p in opt.profile.split(",")]
    for n in names:
        if n not in PROFILES:
            ap.error(f"unknown profile '{n}'. Choose from: {', '.join(PROFILES)}, all")

    r_t, r_e = opt.dt / 2.0, opt.de / 2.0
    geo = "planar" if opt.planar else "axisymmetric"
    AR = (r_e / r_t) if opt.planar else (r_e / r_t) ** 2
    print("=" * 78)
    print(" NOZZLE PROFILE CHECK  (1-D validity + Mach-wave shock initiation)")
    print("=" * 78)
    print(f" D_t = {opt.dt:g} m   D_e = {opt.de:g} m   L = {opt.L:g} m   {geo}   gamma = {opt.gamma:g}")
    print(f" A_e/A_t = {AR:.4f}   design exit Mach (1-D) = {mach_from_area(AR, opt.gamma)[0]:.4f}"
          f"   nu_e = {pm_nu_deg(mach_from_area(AR, opt.gamma)[0], opt.gamma):.3f} deg")

    results = []
    for n in names:
        try:
            res = analyse(n, r_t, r_e, opt.L, opt)
        except ValueError as e:
            print("-" * 78)
            print(f" PROFILE: {n}  -> skipped: {e}")
            continue
        results.append(res)
        report(res, opt)
    if not results:
        sys.exit(1)
    if len(results) > 1:
        summary_table(results, opt)

    stem = opt.out or (f"nozzle_{'all' if len(names) > 1 else names[0]}_"
                       f"Dt{opt.dt:g}_De{opt.de:g}_L{opt.L:g}")
    write_csv(results, stem + ".csv", opt)
    print(f" station data -> {stem}.csv")
    if not opt.no_plot:
        title = (f"{', '.join(r['name'] for r in results)}  |  D_t={opt.dt:g} m, D_e={opt.de:g} m, "
                 f"L={opt.L:g} m, {geo}, γ={opt.gamma:g}")
        plot(results, stem + ".png", opt, title)
        print(f" figure       -> {stem}.png")


if __name__ == "__main__":
    main()
