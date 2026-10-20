"""Figures 1-5 of the paper, as vector PDFs in ../paper/figures/ (run after the computation scripts)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join("..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)
INK, GREY, LIGHT = "#111111", "#7a7a7a", "#e4e4e4"
BLUE, ORANGE, GREEN = "#1f6fd1", "#e2622c", "#1a9a6c"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9.5, "axes.labelsize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.bbox": "tight", "figure.dpi": 150,
})


def panel_label(ax, s):
    ax.text(-0.12, 1.02, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def load(name):
    return json.load(open(os.path.join("data", name)))


# ---------- Figure 1: the race, the echo and the noise ----------
path = "data/twin_swing.npz" if os.path.exists("data/twin_swing.npz") else "data/twin_swing_slim.npz"
z = np.load(path)
ec = load("echo_constant.json")
x, D, E, sig = z["x"].astype(float), z["D"], z["E"], z["sig"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axes[0]
ax.semilogx(x, D / np.sqrt(x), color=BLUE, lw=0.7, label=r"$D(x)/\sqrt{x}$")
ax.semilogx(x, E / np.sqrt(x), color=ORANGE, lw=1.4, label=r"$E(x)/\sqrt{x}$")
ax.axhline(ec["C_BH"], color=INK, lw=0.8, ls=(0, (4, 3)), label=r"$C_{\mathrm{BH}}=3.383$")
ax.axhline(0, color=GREY, lw=0.6)
ax.set_xlim(1e4, 1e10)
ax.set_xlabel(r"$x$")
ax.legend(loc="upper left", ncol=1)
panel_label(ax, "a")
ax = axes[1]
ax.fill_between(x, -1, 1, color=LIGHT, lw=0, label=r"$\pm\sigma(x)$")
ax.semilogx(x, D / sig, color=BLUE, lw=0.7, label=r"$D(x)/\sigma(x)$")
ax.semilogx(x, E / sig, color=ORANGE, lw=1.4, label=r"$E(x)/\sigma(x)$")
L = np.log(x)
ax.semilogx(x, ec["C_BH"] / (np.sqrt(ec["S2"]) * L), color=INK, lw=0.8, ls=(0, (4, 3)),
            label=r"$C_{\mathrm{BH}}/(\sqrt{\mathfrak{S}_2}\,\log x)$")
ax.axhline(0, color=GREY, lw=0.6)
ax.set_xlim(1e4, 1e10)
ax.set_ylim(-2, 2.2)
ax.set_xlabel(r"$x$")
ax.legend(loc="upper left", ncol=2)
panel_label(ax, "b")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_race.pdf"))
plt.close(fig)

# ---------- Figure 2: window variances against Hardy-Littlewood ----------
r = load("twin_hl.json")
zh = np.load("data/twin_hl.npz")
S2 = r["S2"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axes[0]
th = np.linspace(0, 1, 200)
ax.plot(th, 1 - th ** 2 / 2, color=BLUE, lw=0.9, ls=(0, (4, 3)))
ax.plot(th, 1 - th, color=ORANGE, lw=0.9, ls=(0, (4, 3)))
ax.text(0.975, 0.555, r"$1-\theta^2/2$", color=BLUE, fontsize=8.5, ha="center")
ax.text(0.975, 0.075, r"$1-\theta$", color=ORANGE, fontsize=8.5, ha="center")
for (lo, hi), mk, filled in (((1e9, 1e10), "o", True), ((1e8, 1e9), "s", False)):
    rows = [p for p in r["predictions"] if p["band"] == [lo, hi] and p["windows"] >= 90]
    Lm = np.log(np.sqrt(lo * hi))
    t = np.array([np.log(p["H"]) / Lm for p in rows])
    for kind, col in (("twins", BLUE), ("primes", ORANGE)):
        hl = np.array([p[f"{kind}_HL"] for p in rows])
        me = np.array([p[f"{kind}_measured"] for p in rows])
        se = me * np.sqrt(2 / np.array([p["windows"] for p in rows]))
        ax.plot(t, hl, "-", color=col, lw=1.2, alpha=1 if filled else 0.5)
        ax.errorbar(t, me, yerr=se, fmt=mk, color=col, ms=3.5, mfc=col if filled else "white", capsize=1.5, lw=0.8)
ax.plot([], [], "-", color=GREY, lw=1.2, label="Hardy–Littlewood prediction")
ax.plot([], [], "o", color=GREY, ms=3.5, label=r"measured, $x\in[10^9,10^{10}]$")
ax.plot([], [], "s", color=GREY, mfc="white", ms=3.5, label=r"measured, $x\in[10^8,10^9]$")
ax.axhline(1, color=GREY, lw=0.6)
ax.set_xlim(0.3, 1.0)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"$\theta=\log H/\log x$")
ax.set_ylabel(r"$\mathcal{R}(H;x)$")
ax.legend(loc="lower left")
panel_label(ax, "a")
ax = axes[1]
lH = np.log(zh["H"])
ax.plot(lH, zh["T4"], color=BLUE, lw=1.5, label=r"$T_4(H)$")
ax.plot(lH, -S2 / 4 * lH ** 2, color=BLUE, lw=0.8, ls=(0, (4, 3)), label=r"$-\frac{\mathfrak{S}_2}{4}\log^2 H$")
ax.plot(lH, 20 * zh["T2"], color=ORANGE, lw=1.5, label=r"$20\,T_2(H)$")
ax.plot(lH, -10 * lH, color=ORANGE, lw=0.8, ls=(0, (4, 3)), label=r"$-10\log H$")
ax.set_xlim(lH.min(), lH.max())
ax.set_xlabel(r"$\log H$")
ax.legend(loc="lower left")
panel_label(ax, "b")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_rigidity.pdf"))
plt.close(fig)

# ---------- Figure 3: main term and remainder ----------
c = load("twin_coefficient.json")
zc = np.load("data/twin_coefficient.npz")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axes[0]
lH = np.log(zc["H"])
ax.plot(lH, zc["T4"], color=INK, lw=1.6, label=r"$T_4(H)$")
ax.plot(lH, zc["M"], color=BLUE, lw=1.2, label=r"$M(H)$ (double coincidences)")
ax.plot(lH, zc["R"], color=ORANGE, lw=1.2, label=r"$R(H)$ (single coincidences)")
ax.plot(lH, -c["S2"] / 4 * lH ** 2, color=BLUE, lw=0.8, ls=(0, (4, 3)), label=r"$-\frac{\mathfrak{S}_2}{4}\log^2H$")
ax.axhline(0, color=GREY, lw=0.6)
ax.set_xlim(lH.min(), lH.max())
ax.set_xlabel(r"$\log H$")
ax.legend(loc="lower left")
panel_label(ax, "a")
ax = axes[1]
ly = np.log(zc["ys"].astype(float))
P = c["P"]
ax.plot(ly, zc["Gsum"], color=BLUE, lw=1.5, label=r"$\sum_{d\leq y} g(d)$")
ax.plot(ly, P / 2 * ly ** 2 + 0.9603 * ly, color=INK, lw=0.8, ls=(0, (4, 3)), label=r"$\frac{P}{2}\log^2y+A_1\log y$")
ax.plot(ly, P / 2 * ly ** 2, color=GREY, lw=0.8, ls=(0, (1, 2)), label=r"$\frac{P}{2}\log^2 y$")
ax.set_xlim(ly.min(), ly.max())
ax.set_xlabel(r"$\log y$")
ax.legend(loc="upper left")
panel_label(ax, "b")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_main_term.pdf"))
plt.close(fig)

# ---------- Figure 4: the boundary layer ----------
zb = np.load("data/boundary_layer.npz")
rb = load("boundary_layer.json")
bm = load("bl_model.json")
zm = np.load("data/bl_model.npz")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axes[0]
for a, col, mk in ((6, GREY, "s"), (7, ORANGE, "^"), (8, BLUE, "o")):
    N, w = zb[f"N_{a}"], zb[f"wmean_rho_{a}"]
    g = ~np.isnan(w)
    ax.semilogx(N[g], w[g], mk + "-", color=col, ms=3, lw=0.8, label=rf"$H=10^{a}$")
ax.semilogx(zm["Nm"], zm["Phi_cond"], color=INK, lw=0.9, ls=(0, (4, 3)), label=r"model $C_4A(N)\,\mathrm{Tail}(N)$")
ax.axhline(0, color=GREY, lw=0.6)
ax.set_xlim(1, 1e3)
ax.set_ylim(-0.85, 0.15)
ax.set_xlabel(r"$N=H/6d$")
ax.set_ylabel(r"$\Phi(N)$")
ax.legend(loc="lower right")
panel_label(ax, "a")
ax = axes[1]
Hh = zc["H"]
ax.plot(np.log(Hh), zc["R"], color=INK, lw=1.5, label=r"$R(H)$")
Nb = zb["Nbins"]
dl = np.diff(np.log(Nb))
Nc = zb["N_8"]
Phi = np.nan_to_num(np.nanmean([zb["wmean_rho_7"], zb["wmean_rho_8"]], axis=0))
rec = [rb["H=1e+08"]["rho_1"] + np.sum((Phi * (P * np.log(H / (6 * Nc)) + 0.9603) * dl)[Nc < H / 6]) for H in Hh]
ax.plot(np.log(Hh), rec, color=BLUE, lw=1.0, ls=(0, (4, 3)), label=r"rebuilt from $\Phi$")
ax.axhline(rb["H=1e+08"]["rho_1"], color=GREEN, lw=0.8, ls=(0, (1, 2)), label=r"$\rho_1$")
ax.set_xlim(np.log(1e3), np.log(Hh).max())
ax.set_ylim(-8, 3)
ax.set_xlabel(r"$\log H$")
ax.legend(loc="lower left")
panel_label(ax, "b")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_boundary_layer.pdf"))
plt.close(fig)

# ---------- Figure 5: the spectral route ----------
s = load("spectral_check.json")
nq = load("near_quarter.json")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axes[0]
for row, col in zip([q for q in s["rows"] if q["H"] >= 100], (GREEN, ORANGE, BLUE, INK)):
    H = row["H"]
    keys = [k for k in row if k.startswith("Q=")]
    Q = np.array([int(k[2:]) for k in keys])
    part = np.array([row[k]["partial"] for k in keys])
    ax.semilogx(Q / H, part - (row["2T4"] - 2.02), "o-", color=col, ms=3, lw=0.9, label=rf"$H={H}$")
ax.axvspan(0.25, 4, color=LIGHT, alpha=0.7, lw=0)
ax.axhline(0, color=GREY, lw=0.6)
ax.set_xlabel(r"cutoff $Q/H$")
ax.set_ylabel(r"partial sum $-\,(2T_4-c_{\pm2})$")
ax.legend(loc="lower right")
panel_label(ax, "a")
ax = axes[1]
labels = list(nq)
xx = np.arange(len(labels))
ax.bar(xx - 0.27, [nq[k]["mean_worst_case_ratio"] for k in labels], 0.25, color=LIGHT, edgecolor=GREY, label="worst case")
ax.bar(xx, [nq[k]["mean_ratio_omega>=3"] for k in labels], 0.25, color=ORANGE, label=r"$\omega(q)\geq3$")
ax.bar(xx + 0.27, [nq[k]["mean_ratio_to_average"] for k in labels], 0.25, color=BLUE, label="all $q$")
ax.axhline(1, color=INK, lw=0.8, ls=(0, (4, 3)))
ax.set_xticks(xx)
ax.set_xticklabels([r"$5$–$10^3$", r"$10^3$–$10^4$", r"$10^4$–$5{\cdot}10^4$", r"$5{\cdot}10^4$–$2{\cdot}10^5$"], fontsize=7.5)
ax.set_xlabel(r"range of $q$")
ax.set_ylabel("weight / average weight")
ax.legend(loc="upper left")
panel_label(ax, "b")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_spectral.pdf"))
plt.close(fig)
print("figures written to", OUT)
