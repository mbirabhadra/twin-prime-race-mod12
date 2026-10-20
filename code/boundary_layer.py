"""The remainder is a boundary layer: rho_d(H) depends (on average) only on N = H/(6d), the number of multiples
of d inside the window.  If rho_d ~ Phi(N) with Phi integrable against dN/N, then, because sum_{d in [y, 2y]} g(d)
~ P log y * log 2,
    R(H) = sum_d g(d) rho_d(H) ~ P log H * integral Phi(N) dN/N + O(1)   -> linear in log H, no log^2 term.
Measures rho_d for every squarefree d coprime to 6 up to H/6 at H = 1e6, 1e7, 1e8, bins by N, checks collapse,
and computes the predicted log H coefficient.  Writes data/boundary_layer.json and .npz."""
import json
import time

import numpy as np

t0 = time.time()
HMAX = 100_000_000
K = HMAX // 6
lim = 6 * K + 3
sv = np.ones(lim + 1, dtype=bool)
sv[:2] = False
for p in range(2, int(lim ** 0.5) + 1):
    if sv[p]:
        sv[p * p :: p] = False
P_ = np.nonzero(sv)[0]
del sv
p5 = P_[P_ >= 5].astype(np.float64)
C4 = 13.5 * np.exp(np.sum(np.log1p(-4 / p5) - 4 * np.log1p(-1 / p5)))
Pc = 0.25 * (4 / 9) * np.exp(np.sum(np.log1p(2 / (p5 - 4)) + 2 * np.log1p(-1 / p5)))
S2 = C4 * Pc

logG = np.zeros(K + 1)
for p in P_[(P_ >= 5) & (P_ <= 200_000)]:
    inv6 = (1 + 5 * p) // 6 if p % 6 == 1 else (1 + p) // 6
    f3 = np.log((p - 3) / (p - 4))
    for r in ((-2 * inv6) % p, (2 * inv6) % p):
        logG[r::p] += f3
big = P_[(P_ > 200_000) & (P_ <= 6 * K + 2)]
for chunk in np.array_split(big, 40):
    ch = chunk.astype(np.int64)
    f3 = np.log((ch - 3) / (ch - 4.0))
    inv6 = np.where(ch % 6 == 1, (1 + 5 * ch) // 6, (1 + ch) // 6)
    for base in ((-2 * inv6) % ch, (2 * inv6) % ch):
        n = np.maximum(0, (K - base) // ch + 1)
        n[(base > K) | (base == 0)] = 0
        tot = int(n.sum())
        if tot == 0:
            continue
        rep = np.repeat(np.arange(len(ch)), n)
        off = np.arange(tot) - np.repeat(np.cumsum(n) - n, n)
        np.add.at(logG, base[rep] + off * ch[rep], f3[rep])
GG1 = np.expm1(logG)
del logG

# g(d) on squarefree d coprime to 6, d <= K
gl = np.zeros(K + 1)
ok = np.ones(K + 1, dtype=bool)
ok[0] = False
for p in P_[P_ <= K]:
    if p in (2, 3):
        ok[p::p] = False
        continue
    gl[p::p] += np.log(2.0 / (p - 4))
    if p * p <= K:
        ok[p * p :: p * p] = False
ds = np.nonzero(ok)[0]
gd = np.exp(gl[ds])
del gl, ok
print("setup", len(ds), round(time.time() - t0, 1), flush=True)

out = {"C4": C4, "P": Pc, "S2": S2}
Nbins = np.logspace(0, 6, 49)
arrs = {}
for H in (1e6, 1e7, 1e8):
    Kh = int(H // 6)
    sel = ds <= Kh
    dd = ds[sel]
    rho = np.zeros(len(dd))
    # small d: loop over d; large d: loop over multiplier i, vectorised over d
    split = 3000
    js = np.nonzero(dd <= split)[0]
    for j in js:
        d = dd[j]
        i = np.arange(1, Kh // d + 1)
        k = d * i
        m = 6 * k < H
        rho[j] = np.sum(np.where(i[m] % 2 == 0, 1.0, -1.0) * (1 - 6 * k[m] / H) * GG1[k[m]])
    jl = np.nonzero(dd > split)[0]
    dl = dd[jl]
    for i in range(1, Kh // split + 2):
        k = dl * i
        m = 6 * k < H
        if not m.any():
            break
        rho[jl[m]] += (1.0 if i % 2 == 0 else -1.0) * (1 - 6 * k[m] / H) * GG1[k[m]]
    rho *= C4
    N = H / (6.0 * dd)
    contrib = gd[sel] * rho
    idx = np.digitize(N, Nbins) - 1
    mean_rho = np.array([rho[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(Nbins) - 1)])
    gs = gd[sel]
    wmean_rho = np.array([(gs[idx == b] * rho[idx == b]).sum() / gs[idx == b].sum() if np.any(idx == b) else np.nan
                          for b in range(len(Nbins) - 1)])
    gsum_bin = np.array([gs[idx == b].sum() for b in range(len(Nbins) - 1)])
    arrs[f"wmean_rho_{int(np.log10(H))}"] = wmean_rho
    arrs[f"gsum_{int(np.log10(H))}"] = gsum_bin
    sum_contrib = np.array([contrib[idx == b].sum() for b in range(len(Nbins) - 1)])
    Ncent = np.sqrt(Nbins[:-1] * Nbins[1:])
    good = ~np.isnan(mean_rho)
    # predicted coefficient: P * sum over bins mean_rho * dlnN   (excluding d = 1, which is the O(1) term)
    rho_no1 = rho.copy()
    rho_no1[dd == 1] = 0
    mean_no1 = np.array([rho_no1[idx == b].mean() if np.any(idx == b) else 0.0 for b in range(len(Nbins) - 1)])
    integ = float(np.nansum(mean_no1 * np.diff(np.log(Nbins))))
    out[f"H={H:.0e}"] = {"R_sum": float(contrib.sum()), "rho_1": float(rho[dd == 1][0]),
                         "integral_Phi_dlnN": integ, "P_times_integral": Pc * integ,
                         "share_from_N<100": float(contrib[(N < 100) & (dd > 1)].sum() / contrib[dd > 1].sum())}
    arrs[f"N_{int(np.log10(H))}"] = Ncent
    arrs[f"mean_rho_{int(np.log10(H))}"] = mean_rho
    arrs[f"sum_contrib_{int(np.log10(H))}"] = sum_contrib
    print(H, out[f"H={H:.0e}"], round(time.time() - t0, 1), flush=True)

# naive model: unresolved part of GG-1 for k = d*i, with first multiple odd: Phi_model(N) = C4 * A(N) * Tail(N)
# A(N) = sum_{i < N} (-1)^i (1 - i/N), Tail(N) = sum_{e > N} gam(e) 2^omega(e) / e   (expected unresolved mass)
E = 2_000_000
tl = np.zeros(E + 1)
ok = np.ones(E + 1, dtype=bool)
ok[0] = False
for p in P_[P_ <= E]:
    if p in (2, 3):
        ok[p::p] = False
        continue
    tl[p::p] += np.log(2.0 / (p - 4) / p)
    if p * p <= E:
        ok[p * p :: p * p] = False
w = np.where(ok, np.exp(tl), 0.0)
w[1] = 0.0
tail_from = np.cumsum(w[::-1])[::-1]  # sum_{e >= n}
lam = float(np.sum(w * np.log(np.maximum(np.arange(E + 1), 1))))
mu1 = float(w.sum())
Nm = np.logspace(0, 6, 400)


def A(Nv):
    n = np.floor(Nv - 1e-12).astype(int)
    outA = np.zeros_like(Nv)
    for t, (nn, NN) in enumerate(zip(n, Nv)):
        i = np.arange(1, nn + 1)
        outA[t] = np.sum(np.where(i % 2 == 0, 1.0, -1.0) * (1 - i / NN))
    return outA


Phi_model = C4 * A(Nm) * tail_from[np.minimum(np.ceil(Nm).astype(int), E)]
out["mu_minus_1"] = mu1
out["lambda_sum_log"] = lam
out["model_coefficient_P_int"] = float(Pc * np.trapezoid(Phi_model, np.log(Nm)))
arrs["N_model"] = Nm
arrs["Phi_model"] = Phi_model
json.dump(out, open("data/boundary_layer.json", "w"), indent=1)
np.savez("data/boundary_layer.npz", Nbins=Nbins, **arrs)
print(json.dumps(out, indent=1))
print("done", round(time.time() - t0, 1))
