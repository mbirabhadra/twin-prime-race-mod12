"""Structure of the single-coincidence remainder R(H) = T4 - M.

Exact rewriting (no approximation): G(6k+2) G(6k-2) = sum_{e | 9k^2-1} gam(e), gam(p) = 1/(p-4), e squarefree, (e,6)=1
(the split e = e1 e2 with e1 | 3k+1, e2 | 3k-1 is forced). Hence
  R(H) = C4 sum_d g(d) rho_d(H),   rho_d(H) = sum_{e>1} gam(e) sum_{k: d|k, e | 9k^2-1} (-1)^k (1 - 6k/H).
Pairing lemma: the roots of {k = 0 mod d, 9k^2 = 1 mod e} mod m = de are closed under k -> -k, never 0,
and the least residues of R and m - R have opposite parity (m odd). So in every class pair the leading
(-1)^R * (1/2) terms cancel exactly, leaving O(min(1, m/H)) per pair.
This script measures rho_d(H) for d up to D, its convergence in d, and the parity/position statistic per pair.
Writes data/remainder_structure.json and .npz."""
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
GG1 = np.expm1(logG)  # G(6k+2)G(6k-2) - 1
del logG
print("GG built", round(time.time() - t0, 1), flush=True)

# squarefree d coprime to 6 with g(d)
D = 20000
small_p = P_[P_ <= D]
gd = np.ones(D + 1)
ok = np.ones(D + 1, dtype=bool)
ok[0] = False
for p in small_p:
    if p in (2, 3):
        ok[p::p] = False
        continue
    gd[p::p] *= 2.0 / (p - 4)
    ok[p * p :: p * p] = False
ds = np.nonzero(ok)[0]
Hs = [1e5, 1e6, 1e7, 1e8]
rho = {H: np.zeros(len(ds)) for H in Hs}
for j, d in enumerate(ds):
    i = np.arange(1, K // d + 1)
    k = d * i
    s = np.where(i % 2 == 0, 1.0, -1.0) * GG1[k]
    for H in Hs:
        m = 6 * k < H
        rho[H][j] = C4 * np.sum(s[m] * (1 - 6 * k[m] / H))
print("rho done", round(time.time() - t0, 1), flush=True)

coef = json.load(open("data/twin_coefficient.json"))
Rfull = {f"{H:.0e}": None for H in Hs}
zc = np.load("data/twin_coefficient.npz")
for H in Hs:
    Rfull[f"{H:.0e}"] = float(np.interp(np.log(H), np.log(zc["H"]), zc["R"]))
res = {"C4": C4, "D": D, "R_full": Rfull}
gw = gd[ds]
for H in Hs:
    contrib = gw * rho[H]
    cum = np.cumsum(contrib)
    res[f"H={H:.0e}"] = {
        "rho_1": float(rho[H][0]),
        "partial_sum_d<=10": float(cum[ds <= 10][-1]),
        "partial_sum_d<=100": float(cum[ds <= 100][-1]),
        "partial_sum_d<=1000": float(cum[ds <= 1000][-1]),
        "partial_sum_d<=20000": float(cum[-1]),
        # mean of rho_d over dyadic d-ranges, weighted and unweighted
        "mean_rho_by_range": {f"{a}-{b}": float(np.mean(rho[H][(ds >= a) & (ds < b)]))
                              for a, b in ((5, 50), (50, 500), (500, 5000), (5000, 20000))},
        "rms_rho_by_range": {f"{a}-{b}": float(np.sqrt(np.mean(rho[H][(ds >= a) & (ds < b)] ** 2)))
                             for a, b in ((5, 50), (50, 500), (500, 5000), (5000, 20000))},
    }

# pairing lemma check and the per-pair statistic sigma(d, e) = sum over pairs (-1)^R (1 - 2R/m), R < m/2
def roots(d, e):
    m = d * e
    k = np.arange(0, m, d)
    k = k[(9 * k * k - 1) % e == 0]
    return m, k


chk = 0
viol = 0
sig_rows = []
es = [e for e in range(5, 4000) if e % 2 and e % 3 and all(e % (q * q) for q in range(5, 64, 2))]  # squarefree, coprime to 6
for d in (1, 5, 7, 11, 13, 35, 77, 143, 1001):
    sig = []
    for e in es:
        if np.gcd(d, e) != 1:
            continue
        m, R = roots(d, e)
        if len(R) == 0:
            continue
        S = set(R.tolist())
        chk += 1
        if any(((m - r) % m) not in S or r == 0 for r in S):
            viol += 1
        Rl = R[R < m / 2]
        sig.append(float(np.sum(np.where(Rl % 2 == 0, 1.0, -1.0) * (1 - 2 * Rl / m))))
    sig = np.array(sig)
    sig_rows.append(dict(d=d, n=len(sig), mean_sigma=float(sig.mean()), mean_abs=float(np.abs(sig).mean())))
res["pairing_lemma_checked"] = chk
res["pairing_lemma_violations"] = viol
res["sigma_by_d"] = sig_rows
json.dump(res, open("data/remainder_structure.json", "w"), indent=1)
np.savez("data/remainder_structure.npz", ds=ds, g=gw, **{f"rho_{int(np.log10(H))}": rho[H] for H in Hs})
print(json.dumps(res, indent=1))
print("done", round(time.time() - t0, 1))
