"""Why T4(H) = -(S2/4) log^2 H + O(log H): split the 4-point singular series into its two kinds of coincidence.

For h = 6k, S4(6k) = C4 * F(k) * G(6k+2) * G(6k-2),  C4 = (27/2) prod_{p>=5} p^3(p-4)/(p-1)^4,
  F(k)   = prod_{p | k, p>=5} (p-2)/(p-4)  = sum_{d | k} g(d),   g(p) = 2/(p-4)  ~ 2/p   (p | h: DOUBLE coincidence,
                                                                    n = n+h and n+2 = n+h+2 mod p at once)
  G(n)   = prod_{p | n, p>=5} (p-3)/(p-4) = sum_{e | n} 1/(p-4)...      ~ 1/p   (p | h+-2: single coincidence)
Main term (e1 = e2 = 1):  M(H) = C4 sum_k (-1)^k (1-6k/H) F(k) = C4 sum_d g(d) sum_i (-1)^{di} (1-6di/H)
                                = -(1/2) C4 sum_{d<H/6} g(d) + O(log H)          (d odd: (-1)^{di} = (-1)^i)
and sum_{d<=y} g(d) = (P/2) log^2 y + O(log y) since g(p) ~ 2/p (Dirichlet series ~ zeta(s+1)^2 Q(s)),
  P = Q(0) = (1/2)^2 (2/3)^2 prod_{p>=5} (1 + 2/(p-4)) (1 - 1/p)^2.
Identity: C4 * P = (3/2) prod_{p>=5} (1 - 1/(p-1)^2) = S2.  Hence M(H) = -(S2/4) log^2 H + O(log H).
The remainder R = T4 - M carries the single coincidences (p | h+-2); their residue classes k = r mod e have
parities that alternate with chi_-3(e), so they cancel to O(log H) (checked here numerically).
Pair analogue (exact, no remainder): S(h) = S2 sum_{d | h, d odd} mu^2(d)/prod(p-2), g2(p) = 1/(p-2) ~ 1/p,
  T2(H) = -(1/2) S2 sum_{d<H/2} g2(d) + O(1) = -(1/2) log H + O(1), since S2 * P2 = 1 with P2 = 1/S2.
Writes data/twin_coefficient.json and .npz."""
import json
import time

import numpy as np

t0 = time.time()
HMAX = 100_000_000
K = HMAX // 6
lim = HMAX + 3
sv = np.ones(lim + 1, dtype=bool)
sv[:2] = False
for p in range(2, int(lim ** 0.5) + 1):
    if sv[p]:
        sv[p * p :: p] = False
P_ = np.nonzero(sv)[0]
del sv
po = P_[P_ > 2].astype(np.float64)
S2 = 2 * np.exp(np.sum(np.log1p(-1 / (po - 1) ** 2)))
p5 = P_[P_ >= 5].astype(np.float64)
C4 = 13.5 * np.exp(np.sum(np.log1p(-4 / p5) - 4 * np.log1p(-1 / p5)))
Pc = 0.25 * (4 / 9) * np.exp(np.sum(np.log1p(2 / (p5 - 4)) + 2 * np.log1p(-1 / p5)))
res = {"S2": S2, "C4": C4, "P": Pc, "C4_times_P": C4 * Pc, "identity_error": C4 * Pc - S2}
print(res, flush=True)

logF = np.zeros(K + 1)
logG = np.zeros(K + 1)
small = P_[(P_ >= 5) & (P_ <= 200_000)]
for p in small:
    logF[p::p] += np.log((p - 2) / (p - 4))
    inv6 = (1 + 5 * p) // 6 if p % 6 == 1 else (1 + p) // 6
    f3 = np.log((p - 3) / (p - 4))
    for r in ((-2 * inv6) % p, (2 * inv6) % p):
        logG[r::p] += f3
big = P_[(P_ > 200_000) & (P_ <= 6 * K + 2)]
for chunk in np.array_split(big, 40):
    ch = chunk.astype(np.int64)
    f2 = np.log((ch - 2) / (ch - 4.0))
    f3 = np.log((ch - 3) / (ch - 4.0))
    inv6 = np.where(ch % 6 == 1, (1 + 5 * ch) // 6, (1 + ch) // 6)
    for base, f, arr in ((ch, f2, logF), ((-2 * inv6) % ch, f3, logG), ((2 * inv6) % ch, f3, logG)):
        n = np.maximum(0, (K - base) // ch + 1)
        n[(base > K) | (base == 0)] = 0
        tot = int(n.sum())
        if tot == 0:
            continue
        rep = np.repeat(np.arange(len(ch)), n)
        off = np.arange(tot) - np.repeat(np.cumsum(n) - n, n)
        np.add.at(arr, base[rep] + off * ch[rep], f[rep])
F = np.exp(logF)
GG = np.exp(logG)
del logF, logG
print("series built", round(time.time() - t0, 1), flush=True)

sg = np.where(np.arange(K + 1) % 2 == 0, 1.0, -1.0)
kk = np.arange(K + 1, dtype=np.float64)
Hs = np.unique(np.logspace(2, 8, 300).astype(np.int64)).astype(np.float64)


def tsum(a):
    a = a.copy()
    a[0] = 0
    c0, c1 = np.cumsum(a), np.cumsum(a * 6 * kk)
    m = np.minimum(K, np.ceil(Hs / 6).astype(np.int64) - 1)
    return c0[m] - c1[m] / Hs


T4 = C4 * tsum(sg * F * GG)
M = C4 * tsum(sg * F)
R = T4 - M
# the divisor sum of g: sum_{d<=y, (d,6)=1, squarefree} g(d), via g = F * mu-ish: sum_{d<=y} g(d) = sum_{n<=y} (F*mu)(n)...
# easier: g(d) = F(d) * prod_{p|d}(1 - (p-4)/(p-2)) ... compute directly with a sieve on d
y = K
gl = np.zeros(y + 1)
sq = np.ones(y + 1, dtype=bool)
sq[0] = False
for p in P_[P_ <= y]:
    if p in (2, 3):
        sq[p::p] = False
        continue
    gl[p::p] += np.log(2.0 / (p - 4))
    if p * p <= y:
        sq[p * p :: p * p] = False
g = np.where(sq, np.exp(gl), 0.0)
g[1] = 1.0
del gl, sq
Gsum = np.cumsum(g)
ys = np.unique(np.logspace(1, np.log10(y), 300).astype(np.int64))
lead_g = Pc / 2 * np.log(ys) ** 2
L = np.log(Hs)
lead = -S2 / 4 * L ** 2
sel = Hs >= 1e4
fitM = np.polyfit(L[sel], M[sel] - lead[sel], 1)
fitR = np.polyfit(L[sel], R[sel], 1)
fitRq = np.polyfit(L[sel], R[sel], 2)
fitG = np.polyfit(np.log(ys[ys >= 1e3]), Gsum[ys[ys >= 1e3]] - lead_g[ys >= 1e3], 1)
res.update({
    "M_minus_lead_linear_fit": list(map(float, fitM)),
    "R_linear_fit": list(map(float, fitR)),
    "R_quadratic_fit_leading_coeff": float(fitRq[0]),
    "sum_g_minus_lead_linear_fit": list(map(float, fitG)),
    "T4_at": {f"1e{a}": float(np.interp(a, np.log10(Hs), T4)) for a in range(3, 9)},
    "M_at": {f"1e{a}": float(np.interp(a, np.log10(Hs), M)) for a in range(3, 9)},
    "R_at": {f"1e{a}": float(np.interp(a, np.log10(Hs), R)) for a in range(3, 9)},
    "lead_at": {f"1e{a}": float(-S2 / 4 * (a * np.log(10)) ** 2) for a in range(3, 9)},
    "share_of_T4_from_main_at_1e8": float(M[-1] / T4[-1]),
})
# chi_-3 parity check for single coincidences: for primes e, residue r of k with 6k = -2 (mod e)
pr = P_[(P_ >= 5) & (P_ < 2000)]
inv6 = np.where(pr % 6 == 1, (1 + 5 * pr) // 6, (1 + pr) // 6)
rplus = (-2 * inv6) % pr
chi3 = np.where(pr % 3 == 1, 1, -1)
res["parity_rule_plus"] = bool(np.all(np.where(rplus % 2 == 0, 1, -1) == chi3))
rminus = (2 * inv6) % pr
res["parity_rule_minus"] = bool(np.all(np.where(rminus % 2 == 0, 1, -1) == -chi3))
json.dump(res, open("data/twin_coefficient.json", "w"), indent=1)
np.savez("data/twin_coefficient.npz", H=Hs, T4=T4, M=M, R=R, ys=ys, Gsum=Gsum[ys], lead_g=lead_g)
print(json.dumps(res, indent=1))
print("done", round(time.time() - t0, 1))
