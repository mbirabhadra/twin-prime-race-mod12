"""Spectral route to T4(H) = -(S2/4) log^2 H.

Process: Y(n) = chi_-4(n) prod_p nu_p(n), nu_p(n) = (1-1/p)^-2 1[p does not divide n(n+2)], p odd (2-adic part
absorbed in chi_-4 and the factor 4).  Then E[Y(n) Y(n+h)] = (-1)^{h/2} S({0,2,h,h+2}) for even h (0 for odd h).
Fourier atoms: alpha = +-1/4 + a/q, q odd squarefree, a coprime to q, with weight (each sign)
    W(a/q) = 4 (S2/2)^2 prod_{p|q} 4 cos^2(2 pi b_p / p) / (p-2)^2,   b_p = a * ((q/p)^-1 mod p) mod p.
Mass of denominator q (both signs): 2 S2^2 prod_{p|q} 2/(p-2);  M(Q) = sum_{q<=Q} mass_q ~ (S2/2) log^2 Q.
Wiener-Khinchin with the Fejer kernel F_H:  2 T4(H) = sum_alpha W(alpha) (F_H(alpha) - 1).
Claims to test:  (a) the identity (tail -> 0 as the atom cutoff Q grows);
                 (b) E_small(H) = sum_{q <= H/4} sum_a W F_H  grows only like log H;
                 (c) hence 2 T4 = -M(H/4) + O(log H) + (middle range).
Writes data/spectral_check.json."""
import json
import time

import numpy as np

t0 = time.time()
QMAX = 60000
HS = [30, 100, 300, 1000, 3000]
lim = 2 * QMAX
sv = np.ones(lim + 1, dtype=bool); sv[:2] = False
for p in range(2, int(lim ** 0.5) + 1):
    if sv[p]:
        sv[p * p :: p] = False
primes = np.nonzero(sv)[0]
po = primes[primes > 2].astype(float)
S2 = 2 * np.exp(np.sum(np.log1p(-1 / (po - 1) ** 2)))

# 4-point singular series for h = 6k up to max H (direct)
C4 = 13.5 * np.exp(np.sum(np.log1p(-4 / po[po >= 5]) - 4 * np.log1p(-1 / po[po >= 5])))
Kmax = max(HS) // 6 + 1
s4 = np.zeros(Kmax + 1)
for k in range(1, Kmax + 1):
    h = 6 * k
    v = C4
    for p in primes[(primes >= 5) & (primes <= h + 2)]:
        if h % p == 0:
            v *= (p - 2) / (p - 4)
        elif (h + 2) % p == 0 or (h - 2) % p == 0:
            v *= (p - 3) / (p - 4)
    s4[k] = v
T4 = {}
for H in HS:
    k = np.arange(1, Kmax + 1)
    m = 6 * k < H
    T4[H] = float(np.sum(np.where(k[m] % 2 == 0, 1.0, -1.0) * (1 - 6 * k[m] / H) * s4[k[m]]))
print("T4", T4, round(time.time() - t0, 1), flush=True)

# odd squarefree q <= QMAX with their prime factors
spf = np.zeros(QMAX + 1, dtype=np.int64)
for p in primes[primes <= QMAX]:
    sl = spf[p::p]
    sl[sl == 0] = p
qs = []
for q in range(1, QMAX + 1, 2):
    n, fac, ok = q, [], True
    while n > 1:
        p = int(spf[n]); n //= p
        if n % p == 0:
            ok = False; break
        fac.append(p)
    if ok:
        qs.append((q, fac))
print("q list", len(qs), round(time.time() - t0, 1), flush=True)


def fejer(t, H):
    s = np.sin(np.pi * t)
    out = np.empty_like(t)
    small = np.abs(s) < 1e-12
    out[~small] = (np.sin(np.pi * H * t[~small]) / s[~small]) ** 2 / H
    out[small] = H
    return out


base = 4 * (S2 / 2) ** 2
cum_small = {H: [] for H in HS}  # running sum over q of sum_a W F_H
cum_mass = []
qlist = []
runF = {H: 0.0 for H in HS}
runM = 0.0
checkpoints = set()
for q, fac in qs:
    if q == 1:
        w = np.array([base])
        alpha = np.array([0.0])
    else:
        a = np.arange(q)
        w = np.full(q, base)
        for p in fac:
            inv = pow(q // p, -1, p)
            b = (a * inv) % p
            w = w * (4 * np.cos(2 * np.pi * b / p) ** 2 / (p - 2) ** 2)
        keep = np.gcd(a, q) == 1
        w = w[keep]
        alpha = a[keep] / q
    runM += 2 * w.sum()
    for H in HS:
        runF[H] += float(np.sum(w * (fejer(alpha + 0.25, H) + fejer(alpha - 0.25, H))))
    qlist.append(q)
    cum_mass.append(runM)
    for H in HS:
        cum_small[H].append(runF[H])
qarr = np.array(qlist)
cum_mass = np.array(cum_mass)
res = {"S2": S2, "QMAX": QMAX, "T4": T4, "rows": []}
for H in HS:
    cs = np.array(cum_small[H])
    row = {"H": H, "2T4": 2 * T4[H]}
    for Q in (H // 4, H, 4 * H, 16 * H, QMAX):
        i = np.searchsorted(qarr, Q, side="right") - 1
        row[f"Q={Q}"] = {"sum_WF": float(cs[i]), "M": float(cum_mass[i]),
                         "partial": float(cs[i] - cum_mass[i]),
                         "tail_implied": float(2 * T4[H] - (cs[i] - cum_mass[i]))}
    i = np.searchsorted(qarr, H // 4, side="right") - 1
    row["E_small"] = float(cs[i])
    row["M(H/4)"] = float(cum_mass[i])
    row["leading_(S2/2)log^2(H/4)"] = float(S2 / 2 * np.log(H / 4) ** 2)
    res["rows"].append(row)
    print(json.dumps(row), flush=True)
# M(Q) against (S2/2) log^2 Q
res["M_vs_lead"] = {str(Q): [float(cum_mass[np.searchsorted(qarr, Q, side='right') - 1]), float(S2 / 2 * np.log(Q) ** 2)]
                    for Q in (100, 1000, 10000, QMAX)}
json.dump(res, open("data/spectral_check.json", "w"), indent=1)
print(json.dumps(res["M_vs_lead"]))
print("done", round(time.time() - t0, 1))
