"""Derive the window-variance ratio of the twin race (and the prime race) from Hardy-Littlewood.

Twin race in a window of length H at height x:  S = sum_{x<n<=x+H} chi_-4(n) Lambda(n) Lambda(n+2).
  Var S = sum w^2 (diagonal, ~ H S2 L^2)  +  sum_{h != 0} (H - |h|) chi(n)chi(n+h) E[Lambda(n)Lambda(n+2)Lambda(n+h)Lambda(n+h+2)].
For odd n and even h, chi_-4(n) chi_-4(n+h) = (-1)^{h/2}. The 4-point singular series S4(h) of {0, 2, h, h+2}
vanishes unless 6 | h (mod-3 obstruction), so only h = 6k contribute, with sign (-1)^k:
  R_twin(H) = 1 + 2 T4(H) / (S2 <L^2>),   T4(H) = sum_{0<6k<H} (1 - 6k/H) (-1)^k S4(6k).
Prime control (race sum chi_-4(p) log p):
  R_prime(H) = 1 + 2 T2(H) / <L>,          T2(H) = sum_{0<h<H, h even} (1 - h/H) (-1)^{h/2} S(h).
S(h) = 2 C2 prod_{p | h, p > 2} (p-1)/(p-2);  S4(6k) = (27/2) B4 prod_{p | k} (p-2)/(p-4) prod_{p | 6k +- 2} (p-3)/(p-4),
B4 = prod_{p >= 5} p^3 (p-4)/(p-1)^4  (so S4(6) = 4.1511..., the prime-quadruplet constant).
No fitted parameters. Writes data/twin_hl.json, data/twin_hl.npz."""
import json
import time

import numpy as np

t0 = time.time()
HMAX = 100_000_000
J = HMAX // 2          # h = 2j
K = HMAX // 6          # h = 6k
lim = HMAX + 3
sv = np.ones(lim + 1, dtype=bool)
sv[:2] = False
for p in range(2, int(lim ** 0.5) + 1):
    if sv[p]:
        sv[p * p :: p] = False
P = np.nonzero(sv)[0]
del sv
print("primes", len(P), round(time.time() - t0, 1), flush=True)

# twin constant and B4
po = P[P > 2].astype(np.float64)
S2 = 2 * np.exp(np.sum(np.log1p(-1 / (po - 1) ** 2)))
p5 = P[P >= 5].astype(np.float64)
B4 = np.exp(np.sum(np.log1p(-4 / p5) - 4 * np.log1p(-1 / p5)))
print("S2", S2, "S4(6)", 13.5 * B4, flush=True)

# 2-point singular series at h = 2j, j = 0..J
c2 = np.zeros(J + 1)
for p in P[(P > 2) & (P <= J)]:
    c2[p::p] += np.log((p - 1) / (p - 2))
Sh = S2 * np.exp(c2)
Sh[0] = 0
del c2
print("S(h) done", round(time.time() - t0, 1), flush=True)

# 4-point singular series at h = 6k, k = 0..K
c4 = np.zeros(K + 1)
small = P[(P >= 5) & (P <= 200_000)]
for p in small:
    c4[p::p] += np.log((p - 2) / (p - 4))
    inv6 = (1 + 5 * p) // 6 if p % 6 == 1 else (1 + p) // 6
    f3 = np.log((p - 3) / (p - 4))
    for r in ((-2 * inv6) % p, (2 * inv6) % p):
        c4[r::p] += f3
big = P[P > 200_000]
big = big[big <= 6 * K + 2]
for chunk in np.array_split(big, 40):
    ch = chunk.astype(np.int64)
    f2 = np.log((ch - 2) / (ch - 4.0))
    f3 = np.log((ch - 3) / (ch - 4.0))
    inv6 = np.where(ch % 6 == 1, (1 + 5 * ch) // 6, (1 + ch) // 6)
    starts = [(ch, f2)] if True else []
    for base, f in ((ch, f2), ((-2 * inv6) % ch, f3), ((2 * inv6) % ch, f3)):
        n = np.maximum(0, (K - base) // ch + 1)
        n[base > K] = 0
        n[base == 0] = 0
        tot = int(n.sum())
        if tot == 0:
            continue
        rep = np.repeat(np.arange(len(ch)), n)
        off = np.arange(tot) - np.repeat(np.cumsum(n) - n, n)
        idx = base[rep] + off * ch[rep]
        np.add.at(c4, idx, f[rep])
S4 = 13.5 * B4 * np.exp(c4)
S4[0] = 0
del c4
print("S4 done", round(time.time() - t0, 1), flush=True)

# validations
res = {"S2": S2, "S4_6_quadruplet": float(S4[1]), "S4_12": float(S4[2]),
       "mean_S_over_even_h": float(Sh[1:].mean()), "mean_S4_over_6k": float(S4[1:].mean()),
       "gallagher_targets": {"mean_S_even": 2.0, "mean_S4_6k": 6 * S2 ** 2}}


def tsum(a, step, Hs):
    """T(H) = sum_{0 < step*i < H} (1 - step*i/H) a_i for many H (a_0 ignored)."""
    i = np.arange(len(a), dtype=np.float64)
    c0 = np.cumsum(a)
    c1 = np.cumsum(a * i * step)
    out = []
    for H in Hs:
        m = int(min(len(a) - 1, np.ceil(H / step) - 1))
        out.append(c0[m] - c0[0] - c1[m] / H)
    return np.array(out)


sgn2 = np.where(np.arange(J + 1) % 2 == 0, 1.0, -1.0)
sgn4 = np.where(np.arange(K + 1) % 2 == 0, 1.0, -1.0)
Hs = np.unique(np.logspace(1.5, 8, 400).astype(np.int64)).astype(np.float64)
T2 = tsum(sgn2 * Sh, 2, Hs)
T4 = tsum(sgn4 * S4, 6, Hs)
# untwisted control: Montgomery-Soundararajan sum_{h<H}(1-h/H)(S(h)-1) over all h (S(odd)=0) ~ -(1/2) log H + c
Tms = tsum(Sh - 2.0, 2, Hs) - 0.5 * Hs * (1 - 1 / Hs)  # odd h contribute (0-1) each: sum_{odd h<H} (1-h/H)(-1) ~ -H/4
fit = lambda T: np.polyfit(np.log(Hs[Hs >= 1e4]), T[Hs >= 1e4], 1)
res["fit_T2_vs_logH"] = list(map(float, fit(T2)))
res["fit_T4_vs_logH"] = list(map(float, fit(T4)))
res["T2_at"] = {f"1e{a}": float(np.interp(a, np.log10(Hs), T2)) for a in range(2, 9)}
res["T4_at"] = {f"1e{a}": float(np.interp(a, np.log10(Hs), T4)) for a in range(2, 9)}

# predicted ratios in the two height bands (windows uniform in x)
pred = []
meas = json.load(open("data/twin_swing.json"))["variance_ratios"]
for lo, hi in ((1e8, 1e9), (1e9, 1e10)):
    xs = np.linspace(lo, hi, 20001)
    L1 = float(np.mean(np.log(xs)))
    L2 = float(np.mean(np.log(xs) * np.log(xs + 2)))
    for H in (1e4, 1e5, 1e6, 1e7, 1e8):
        t2 = float(np.interp(np.log10(H), np.log10(Hs), T2))
        t4 = float(np.interp(np.log10(H), np.log10(Hs), T4))
        rp = 1 + 2 * t2 / L1
        rt = 1 + 2 * t4 / (S2 * L2)
        mt = [m["ratio"] for m in meas if m["kind"] == "twins" and m["H"] == H and m["band"] == [lo, hi]]
        mp = [m["ratio"] for m in meas if m["kind"] == "primes" and m["H"] == H and m["band"] == [lo, hi]]
        nw = [m["windows"] for m in meas if m["kind"] == "twins" and m["H"] == H and m["band"] == [lo, hi]]
        pred.append(dict(band=[lo, hi], H=H, twins_HL=rt, twins_measured=mt[0] if mt else None,
                         primes_HL=rp, primes_MS=float(np.log(np.sqrt(lo * hi) / H) / np.log(np.sqrt(lo * hi))),
                         primes_measured=mp[0] if mp else None, windows=nw[0] if nw else None,
                         # sampling error of a variance estimate from n windows (Gaussian): ratio * sqrt(2/n)
                         twins_se=(mt[0] * np.sqrt(2 / nw[0])) if mt else None))
res["predictions"] = pred
for r in pred:
    print(r["band"], int(r["H"]), "twins HL %.3f meas %.3f (se %.3f) | primes HL %.3f MS %.3f meas %.3f" % (
        r["twins_HL"], r["twins_measured"], r["twins_se"], r["primes_HL"], r["primes_MS"], r["primes_measured"]))
json.dump(res, open("data/twin_hl.json", "w"), indent=1)
np.savez("data/twin_hl.npz", H=Hs, T2=T2, T4=T4, S4_first=S4[:2001], S_first=Sh[:2001])
print(json.dumps({k: v for k, v in res.items() if k != "predictions"}, indent=1))
print("done", round(time.time() - t0, 1))
