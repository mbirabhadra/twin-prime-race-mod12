"""Branch step: what drives the swing in the twin-channel race?

Twin pairs (p, p+2), p >= 5, carry label s = +1 if p = 5 (mod 12) [channel (5,7)], -1 if p = 11 [channel (11,1)],
weight w = log p log(p+2).  D(x) = sum s w.  Echo term E(x) as before (pairs (q^2-2, q^2) land in (11,1)).

Hypothesis (shot noise): D(x) = E(x) + W(x), W behaving like a sum of independent +-w (variance sum w^2).
Control (rigid waves): the prime race R(x) = sum chi_-4(p) log p, whose fluctuations are fixed by zeros.

Tests to X = 1e10 (segmented sieve):
 1. Window variance ratio: for windows of length H at height ~x, Var(sum over window) / E(sum of squared weights).
    Shot noise -> ratio 1 at every H. Rigid waves -> ratio falls as H grows (primes: ~ log(x/H)/log x).
 2. Lag-1 correlation of consecutive twin labels.
 3. Global race statistics against coin-flip nulls (random signs on the real twins, plus E(x)).
Writes data/twin_swing.json and data/twin_swing.npz.
"""
import json
import time

import numpy as np

t0 = time.time()
X = 10_000_000_000
SEG = 100_000_000
HS = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
base_lim = int(X ** 0.5) + 10
bs = np.ones(base_lim + 1, dtype=bool)
bs[:2] = False
for p in range(2, int(base_lim ** 0.5) + 1):
    if bs[p]:
        bs[p * p :: p] = False
base = np.nonzero(bs)[0]

tw_p, tw_s = [], []
win = {H: {"tw": [], "tw2": [], "pr": [], "pr2": [], "start": []} for H in HS}
echo_q = []
lo = 0
while lo < X:
    hi = min(X, lo + SEG)
    seg = np.ones(hi - lo + 2, dtype=bool)
    if lo == 0:
        seg[:2] = False
    for p in base:
        if p * p >= hi + 2:
            break
        st = max(p * p, ((lo + p - 1) // p) * p)
        seg[st - lo :: p] = False
    pr = np.nonzero(seg[: hi - lo])[0] + lo
    pr = pr[pr >= 5]
    qlo, qhi = int(np.ceil(np.sqrt(lo + 2))), int(np.floor(np.sqrt(hi + 1)))
    for q in range(max(qlo, 5), qhi + 1):
        if q * q - 2 < hi and bs[q] and seg[q * q - 2 - lo]:
            echo_q.append(q)
    r = pr % 12
    tw = seg[pr - lo + 2] & ((r == 5) | (r == 11))
    tp = pr[tw]
    ts = np.where(tp % 12 == 5, 1, -1).astype(np.int8)
    tw_p.append(tp)
    tw_s.append(ts)
    lp = np.log(pr.astype(np.float64))
    chi = np.where(pr % 4 == 1, 1.0, -1.0)
    ww = np.log(tp.astype(np.float64)) * np.log(tp + 2.0)
    for H in HS:
        if H > SEG:
            continue
        nb = (hi - lo) // H
        bp = (pr - lo) // H
        bt = (tp - lo) // H
        win[H]["pr"].append(np.bincount(bp, weights=chi * lp, minlength=nb)[:nb])
        win[H]["pr2"].append(np.bincount(bp, weights=lp * lp, minlength=nb)[:nb])
        win[H]["tw"].append(np.bincount(bt, weights=ts * ww, minlength=nb)[:nb])
        win[H]["tw2"].append(np.bincount(bt, weights=ww * ww, minlength=nb)[:nb])
        win[H]["start"].append(lo + H * np.arange(nb))
    if (lo // SEG) % 10 == 0:
        print("segment", lo, round(time.time() - t0, 1), flush=True)
    lo = hi

tp = np.concatenate(tw_p)
ts = np.concatenate(tw_s)
w = np.log(tp.astype(np.float64)) * np.log(tp + 2.0)
res = {"X": X, "twins": int(len(tp))}
print("twins", len(tp), round(time.time() - t0, 1), flush=True)

# windows of 1e8 built from 1e7 windows
for H in HS:
    for k in list(win[H]):
        win[H][k] = np.concatenate(win[H][k]) if win[H][k] else np.zeros(0)
big = 100_000_000
w7 = win[10_000_000]
nbig = len(w7["tw"]) // 10
win[big] = {k: (w7[k][: nbig * 10].reshape(nbig, 10).sum(1) if k != "start" else w7[k][: nbig * 10: 10]) for k in w7}

# 1. variance ratios by height band and window length
ratios = []
for H in HS:
    W = win[H]
    for xlo, xhi in ((1e8, 1e9), (1e9, 1e10)):
        m = (W["start"] >= xlo) & (W["start"] < xhi)
        if m.sum() < 8:
            continue
        for kind, a, b in (("twins", "tw", "tw2"), ("primes", "pr", "pr2")):
            vals = W[a][m]
            # remove the slow mean drift (echo / bias) by subtracting the band mean
            v = np.mean((vals - vals.mean()) ** 2) / np.mean(W[b][m])
            ratios.append(dict(kind=kind, H=H, band=[xlo, xhi], windows=int(m.sum()), ratio=float(v)))
res["variance_ratios"] = ratios
print(json.dumps(ratios, indent=0)[:3000], flush=True)

# 2. lag-1 correlation of consecutive twin labels
s = ts.astype(np.float64)
res["label_lag1_corr"] = float(np.corrcoef(s[:-1], s[1:])[0, 1])
res["label_mean"] = float(s.mean())

# 3. global race against coin-flip nulls
eq = np.array(echo_q, dtype=np.float64)
ecum = np.cumsum(np.log(eq * eq - 2) * np.log(eq))
xs = np.unique(np.logspace(4, np.log10(X), 3000).astype(np.int64))
idx = np.searchsorted(tp, xs, side="right") - 1
Dw = np.cumsum(s * w)
D = np.where(idx >= 0, Dw[np.maximum(idx, 0)], 0.0)
eidx = np.searchsorted(eq * eq - 2, xs, side="right") - 1
E = np.where(eidx >= 0, ecum[np.maximum(eidx, 0)], 0.0)
sig = np.sqrt(np.where(idx >= 0, np.cumsum(w * w)[np.maximum(idx, 0)], 1.0))
sq = np.sqrt(xs.astype(np.float64))


def stats(Dc):
    return dict(share_pos=float(np.mean(Dc > 0)),
                mean_over_sqrtx={f"1e{a}-1e{a+1}": float(np.mean((Dc / sq)[(xs >= 10 ** a) & (xs < 10 ** (a + 1))])) for a in range(4, 10)},
                rms_over_sigma=float(np.sqrt(np.mean(((Dc - E) / sig) ** 2))))


res["real"] = stats(D)
rng = np.random.default_rng(7)
null = []
NN = 200
for k in range(NN):
    eps = rng.choice(np.array([-1.0, 1.0]), size=len(w))
    Dn = np.cumsum(eps * w)
    Dn = np.where(idx >= 0, Dn[np.maximum(idx, 0)], 0.0) + E
    null.append(stats(Dn))
res["null_share_pos_quantiles"] = [float(np.quantile([n["share_pos"] for n in null], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
res["null_rms_over_sigma_quantiles"] = [float(np.quantile([n["rms_over_sigma"] for n in null], q)) for q in (0.05, 0.5, 0.95)]
res["echo_over_sigma_at"] = {f"1e{a}": float(np.interp(np.log(10.0 ** a), np.log(xs), E / sig)) for a in (5, 6, 7, 8, 9, 10)}
res["sigma_over_sqrtx_logx_at_1e10"] = float(sig[-1] / (sq[-1] * np.log(xs[-1])))
json.dump(res, open("data/twin_swing.json", "w"), indent=1)
np.savez("data/twin_swing.npz", x=xs, D=D, E=E, sig=sig,
         null_example=np.array([0.0]),
         **{f"win_{H}_{k}": win[H][k] for H in HS + [big] for k in ("tw", "tw2", "pr", "pr2", "start")})
print(json.dumps({k: v for k, v in res.items() if k != "variance_ratios"}, indent=1))
print("done", round(time.time() - t0, 1))
