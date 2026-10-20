"""Key lemma test: do the Fourier atoms closest to +-1/4 (alpha = a/q -+ 1/4 within ~j/(4q), j <= 4) carry the
AVERAGE weight mass_q / (2 phi(q))?  A worst-case bound would allow up to prod 4/(p-2)^2 / prod 2/((p-2)) ~ 2^omega(q)
times more.  Writes data/near_quarter.json."""
import json
import numpy as np
QMAX = 200000
spf = np.zeros(QMAX + 1, dtype=np.int64)
for p in range(2, QMAX + 1):
    if spf[p] == 0:
        spf[p::p][spf[p::p] == 0] = p
rows = []
for q in range(5, QMAX + 1, 2):
    n, fac, ok = q, [], True
    while n > 1:
        p = int(spf[n]); n //= p
        if n % p == 0:
            ok = False; break
        fac.append(p)
    if not ok:
        continue
    phi = int(np.prod([p - 1 for p in fac]))
    avg = np.prod([2.0 / (p - 2) for p in fac]) / phi   # mass per atom, relative to base
    mx = np.prod([4.0 / (p - 2) ** 2 for p in fac])
    for c in (q / 4, 3 * q / 4):
        for a in range(int(c) - 1, int(c) + 3):
            if a <= 0 or a >= q or np.gcd(a, q) != 1:
                continue
            j = abs(4 * a - (q if c < q / 2 else 3 * q))
            if j > 4:
                continue
            w = 1.0
            for p in fac:
                b = (a * pow(q // p, -1, p)) % p
                w *= 4 * np.cos(2 * np.pi * b / p) ** 2 / (p - 2) ** 2
            rows.append((q, j, w / avg, mx / avg, len(fac)))
r = np.array(rows)
out = {}
for lo, hi in ((5, 1e3), (1e3, 1e4), (1e4, 5e4), (5e4, 2e5)):
    m = (r[:, 0] >= lo) & (r[:, 0] < hi)
    out[f"{int(lo)}-{int(hi)}"] = {"atoms": int(m.sum()), "mean_ratio_to_average": float(r[m, 2].mean()),
                                    "mean_worst_case_ratio": float(r[m, 3].mean()),
                                    "mean_ratio_omega>=3": float(r[m & (r[:, 4] >= 3), 2].mean())}
json.dump(out, open("data/near_quarter.json", "w"), indent=1)
print(json.dumps(out, indent=1))
