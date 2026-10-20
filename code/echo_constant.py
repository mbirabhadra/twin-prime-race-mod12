"""Constants of the race: the echo E(x) ~ C_BH sqrt(x) and the coin-flip spread sigma(x).

C_BH is the Bateman-Horn constant of the pair (q, q^2 - 2):
    C_BH = 2 prod_{p > 2} (1 - (2 + (2/p))/p) / (1 - 1/p)^2,    (2/p) the Legendre symbol,
so that E(x) = sum_{5 <= q, q^2 - 2 <= x, q and q^2 - 2 prime} log(q^2 - 2) log q ~ C_BH sqrt(x).
sigma(x)^2 = sum over twin pairs p <= x of (log p log(p+2))^2 ~ S2 x (log^2 x - 2 log x + 2)  (Hardy-Littlewood).
Reads data/twin_swing.npz (or data/twin_swing_slim.npz); writes data/echo_constant.json."""
import json
import os

import numpy as np

N = 10 ** 7
s = np.ones(N, dtype=bool)
s[:2] = False
for p in range(2, int(N ** 0.5) + 1):
    if s[p]:
        s[p * p :: p] = False
po = np.nonzero(s)[0]
po = po[po > 2]
leg = np.where((po % 8 == 1) | (po % 8 == 7), 1, -1)
C_BH = 2 * np.exp(np.sum(np.log1p(-(2 + leg) / po) - 2 * np.log1p(-1 / po)))
S2 = 2 * np.exp(np.sum(np.log1p(-1 / (po - 1.0) ** 2)))
path = "data/twin_swing.npz" if os.path.exists("data/twin_swing.npz") else "data/twin_swing_slim.npz"
z = np.load(path)
x, E, sig = z["x"].astype(float), z["E"], z["sig"]
out = {"C_BH": float(C_BH), "S2": float(S2), "rows": []}
for a in (5, 6, 7, 8, 9, 10):
    i = np.searchsorted(x, 10.0 ** a, side="right") - 1
    L = np.log(x[i])
    out["rows"].append({"x": float(x[i]), "E_over_sqrtx": float(E[i] / np.sqrt(x[i])),
                        "sigma_over_sqrtx_logx": float(sig[i] / (np.sqrt(x[i]) * L)),
                        "sigma_predicted": float(np.sqrt(S2 * (1 - 2 / L + 2 / L ** 2))),
                        "echo_over_sigma": float(E[i] / sig[i]),
                        "echo_over_sigma_predicted_leading": float(C_BH / (np.sqrt(S2) * L))})
json.dump(out, open("data/echo_constant.json", "w"), indent=1)
print(json.dumps(out, indent=1))
