"""Semi-analytic model of the boundary layer Phi(N) (g-weighted) and the remainder's log H coefficient.
For d with N = H/(6d) multiples in the window, k = d i (i = 1..N), first multiple odd.  Classes e with period
d e > H/6 (e > N) are unresolved: their roots in range carry the parity of i, so they contribute their mean mass
times the alternating sum A(N) = sum_{i<N} (-1)^i (1 - i/N)  (about -1/2).  Resolved classes (e < N) cancel in
pairs (pairing lemma).  Conditioning on d | k under the g-weighting removes primes of d from e: each p divides a
g-weighted d with probability 2/(p-2), so the unresolved mass is Tail(N) = sum_{e > N} w(e), w(p) = 2/(p(p-2)).
  Phi_model(N) = C4 * A(N) * Tail(N),   kappa_model = P * integral Phi_model dlnN = S2 * integral A Tail dlnN.
Writes data/bl_model.json and .npz."""
import json
import numpy as np

r = json.load(open("data/boundary_layer.json"))
C4, P, S2 = r["C4"], r["P"], r["S2"]
E = 4_000_000
sv = np.ones(E + 1, dtype=bool); sv[:2] = False
for p in range(2, int(E ** 0.5) + 1):
    if sv[p]:
        sv[p * p :: p] = False
pr = np.nonzero(sv)[0]
def tail_array(wp):
    tl = np.zeros(E + 1); ok = np.ones(E + 1, dtype=bool); ok[0] = False
    for p in pr:
        if p in (2, 3):
            ok[p::p] = False; continue
        tl[p::p] += np.log(wp(p))
        if p * p <= E:
            ok[p * p :: p * p] = False
    w = np.where(ok, np.exp(tl), 0.0); w[1] = 0.0
    return w, np.cumsum(w[::-1])[::-1]
w_naive, T_naive = tail_array(lambda p: 2.0 / (p * (p - 4)))
w_cond, T_cond = tail_array(lambda p: 2.0 / (p * (p - 2)))
Nm = np.logspace(0, 6, 600)
def A(NN):
    n = int(np.floor(NN - 1e-12)); i = np.arange(1, n + 1)
    return float(np.sum(np.where(i % 2 == 0, 1.0, -1.0) * (1 - i / NN)))
Av = np.array([A(x) for x in Nm])
idx = np.minimum(np.ceil(Nm).astype(int), E)
Phi_naive = C4 * Av * T_naive[idx]
Phi_cond = C4 * Av * T_cond[idx]
z = np.load("data/boundary_layer.npz")
Nb = z["Nbins"]; dl = np.diff(np.log(Nb))
meas_w = np.nanmean([z["wmean_rho_7"], z["wmean_rho_8"]], axis=0)
meas_u = np.nanmean([z["mean_rho_7"], z["mean_rho_8"]], axis=0)
lam_cond = float(np.sum(w_cond * np.log(np.maximum(np.arange(E + 1), 1))))
out = {
    "kappa_measured_from_Phi_w": float(P * np.nansum(meas_w * dl)),
    "kappa_model_conditioned": float(P * np.trapezoid(Phi_cond, np.log(Nm))),
    "kappa_closed_form_minus_S2_lambda_over_2": -S2 * lam_cond / 2,
    "lambda_conditioned": lam_cond,
    "unweighted_measured_int": float(P * np.nansum(meas_u * dl)),
    "unweighted_model_naive": float(P * np.trapezoid(Phi_naive, np.log(Nm))),
    "R_slope_finite_diff": {"1e6-1e7": (r["H=1e+07"]["R_sum"] - r["H=1e+06"]["R_sum"]) / np.log(10),
                             "1e7-1e8": (r["H=1e+08"]["R_sum"] - r["H=1e+07"]["R_sum"]) / np.log(10)},
}
Nc = z["N_8"]
out["Phi_w_vs_model_at"] = {f"{n:.2f}": [float(m), float(np.interp(np.log(n), np.log(Nm), Phi_cond))]
                           for n, m in zip(Nc[:16:2], meas_w[:16:2])}
json.dump(out, open("data/bl_model.json", "w"), indent=1)
np.savez("data/bl_model.npz", Nm=Nm, Phi_cond=Phi_cond, Phi_naive=Phi_naive, Nc=Nc, meas_w=meas_w, meas_u=meas_u)
print(json.dumps(out, indent=1))
