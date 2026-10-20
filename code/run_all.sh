#!/usr/bin/env bash
# Reproduce every table and figure of the paper. Total about 12 minutes on one core, peak memory about 4 GB.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
python3 twin_swing.py           # race to 1e10, window variances, coin-flip nulls      (~2 min)
python3 echo_constant.py        # Bateman-Horn constant, sigma check                   (~10 s)
python3 twin_hl.py              # exact singular series, Hardy-Littlewood predictions  (~15 s)
python3 twin_coefficient.py     # T4 = M + R, identity C4 P = S2                        (~10 s)
python3 remainder_structure.py  # per-divisor split, pairing lemma check               (~10 s)
python3 boundary_layer.py       # rho_d for all d <= H/6, profile Phi(N)               (~4 min)
python3 bl_model.py             # fit-free model of Phi, slope kappa                    (~1 min)
python3 spectral_check.py       # Wiener-Khinchin identity, small-q part, mass M(Q)    (~5 min)
python3 near_quarter.py         # weights of atoms nearest +-1/4                       (~3 min)
python3 paper_figures.py        # figures 1-5 -> ../paper/figures/*.pdf
echo "done"
