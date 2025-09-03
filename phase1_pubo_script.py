

#onvert to spin-monomials suitable for HO-Ising input.
#enerated by ChatGPT (reproducibility harness).

import itertools, json, math
from collections import defaultdict
import numpy as np

# ====== Problem parameters (small instance) ======
m = 3; n = 3; T = 2
y = np.array([0.0, 1.0, 2.0])
C = np.array([3.0, 2.0])

# set inequality-related caps very large so we only enforce aggregate equality in this demo
p = np.full((m, T), 100.0)
l = np.full((m, T), 100.0)
E = np.full(m, 100.0)
feeders = {0: list(range(m))}
P_f = {0: np.array([100.0, 100.0])}

# utilities & weights
U = np.zeros((m, T, n))
for i in range(m):
    for t in range(T):
        U[i,t,0] = 1.0; U[i,t,1] = 0.6; U[i,t,2] = 0.2
gamma = 1.0; kappa_fair_true = 0.5; kappa_sw_true = 0.2

def var_index(i,t,j): return (i * T + t) * n + j
Nvars = m * T * n

def choices_to_b_flat(choices):
    b = [0]*Nvars
    for idx, sel in enumerate(choices):
        i = idx // T; t = idx % T
        b[var_index(i,t,sel)] = 1
    return b

def compute_Y_from_choices(choices):
    Yt = [0.0]*T
    for i in range(m):
        for t in range(T):
            sel = choices[i*T + t]
            Yt[t] += y[sel]
    return np.array(Yt, dtype=float)

def onehot_metrics(choices):
    b_flat = choices_to_b_flat(choices)
    Yt = compute_Y_from_choices(choices)
    # aggregate equality full-square
    v_agg = sum((sum(y[choices[i*T + t]] for i in range(m)) - C[t])**2 for t in range(T))
    base_cost = sum(0.5 * (Yt[t]**2) for t in range(T))
    util = sum(U[i,t,choices[i*T + t]] for i in range(m) for t in range(T))
    pi = [ sum(0 if choices[i*T + t] == 0 else 1 for t in range(T)) for i in range(m) ]
    mean_pi = sum(pi)/len(pi)
    fair = sum((pi_i - mean_pi)**2 for pi_i in pi)
    sw = sum(1.0 if choices[i*T + t] != choices[i*T + (t-1)] else 0.0 for i in range(m) for t in range(1,T))
    base_obj = base_cost - gamma * util + kappa_fair_true * fair + kappa_sw_true * sw
    return {"b_flat": b_flat, "Yt": Yt.tolist(), "v_agg": v_agg, "base_obj": base_obj, "fair": fair, "sw": sw}

# enumerate one-hot assignments
all_results = []
for prod in itertools.product(range(n), repeat=m*T):
    info = onehot_metrics(prod)
    all_results.append((prod, info))

feasible = [(choices,info) for choices,info in all_results if abs(info["v_agg"]) < 1e-12]
if len(feasible) == 0:
    raise RuntimeError("No feasible assignment for aggregate equality; adjust C or y.")

best = min(feasible, key=lambda x: x[1]["base_obj"])
# compute kappa to penalize P_agg across one-hot infeasible assignments
infeas = [(choices,info) for choices,info in all_results if info["v_agg"] > 0.0]
kappa_needed = 0.0
for choices,info in infeas:
    req = (best[1]["base_obj"] - info["base_obj"]) / info["v_agg"]
    if req > kappa_needed: kappa_needed = req
kappa = kappa_needed + 1.0
# set kappa_1h as large multiplier of base range
base_vals = [info["base_obj"] for _,info in all_results]
range_base = max(base_vals) - min(base_vals)
kappa_1h = max(10.0, 100.0 * range_base)

# construct monomial dictionary H(b) as multilinear polynomial in b variables
mono = defaultdict(float)
# base quadratic
for t in range(T):
    var_list = [(var_index(i,t,j), float(y[j])) for i in range(m) for j in range(n)]
    for va, ya in var_list:
        for vb, yb in var_list:
            mono[(va,vb)] += 0.5 * ya * yb
# -gamma utility (linear)
for i in range(m):
    for t in range(T):
        for j in range(n):
            mono[(var_index(i,t,j),)] += - gamma * U[i,t,j]
# fairness quadratic via null selection matrix
null_vars = [var_index(i,t,0) for i in range(m) for t in range(T)]
num_null = len(null_vars)
A = [[0.0]*num_null for _ in range(m)]
for i in range(m):
    for k, v in enumerate(null_vars):
        i0 = v // (T*n)
        if i0 == i: A[i][k] = -1.0
        A[i][k] += 1.0 / m
Mmat = [[0.0]*num_null for _ in range(num_null)]
for i in range(m):
    for a in range(num_null):
        for b in range(num_null):
            Mmat[a][b] += A[i][a] * A[i][b]
for a in range(num_null):
    for b in range(num_null):
        mono[(null_vars[a], null_vars[b])] += kappa_fair_true * Mmat[a][b]
# switching base constants and pairwise negatives
mono[()] += kappa_sw_true * (m * (T-1))
for i in range(m):
    for t in range(1,T):
        for j in range(n):
            mono[(var_index(i,t,j), var_index(i,t-1,j))] += - kappa_sw_true
# P1h penalty scaled by kappa_1h
for i in range(m):
    for t in range(T):
        mono[()] += kappa_1h * 1.0
        for j in range(n):
            mono[(var_index(i,t,j),)] += kappa_1h * -2.0
        for j1 in range(n):
            for j2 in range(n):
                mono[(var_index(i,t,j1), var_index(i,t,j2))] += kappa_1h * 1.0
# P_agg scaled by kappa (expanded)
for t in range(T):
    var_list = [(var_index(i,t,j), float(y[j])) for i in range(m) for j in range(n)]
    for va, ya in var_list:
        for vb, yb in var_list:
            mono[(va,vb)] += kappa * (ya * yb)
    for va, ya in var_list:
        mono[(va,)] += kappa * (-2.0 * C[t] * ya)
    mono[()] += kappa * (C[t]**2)

# Convert to spin-monomials using b=(1+s)/2 identity
spin_mono = defaultdict(float)
for S_b, coeff in mono.items():
    k = len(S_b)
    if k == 0:
        spin_mono[()] += coeff; continue
    scale = coeff * (1.0/(2**k))
    for r in range(k+1):
        for U in itertools.combinations(S_b, r):
            spin_mono[tuple(sorted(U))] += scale

# Save outputs
out = {
    "m":m,"n":n,"T":T,"y":y.tolist(),"C":C.tolist(),"best_choices":best[0],
    "best_base":best[1]["base_obj"], "kappa":kappa,"kappa_1h":kappa_1h,
    "num_spin_monomials": sum(1 for k in spin_mono if len(k)>0), "spin_const": spin_mono.get((),0.0)
}
with open("phase1_pubo_output.json","w") as f:
    json.dump(out, f, indent=2)

print("Wrote phase1_pubo_output.json; best feasible choices:", best[0], "best base:", best[1]["base_obj"]) 
