#!/usr/bin/env python3
"""Phase2 HO-Ising runner
- Builds small DR instance with exact slack-encoding (same as interactive Phase2 run).
- Constructs spin-monomials (HO-Ising input) and saves them.
- Runs a graph-colored asynchronous HO-Ising sampler inspired by Algorithm 2 on CPU.
- Compares solver results to ground-truth global minimum via reduced exhaustive search.
"""
import itertools, math, json, random, time
from collections import defaultdict

# ---- Problem definition (tiny instance) ----
m = 2; n = 2; T = 2
y = [0,1]
C = [1,1]

# inequality caps
p = [[1]*T for _ in range(m)]
P_f = {0: [2,2]}
E = [2]*m

# utilities and weights
U = [[[1.0,0.6] for t in range(T)] for i in range(m)]
gamma = 1.0; kappa_fair = 0.5; kappa_sw = 0.2

def b_var_index(i,t,j): return (i * T + t) * n + j
N_b = m * T * n

# slack specs like interactive run
slack_specs = []
for i in range(m):
    for t in range(T):
        cap = p[i][t]; K = max(1, math.ceil(math.log2(cap+1))); slack_specs.append((f"slack_p_i{i}_t{t}", cap, K, ("agent_cap", i, t)))
for t in range(T):
    cap = P_f[0][t]; K = max(1, math.ceil(math.log2(cap+1))); slack_specs.append((f"slack_feed_t{t}", cap, K, ("feeder_cap", 0, t)))
for i in range(m):
    cap = E[i]; K = max(1, math.ceil(math.log2(cap+1))); slack_specs.append((f"slack_budget_i{i}", cap, K, ("budget", i)))

# assign slack indices
slack_var_indices = {}
current_index = N_b
for spec in slack_specs:
    name, cap, K, info = spec
    for k in range(K):
        slack_var_indices[(name,k)] = current_index
        current_index += 1
N_total = current_index

# Build eq_list mapping for constructing equality expressions
spec_to_exprvars = {}
for spec in slack_specs:
    name, cap, K, info = spec
    typ = info[0]
    if typ == "agent_cap":
        i,t = info[1], info[2]
        var_list = [(b_var_index(i,t,j), y[j]) for j in range(n)]
    elif typ == "feeder_cap":
        f,t = info[1], info[2]
        var_list = [(b_var_index(i,t,j), y[j]) for i in range(m) for j in range(n)]
    elif typ == "budget":
        i = info[1]
        var_list = [(b_var_index(i,t,j), y[j]) for t in range(T) for j in range(n)]
    spec_to_exprvars[name] = (var_list, cap, K)

eq_list = []
for spec in slack_specs:
    name, cap, K, info = spec
    var_list, cap_val, Kbits = spec_to_exprvars[name]
    eq_list.append((var_list, name, cap_val, Kbits))

# functions for objective, penalties, etc.
def compute_Yt_from_bflat(b_flat):
    Yt = [0]*T
    for t in range(T):
        total = 0
        for i in range(m):
            for j in range(n):
                total += y[j]*b_flat[b_var_index(i,t,j)]
        Yt[t] = total
    return Yt

def compute_base_objective(b_flat):
    Yt = compute_Yt_from_bflat(b_flat)
    base_cost = sum(0.5 * (Yt[t]**2) for t in range(T))
    util = 0.0
    for i in range(m):
        for t in range(T):
            for j in range(n):
                util += U[i][t][j] * b_flat[b_var_index(i,t,j)]
    pi = [0]*m
    for i in range(m):
        for t in range(T):
            pi[i] += 0 if b_flat[b_var_index(i,t,0)] == 1 else 1
    mean_pi = sum(pi)/m
    fair = sum((pi_i - mean_pi)**2 for pi_i in pi)
    sw = 0.0
    for i in range(m):
        for t in range(1,T):
            same = 0
            for j in range(n):
                same += b_flat[b_var_index(i,t,j)] * b_flat[b_var_index(i,t-1,j)]
            sw += (1 - same)
    base_obj = base_cost - gamma * util + kappa_fair * fair + kappa_sw * sw
    return base_obj

def eq_penalty_value(b_flat):
    tot = 0.0
    for var_list, name, cap, K in eq_list:
        expr = sum(coeff * b_flat[vi] for (vi, coeff) in var_list)
        S = 0
        for k in range(K):
            S += (1 << k) * b_flat[slack_var_indices[(name,k)]]
        diff = expr + S - cap
        tot += diff*diff
    return tot

def P1h_value(b_flat):
    s = 0.0
    for i in range(m):
        for t in range(T):
            sm = sum(b_flat[b_var_index(i,t,j)] for j in range(n))
            s += (1 - sm)**2
    return s

# Calibrate kappas by scanning one-hot decisions x slack combos (reduced space)
onehot_decisions = list(itertools.product(range(n), repeat=m*T))
slack_bits = [idx for idx in range(N_b, N_total)]
slack_count = len(slack_bits)
slack_combos = list(range(1<<slack_count))

# Compute best base among feasible decision-only assignments (inequalities without slack)
feasible_decisions = []
for choice in onehot_decisions:
    b = [0]*N_total
    for pos, sel in enumerate(choice):
        i = pos // T; t = pos % T
        b[b_var_index(i,t,sel)] = 1
    violated = False
    for var_list, name, cap, K in eq_list:
        expr = sum(coeff * b[vi] for (vi, coeff) in var_list)
        if expr > cap + 1e-12:
            violated = True; break
    if not violated:
        feasible_decisions.append((choice, b, compute_base_objective(b)))
if len(feasible_decisions) == 0:
    raise RuntimeError("No feasible decisions under raw inequalities; change instance")
best_base = min(x[2] for x in feasible_decisions)

# compute kappa_eq and kappa_1h
kappa_needed = 0.0
base_vals = []
for choice, b_dec, base_dec in feasible_decisions:
    for s_mask in slack_combos:
        b = b_dec.copy()
        for bitpos, gidx in enumerate(slack_bits):
            b[gidx] = 1 if ((s_mask >> bitpos) & 1) else 0
        base = compute_base_objective(b)
        pen = eq_penalty_value(b)
        base_vals.append(base)
        if pen > 0:
            req = (best_base - base) / pen
            if req > kappa_needed: kappa_needed = req
kappa_eq = kappa_needed + 1.0
range_base = max(base_vals) - min(base_vals) if base_vals else 1.0
kappa_1h = max(10.0, 100.0 * range_base)

# Build b-space monomial Hmono = base_mono + kappa_eq * eq_mono + kappa_1h * p1h_mono (as in interactive run)
base_mono = defaultdict(float)
# base quadratic
for t in range(T):
    var_list = [(b_var_index(i,t,j), y[j]) for i in range(m) for j in range(n)]
    for va, ya in var_list:
        for vb, yb in var_list:
            base_mono[(va,vb)] += 0.5 * ya * yb
# -gamma utility
for i in range(m):
    for t in range(T):
        for j in range(n):
            base_mono[(b_var_index(i,t,j),)] += - gamma * U[i][t][j]
# fairness via null vars
null_vars = [b_var_index(i,t,0) for i in range(m) for t in range(T)]
num_null = len(null_vars)
A = [[0.0]*num_null for _ in range(m)]
for i in range(m):
    for k_idx, v in enumerate(null_vars):
        i0 = v // (T*n)
        if i0 == i: A[i][k_idx] = -1.0
        A[i][k_idx] += 1.0 / m
Mmat = [[0.0]*num_null for _ in range(num_null)]
for i in range(m):
    for a in range(num_null):
        for b in range(num_null):
            Mmat[a][b] += A[i][a]*A[i][b]
for a in range(num_null):
    for b in range(num_null):
        base_mono[(null_vars[a], null_vars[b])] += kappa_fair * Mmat[a][b]
# switching
base_mono[()] += kappa_sw * (m * (T-1))
for i in range(m):
    for t in range(1,T):
        for j in range(n):
            base_mono[(b_var_index(i,t,j), b_var_index(i,t-1,j))] += - kappa_sw

# p1h mono
p1h_mono = defaultdict(float)
for i in range(m):
    for t in range(T):
        p1h_mono[()] += 1.0
        for j in range(n):
            p1h_mono[(b_var_index(i,t,j),)] += -2.0
        for j1 in range(n):
            for j2 in range(n):
                p1h_mono[(b_var_index(i,t,j1), b_var_index(i,t,j2))] += 1.0

# eq_mono expansion
eq_mono = defaultdict(float)
for var_list, name, cap_val, Kbits in eq_list:
    for va, ca in var_list:
        for vb, cb in var_list:
            eq_mono[(va,vb)] += ca * cb
    for va, ca in var_list:
        for k in range(Kbits):
            zidx = slack_var_indices[(name,k)]
            eq_mono[(va, zidx)] += 2.0 * ca * (1 << k)
    for k1 in range(Kbits):
        for k2 in range(Kbits):
            idx1 = slack_var_indices[(name,k1)]; idx2 = slack_var_indices[(name,k2)]
            eq_mono[(idx1, idx2)] += (1 << k1) * (1 << k2)
    for va, ca in var_list:
        eq_mono[(va,)] += -2.0 * cap_val * ca
    for k in range(Kbits):
        idx = slack_var_indices[(name,k)]
        eq_mono[(idx,)] += -2.0 * cap_val * (1 << k)
    eq_mono[()] += cap_val**2

# combine into Hmono
Hmono = defaultdict(float)
for S,c in base_mono.items(): Hmono[S] += c
for S,c in eq_mono.items(): Hmono[S] += kappa_eq * c
for S,c in p1h_mono.items(): Hmono[S] += kappa_1h * c

# convert to spin monomials
spin_mono = defaultdict(float)
import itertools as _it
for S_b, coeff in Hmono.items():
    k = len(S_b)
    if k == 0:
        spin_mono[()] += coeff; continue
    scale = coeff * (1.0/(2**k))
    for r in range(k+1):
        for Uc in _it.combinations(S_b, r):
            spin_mono[tuple(sorted(Uc))] += scale

# prepare HO-Ising rows (each row = list of variable indices) and weights J
H_rows = []; Jcoeffs = []
const_term = spin_mono.get((), 0.0)
for S_s, coeff in spin_mono.items():
    if len(S_s) == 0: continue
    H_rows.append(list(S_s)); Jcoeffs.append(float(coeff))

# save spin monomials to JSON for external use
out = {
    "N_total": N_total,
    "N_decision": N_b,
    "N_slack": N_total - N_b,
    "H_rows": H_rows,
    "Jcoeffs": Jcoeffs,
    "const": const_term,
    "kappa_eq": kappa_eq,
    "kappa_1h": kappa_1h
}
with open("phase2_hosing_input.json","w") as f:
    json.dump(out, f, indent=2)

# ---- Build eH and other datastructures for Algorithm 2 ----
M = len(H_rows)
N = N_total

H_bool = [row[:] for row in H_rows]
J = Jcoeffs[:]

Csum = [0.0]*N
for k, row in enumerate(H_bool):
    wk = J[k]
    for i in row:
        Csum[i] += wk

# Build adjacency for coloring
adj = {i:set() for i in range(N)}
for row in H_bool:
    for a in row:
        for b in row:
            if a!=b:
                adj[a].add(b); adj[b].add(a)
colors = {}
for v in sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True):
    used = set(colors.get(u) for u in adj[v] if u in colors)
    c = 0
    while c in used: c += 1
    colors[v] = c
R = max(colors.values())+1
color_groups = [[] for _ in range(R)]
for v,c in colors.items(): color_groups[c].append(v)

# ---- Algorithm 2 inspired solver ----
alpha = 1e-3; beta = 1.0; B = 1.0; eps = 1e-12
t0 = 1.0; dt = 1.0
max_iters = 20000

def eval_spin_energy(s_vec):
    E = 0.0
    for k,row in enumerate(H_rows):
        prod = 1
        for i in row:
            prod *= s_vec[i]
        E += J[k] * prod
    E += const_term
    return E

# prepare clause-var incidence
clause_vars = H_bool
var_clauses = [[] for _ in range(N)]
for k,row in enumerate(clause_vars):
    for v in row:
        var_clauses[v].append(k)

def run_graph_colored(num_restarts=10, seed_base=100):
    best_global = None
    detailed = []
    for r in range(num_restarts):
        rng = random.Random(seed_base + r)
        s = [rng.choice([-1,1]) for _ in range(N)]
        t = t0
        iter_ct = 0
        best_local_val = eval_spin_energy(s)
        best_local_s = s.copy()
        prod_cache = [1]*M
        T = [0]*M
        for k,row in enumerate(clause_vars):
            prod = 1
            for i in row: prod *= s[i]
            prod_cache[k] = prod
            T[k] = 1 if (1 - prod)//2 == 1 else 0
        iter_limit = max_iters
        while iter_ct < iter_limit:
            for color in range(R):
                V = color_groups[color]
                mV = len(V)
                if mV == 0: continue
                u = [rng.random() for _ in range(mV)]
                mu = [ (beta * math.log(B * uu + eps) / math.log(1 + alpha * t)) for uu in u ]
                eHTT = [0.0]*N
                for k in range(M):
                    if T[k] == 0: continue
                    wk = J[k]
                    for i in clause_vars[k]:
                        eHTT[i] += wk
                qcal = [2.0 * eHTT[i] - Csum[i] for i in range(N)]
                mask_indices = [V[idx] for idx,val in enumerate(V) if qcal[val] < mu[idx]]
                if mask_indices:
                    v = rng.choice(mask_indices)
                    s[v] = -s[v]
                    for k in var_clauses[v]:
                        prod_cache[k] = prod_cache[k] * -1
                        T[k] = 1 if (1 - prod_cache[k])//2 == 1 else 0
                t += dt
                iter_ct += 1
                Ecur = eval_spin_energy(s)
                if Ecur < best_local_val:
                    best_local_val = Ecur; best_local_s = s.copy()
        detailed.append({"restart": r, "best_local_val": best_local_val})
        if best_global is None or best_local_val < best_global[0]:
            best_global = (best_local_val, best_local_s, r)
    return best_global, detailed

print("Running graph-colored HO-Ising solver (num_restarts=20)...")
t0_run = time.time()
best_global, details = run_graph_colored(num_restarts=20, seed_base=1234)
t1_run = time.time()
print("Solver finished in %.3f s" % (t1_run - t0_run))
print("Best found H:", best_global[0], "restart:", best_global[2])

best_s = best_global[1]
best_b = [(1 + si)//2 for si in best_s]

# compute ground truth by reduced exhaustive search
best_gt = None
for choice in onehot_decisions:
    b_dec = [0]*N_total
    for pos, sel in enumerate(choice):
        i = pos // T; t = pos % T
        b_dec[b_var_index(i,t,sel)] = 1
    for s_mask in slack_combos:
        b = b_dec.copy()
        for bitpos, gidx in enumerate(slack_bits):
            b[gidx] = 1 if ((s_mask >> bitpos) & 1) else 0
        val = compute_base_objective(b) + kappa_eq * eq_penalty_value(b) + kappa_1h * P1h_value(b)
        if best_gt is None or val < best_gt[0]:
            best_gt = (val, b.copy(), choice, s_mask)
print("Ground truth (reduced) value:", best_gt[0])

print("Best solver H:", best_global[0], "Ground truth H:", best_gt[0])
match_flag = abs(best_global[0] - best_gt[0]) < 1e-8
print("Solver matched ground truth?", match_flag)

with open("phase2_hosing_result.json","w") as f:
    json.dump({
        "N_total": N_total, "N_decision": N_b, "N_slack": N_total - N_b,
        "H_rows": H_rows, "Jcoeffs": Jcoeffs, "const": const_term,
        "kappa_eq": kappa_eq, "kappa_1h": kappa_1h,
        "solver_best_H": best_global[0], "solver_best_s": best_global[1],
        "ground_truth_H": best_gt[0], "ground_truth_b": best_gt[1]
    }, f, indent=2)

print("Wrote phase2_hosing_input.json and phase2_hosing_result.json")
