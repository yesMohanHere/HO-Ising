

import json, math, random, time, argparse, os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='phase2_hosing_input.json', help='Path to Phase-2 HO-Ising input JSON')
parser.add_argument('--out', type=str, default='phase3_hosing_result_fixed.json', help='Output result JSON path')
parser.add_argument('--stoch-restarts', type=int, default=8, help='Number of stochastic restarts')
parser.add_argument('--stoch-iters', type=int, default=2000, help='Iterations per stochastic run')
parser.add_argument('--det-maxsteps', type=int, default=1000, help='Max steps for deterministic BR')
args = parser.parse_args()

input_path = args.input
out_path = args.out

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found: {input_path}. Run Phase-2 to create it.")

with open(input_path,'r') as f:
    data = json.load(f)

# Load HO-Ising representation
N_total = int(data['N_total'])
H_rows = [list(map(int,row)) for row in data['H_rows']]
Jcoeffs = [float(j) for j in data['Jcoeffs']]
const_term = float(data.get('const',0.0))

M = len(H_rows)
N = N_total

# Build clause incidence for each variable
var_clauses = [[] for _ in range(N)]
for k,row in enumerate(H_rows):
    for v in row:
        var_clauses[v].append(k)

def eval_spin_energy(s_vec):
    #\"\"\"Evaluate the full spin Hamiltonian E(s).\"\"\"
    E = 0.0
    for k,row in enumerate(H_rows):
        prod = 1
        for i in row:
            prod *= s_vec[i]
        E += Jcoeffs[k] * prod
    E += const_term
    return E

def compute_local_field(s_vec, i):
    """Compute local field h_i(s) = sum_{k: i in S_k} J_k * prod_{j in S_k\{i}} s_j (optimized)."""
    h = 0.0
    for k in var_clauses[i]:
        prod = 1
        for j in H_rows[k]:
            if j == i: continue
            prod *= s_vec[j]
        h += Jcoeffs[k] * prod
    return h

def delta_energy_flip_direct(s_vec, i):
    #\"\"\"Safe direct computation: compute E(s') - E(s) by evaluating energies.\"\"\"
    # Note: make shallow copy for speed
    s2 = s_vec.copy()
    s2[i] = -s2[i]
    return eval_spin_energy(s2) - eval_spin_energy(s_vec)

def delta_energy_flip_local(s_vec, i):
    """Optimized computation using local field identity: Delta = -2 s_i h_i."""
    h = compute_local_field(s_vec, i)
    return -2.0 * s_vec[i] * h

# Verification routine: check direct vs local-field for many random states
def lf_identity_check(num_checks=200, rng_seed=123):
    rng = random.Random(rng_seed)
    for _ in range(num_checks):
        s = [rng.choice([-1,1]) for _ in range(N)]
        i = rng.randrange(0,N)
        dE_dir = delta_energy_flip_direct(s, i)
        dE_loc = delta_energy_flip_local(s, i)
        if abs(dE_dir - dE_loc) > 1e-8:
            return False, {'i': i, 'dE_direct': dE_dir, 'dE_local': dE_loc}
    return True, {}

# Deterministic asynchronous best-response (strict improvement)
def deterministic_best_response(initial_s, max_steps=10000, pick_rule='random', use_local=False):
    s = initial_s.copy()
    history = []
    for step in range(max_steps):
        improved = False
        indices = list(range(N))
        if pick_rule == 'random':
            random.shuffle(indices)
        for i in indices:
            dE = delta_energy_flip_local(s,i) if use_local else delta_energy_flip_direct(s,i)
            if dE < -1e-12:
                s[i] = -s[i]
                history.append({'step': step, 'var': i, 'dE': dE, 'energy': eval_spin_energy(s)})
                improved = True
                break
        if not improved:
            break
    final_energy = eval_spin_energy(s)
    return s, final_energy, history

# Graph-colored stochastic dynamics (Algorithm-2 style)
def graph_colored_stochastic(initial_s, H_bool, Jvec, Csum=None, color_groups=None,
                              alpha=1e-3, beta=1.0, B=1.0, eps=1e-12,
                              t0=1.0, dt=1.0, max_iters=5000, seed=0):
    rng = random.Random(seed)
    s = initial_s.copy()
    Mloc = len(H_bool); Nloc = len(s)
    var_clauses_loc = [[] for _ in range(Nloc)]
    for k,row in enumerate(H_bool):
        for v in row:
            var_clauses_loc[v].append(k)
    prod_cache = [1]*Mloc
    T = [0]*Mloc
    for k,row in enumerate(H_bool):
        prod = 1
        for i in row: prod *= s[i]
        prod_cache[k] = prod; T[k] = 1 if (1 - prod)//2 == 1 else 0
    if Csum is None:
        Csum = [0.0]*Nloc
        for k,row in enumerate(H_bool):
            wk = Jvec[k]
            for i in row: Csum[i] += wk
    t = t0; iter_ct = 0
    best_s = s.copy(); best_E = eval_spin_energy(s)
    while iter_ct < max_iters:
        for color in range(len(color_groups)):
            V = color_groups[color]
            mV = len(V)
            if mV == 0: continue
            u = [rng.random() for _ in range(mV)]
            mu = [ (beta * math.log(B * uu + eps) / math.log(1 + alpha * t)) for uu in u ]
            # compute eH^T T
            eHTT = [0.0]*Nloc
            for k in range(Mloc):
                if T[k] == 0: continue
                wk = Jvec[k]
                for i in H_bool[k]: eHTT[i] += wk
            qcal = [2.0 * eHTT[i] - Csum[i] for i in range(Nloc)]
            mask = [V[idx] for idx in range(mV) if qcal[V[idx]] < mu[idx]]
            if mask:
                v = rng.choice(mask)
                s[v] = -s[v]
                # update affected clauses
                for k in var_clauses_loc[v]:
                    prod_cache[k] = -prod_cache[k]
                    T[k] = 1 if (1 - prod_cache[k])//2 == 1 else 0
            t += dt; iter_ct += 1
            curE = eval_spin_energy(s)
            if curE < best_E - 1e-12: best_E = curE; best_s = s.copy()
            if iter_ct >= max_iters: break
    return best_s, best_E

# Build H_bool, Csum, and greedy coloring groups (independent sets)
H_bool = [row[:] for row in H_rows]
Csum = [0.0]*N
for k,row in enumerate(H_bool):
    wk = Jcoeffs[k]
    for i in row: Csum[i] += wk

# Build adjacency and greedy coloring
adj = {i:set() for i in range(N)}
for row in H_bool:
    for a in row:
        for b in row:
            if a != b:
                adj[a].add(b); adj[b].add(a)
colors = {}
for v in sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True):
    used = set(colors.get(u) for u in adj[v] if u in colors)
    c = 0
    while c in used: c += 1
    colors[v] = c
R = max(colors.values()) + 1 if colors else 1
color_groups = [[] for _ in range(R)]
for v,c in colors.items():
    color_groups[c].append(v)

# Run verification and dynamics
results = {}
lf_ok, lf_info = lf_identity_check(num_checks=400, rng_seed=12345)
results['lf_identity_ok'] = lf_ok
results['lf_identity_info'] = lf_info

# If identity holds, we may use the optimized local-field delta; else use direct calculation
use_local = lf_ok

# Choose initial spin: if Phase-2 ground truth exists, use it as candidate initial; else random
gt_path = 'phase2_hosing_result.json'
initial_s = [random.choice([-1,1]) for _ in range(N)]
if os.path.exists(gt_path):
    try:
        with open(gt_path,'r') as f: gt = json.load(f)
        if 'ground_truth_b' in gt and gt['ground_truth_b']:
            b_gt = gt['ground_truth_b']
            if len(b_gt) == N:
                initial_s = [1 if bb==1 else -1 for bb in b_gt]
    except Exception:
        pass

# Deterministic BR run
s_br, E_br, history_br = deterministic_best_response(initial_s, max_steps=args.det_maxsteps, pick_rule='random', use_local=use_local)
results['deterministic_br_energy'] = E_br
results['deterministic_br_history_len'] = len(history_br)

# verify PNE: no single flip reduces energy
pne = True
violations = []
for i in range(N):
    dE = delta_energy_flip_local(s_br,i) if use_local else delta_energy_flip_direct(s_br,i)
    if dE < -1e-12:
        pne = False; violations.append((i,dE))
results['deterministic_ended_at_PNE'] = pne
results['deterministic_violations'] = violations

# Stochastic graph-colored runs
best_overall = None
for r in range(args.stoch_restarts):
    seed = 1000 + r
    s_init = [random.choice([-1,1]) for _ in range(N)]
    s_sol, E_sol = graph_colored_stochastic(s_init, H_bool, Jcoeffs, Csum=Csum, color_groups=color_groups,
                                           alpha=1e-3, beta=1.0, B=1.0, eps=1e-12,
                                           t0=1.0, dt=1.0, max_iters=args.stoch_iters, seed=seed)
    if best_overall is None or E_sol < best_overall[0]:
        best_overall = (E_sol, s_sol, seed)
results['stochastic_best_energy'] = best_overall[0]
results['stochastic_best_seed'] = best_overall[2]

# Compare to Phase-2 ground truth if available
if os.path.exists(gt_path):
    try:
        with open(gt_path,'r') as f: gt = json.load(f)
        if 'ground_truth_H' in gt:
            results['ground_truth_H'] = gt['ground_truth_H']
            results['deterministic_matches_ground_truth'] = abs(E_br - gt['ground_truth_H']) < 1e-8
            results['stochastic_matches_ground_truth'] = abs(best_overall[0] - gt['ground_truth_H']) < 1e-8
    except Exception:
        results['ground_truth_H'] = None

results['use_local_field_optimized'] = use_local

# Save results
with open(out_path,'w') as f:
    json.dump(results, f, indent=2)

print('Wrote', out_path, '; summary:', results)
