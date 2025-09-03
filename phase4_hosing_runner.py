#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-4 (STRICT) — DR → PUBO → HO-Ising (Exact), Algorithm-2 Solver, and Validation
Author: (Your project)
License: MIT
"""

import argparse, json, math, os, random, time, itertools
from collections import defaultdict

# --------------------------- Paths / Defaults -----------------------------
DEFAULT_OUTDIR = r"D:/coding/finalwork/phase4_out"  # your local path

# Configuration profiles
CONFIG_PROFILES = {
    "small_debug": {
        "description": "Tiny runs to verify feasibility and algebra; fastest checks.",
        "scaling": {"delta": 0.1},
        "instance": {
            "num_agents": 5, "T": 4, "options_per_agent": 2, "scenario_count": 2,
            "robust_perturbation_factor_min": 0.85, "robust_perturbation_factor_max": 1.15
        },
        "penalties": {
            "gamma": 1.0, "kappa_eq": 10000.0, "kappa_1h": 300.0,
            "kappa_fair": 0.2, "kappa_sw": 0.5
        },
        "solver": {
            "deterministic_max_steps": 5000, "stochastic_max_iters": 5000,
            "stochastic_alpha0": 0.002, "stochastic_beta": 1.0, "stochastic_B0": 1.0,
            "stochastic_eps": 1e-12, "warm_start_from_deterministic": True, "multi_start_restarts": 2
        },
        "feasibility_phase": {
            "enabled": True, "penalty_scales": [1.0, 10.0, 100.0], "stop_when_residual_zero": True
        },
        "checks": {
            "require_pubo_equals_spin": True, "require_zero_residuals": True,
            "relative_tolerance": 1e-9, "absolute_tolerance": 1e-8
        }
    },
    "medium_default": {
        "description": "Balanced settings for moderate sizes; good baseline.",
        "scaling": {"delta": 0.2},
        "instance": {
            "num_agents": 10, "T": 6, "options_per_agent": 2, "scenario_count": 3,
            "robust_perturbation_factor_min": 0.85, "robust_perturbation_factor_max": 1.30
        },
        "penalties": {
            "gamma": 1.0, "kappa_eq": 30000.0, "kappa_1h": 1000.0,
            "kappa_fair": 0.5, "kappa_sw": 0.8
        },
        "solver": {
            "deterministic_max_steps": 8000, "stochastic_max_iters": 12000,
            "stochastic_alpha0": 0.003, "stochastic_beta": 1.0, "stochastic_B0": 1.0,
            "stochastic_eps": 1e-12, "warm_start_from_deterministic": True, "multi_start_restarts": 4
        },
        "feasibility_phase": {
            "enabled": True, "penalty_scales": [1.0, 10.0, 100.0], "stop_when_residual_zero": True
        },
        "checks": {
            "require_pubo_equals_spin": True, "require_zero_residuals": True,
            "relative_tolerance": 1e-9, "absolute_tolerance": 1e-8
        }
    },
    "aggressive_feasibility": {
        "description": "Use when residuals refuse to go to zero; prioritizes feasibility.",
        "scaling": {"delta": 0.25},
        "instance": {
            "num_agents": 15, "T": 8, "options_per_agent": 2, "scenario_count": 4,
            "robust_perturbation_factor_min": 0.85, "robust_perturbation_factor_max": 1.30
        },
        "penalties": {
            "gamma": 1.0, "kappa_eq": 80000.0, "kappa_1h": 3000.0,
            "kappa_fair": 0.5, "kappa_sw": 1.0
        },
        "solver": {
            "deterministic_max_steps": 12000, "stochastic_max_iters": 20000,
            "stochastic_alpha0": 0.005, "stochastic_beta": 1.0, "stochastic_B0": 1.0,
            "stochastic_eps": 1e-12, "warm_start_from_deterministic": True, "multi_start_restarts": 6
        },
        "feasibility_phase": {
            "enabled": True, "penalty_scales": [10.0, 100.0, 300.0], "stop_when_residual_zero": True
        },
        "checks": {
            "require_pubo_equals_spin": True, "require_zero_residuals": True,
            "relative_tolerance": 1e-9, "absolute_tolerance": 1e-8
        }
    }
}

def now_str(): return time.strftime("%Y%m%d-%H%M%S")
def write_json(path,obj):
    def convert_tuples_to_strings(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: convert_tuples_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tuples_to_strings(item) for item in obj]
        elif isinstance(obj, tuple):
            return str(obj)
        else:
            return obj
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w", encoding="utf-8") as f:
        json.dump(convert_tuples_to_strings(obj), f, indent=2, sort_keys=True)

# --------------------------- Instance Generator ---------------------------
def generate_dr_instance(num_agents=10, T=4, options_per_agent=2,
                         feeder_caps=None, seed=0, scenario_count=1,
                         include_robustness=True, delta=0.1):
    """
    Generate DR instance with quantization to ensure integer compatibility.
    delta: base unit for quantization (e.g., 0.1 kWh)
    """
    rng = random.Random(seed)
    decision_idx = {}
    idx = 0
    for a in range(num_agents):
        for t in range(T):
            for o in range(options_per_agent):
                decision_idx[idx] = {"agent": a, "t": t, "opt": o}
                idx += 1
    N_dec = idx

    # nominal y and utilities (with quantization)
    y_nominal = [0.0]*N_dec
    u = [0.0]*N_dec
    for i,meta in decision_idx.items():
        a,t,o = meta["agent"], meta["t"], meta["opt"]
        if o == 0:
            y_nominal[i] = 0.0  # null option always has zero consumption
            u[i] = 0.15 + 0.35*rng.random()
        else:
            # Quantize consumption to integer grid
            raw_y = 1.0 + 2.0*rng.random()
            y_nominal[i] = round(raw_y / quantize_delta) * quantize_delta
            u[i] = 0.8 + 1.2*rng.random()

    # budgets (quantized)
    E_a = {}
    for a in range(num_agents):
        raw_budget = 0.5*T
        E_a[a] = max(1, round(raw_budget / quantize_delta)) * quantize_delta

    # feeder caps (quantized)
    if feeder_caps is None:
        feeder_caps = []
        for t in range(T):
            raw_cap = 0.6*num_agents
            feeder_caps.append(max(1, round(raw_cap / quantize_delta)) * quantize_delta)

    # scenarios (with quantization)
    scenarios = []
    for r in range(scenario_count):
        if r == 0 or not include_robustness:
            y_r = list(y_nominal)
            P_r = list(feeder_caps)
        else:
            # Quantize perturbed values
            y_r = []
            for yi in y_nominal:
                perturbation = robust_perturbation_factor_min + rng.random() * (robust_perturbation_factor_max - robust_perturbation_factor_min)
                raw_y = max(0.0, yi * perturbation)
                y_r.append(round(raw_y / quantize_delta) * quantize_delta)
            
            P_r = []
            for cap in feeder_caps:
                perturbation = robust_perturbation_factor_min + rng.random() * (robust_perturbation_factor_max - robust_perturbation_factor_min)
                raw_cap = max(quantize_delta, cap * perturbation)
                P_r.append(round(raw_cap / quantize_delta) * quantize_delta)
        scenarios.append({"y": y_r, "P": P_r})

    return {
        "num_agents": num_agents, "T": T, "options_per_agent": options_per_agent,
        "decision_idx": decision_idx, "y_nominal": y_nominal, "u": u, "E_a": E_a,
        "feeder_caps": feeder_caps, "scenarios": scenarios, "quantize_delta": quantize_delta
    }

# --------------------------- PUBO Builder (Exact) -------------------------
def build_pubo(inst, weights):
    decision_idx = inst["decision_idx"]
    N_dec = len(decision_idx)
    y_nom = inst["y_nominal"]
    u = inst["u"]
    E_a = inst["E_a"]
    scenarios = inst["scenarios"]
    T = inst["T"]
    A = inst["num_agents"]
    options = inst["options_per_agent"]

    PUBO = defaultdict(float)
    ato_to_idx = {(meta["agent"],meta["t"],meta["opt"]):i for i,meta in decision_idx.items()}
    def b_index(a,t,o): return ato_to_idx[(a,t,o)]

    # 1) System cost (nominal scenario only)
    for t in range(T):
        idxs_t = [idx for idx,meta in decision_idx.items() if meta["t"]==t]
        for i in idxs_t:
            for j in idxs_t:
                PUBO[tuple(sorted((i,j)))] += 0.5 * y_nom[i] * y_nom[j]

    # 2) -gamma * utility
    gamma = weights.get("gamma", 1.0)
    for i in range(N_dec):
        PUBO[(i,)] += -gamma * u[i]
    # 3) Fairness / equity
    kappa_fair = weights.get("kappa_fair", 0.0)
    if kappa_fair > 0 and options >= 1:
        null_bits = {a: [] for a in range(A)}
        for idx,meta in decision_idx.items():
            if meta["opt"] == 0:  # null
                null_bits[meta["agent"]].append(idx)
        # fair = sum_a pi_a^2 - (1/A) (sum_a pi_a)^2
        for a in range(A):
            L = null_bits.get(a, [])
            for i in L:
                for j in L:
                    PUBO[tuple(sorted((i,j)))] += kappa_fair
        all_null = [i for L in null_bits.values() for i in L]
        for i in all_null:
            for j in all_null:
                PUBO[tuple(sorted((i,j)))] += -kappa_fair * (1.0/A)

    # 4) Switching / ramping
    kappa_sw = weights.get("kappa_sw", 0.0)
    if kappa_sw > 0:
        for a in range(A):
            for t in range(1, T):
                for o in range(options):
                    i = b_index(a,t,o); j = b_index(a,t-1,o)
                    PUBO[(i,)] += kappa_sw
                    PUBO[(j,)] += kappa_sw
                    PUBO[tuple(sorted((i,j)))] += -2.0*kappa_sw

    # 5) Equality constraints (binary slack)
    slack_specs = []
    slack_indices = {}
    cur = N_dec
    # Feeder caps
    for r, sc in enumerate(scenarios):
        for t in range(T):
            cap = int(sc["P"][t])
            K = max(1, math.ceil(math.log2(cap+1)))
            name = f"slack_feed_r{r}_t{t}"
            slack_specs.append((name, cap, K, "feeder", {"r": r, "t": t}))
            for k in range(K):
                slack_indices[(name,k)] = cur; cur += 1
    # Budgets
    for a in range(A):
        cap = int(E_a[a])
        K = max(1, math.ceil(math.log2(cap+1)))
        name = f"slack_budget_a{a}"
        slack_specs.append((name, cap, K, "budget", {"a": a}))
        for k in range(K):
            slack_indices[(name,k)] = cur; cur += 1
    N_total = cur
    kappa_eq = weights.get("kappa_eq", 1e4)

    def add_squared_equality(lin_terms, const, scale):
        for (ia, ca) in lin_terms:
            for (ib, cb) in lin_terms:
                PUBO[tuple(sorted((ia,ib)))] += scale*(ca*cb)
        for (ia, ca) in lin_terms:
            PUBO[(ia,)] += scale*(-2.0*const*ca)
        PUBO[()] += scale*(const*const)

    # Expand feeder equalities
    for (name,cap,K,kind,info) in slack_specs:
        lin = []
        if kind == "feeder":
            r = info["r"]; t = info["t"]
            y_r = scenarios[r]["y"]
            for idx,meta in decision_idx.items():
                if meta["t"] == t:
                    lin.append((idx, y_r[idx]))
            for k in range(K):
                lin.append((slack_indices[(name,k)], float(1<<k)))
            add_squared_equality(lin, cap, kappa_eq)
        elif kind == "budget":
            a = info["a"]
            for idx,meta in decision_idx.items():
                if meta["agent"] == a:
                    lin.append((idx, y_nom[idx]))
            for k in range(K):
                lin.append((slack_indices[(name,k)], float(1<<k)))
            add_squared_equality(lin, cap, kappa_eq)

    # 6) One-hot per (agent, t)
    kappa_1h = weights.get("kappa_1h", 200.0)
    if options > 1 and kappa_1h > 0:
        for a in range(A):
            for t in range(T):
                lin = [(b_index(a,t,o), 1.0) for o in range(options)]
                add_squared_equality(lin, 1.0, kappa_1h)

    PUBO_clean = {tuple(sorted(k)): float(v) for k,v in PUBO.items() if abs(v) > 1e-12}
    meta = {
        "N_dec": N_dec, "N_total": N_total,
        "slack_specs": slack_specs, "slack_indices": slack_indices,
        "decision_idx": decision_idx
    }
    return PUBO_clean, meta
# --------------------------- Evaluation: PUBO & Spin ----------------------
def eval_pubo_binary(PUBO, b_vec):
    E = 0.0
    for S, coeff in PUBO.items():
        prod = 1.0
        for idx in S:
            prod *= b_vec[idx]
            if prod == 0.0:
                break
        E += coeff * prod
    return E

def pubo_to_spin(PUBO):
    spin = defaultdict(float)
    for S_b, coeff in PUBO.items():
        k = len(S_b)
        if k == 0:
            spin[()] += coeff
            continue
        scale = coeff * (1.0/(2**k))
        for r in range(0, k+1):
            for comb in itertools.combinations(S_b, r):
                spin[tuple(sorted(comb))] += scale
    return {tuple(sorted(k)): float(v) for k,v in spin.items() if abs(v) > 1e-12}

def spin_to_Hrows(spin_mono):
    H_rows, J, const = [], [], float(spin_mono.get((),0.0))
    for support, coeff in spin_mono.items():
        if len(support)==0: continue
        H_rows.append(list(support)); J.append(float(coeff))
    return H_rows, J, const

def eval_spin_energy(s_vec,H_rows,Jcoeffs,const_term=0.0):
    E = 0.0
    for k,row in enumerate(H_rows):
        prod = 1
        for i in row: prod *= s_vec[i]
        E += Jcoeffs[k]*prod
    return E+const_term

def b_from_s(s_vec): return [(1 if x==1 else 0) for x in s_vec]

# --------------------------- Local-Field (debug only) ---------------------
def build_var_clauses(H_rows, N):
    vc=[[] for _ in range(N)]
    for k,row in enumerate(H_rows):
        for v in row: vc[v].append(k)
    return vc

def delta_E_direct(s_vec,i,H_rows,Jcoeffs,const_term):
    s2=s_vec.copy(); s2[i]=-s2[i]
    return eval_spin_energy(s2,H_rows,Jcoeffs,const_term)-eval_spin_energy(s_vec,H_rows,Jcoeffs,const_term)

def delta_E_localfield(s_vec,i,H_rows,Jcoeffs,var_clauses):
    h=0.0
    for k in var_clauses[i]:
        prod=1
        for j in H_rows[k]:
            if j==i: continue
            prod*=s_vec[j]
        h+=Jcoeffs[k]*prod
    return -2.0*s_vec[i]*h

def lf_identity_check(H_rows,Jcoeffs,const_term,trials=200,seed=1):
    N=max((max(row) if row else -1) for row in H_rows)+1 if H_rows else 0
    var_clauses=build_var_clauses(H_rows,N)
    rng=random.Random(seed)
    for _ in range(trials):
        s=[rng.choice([-1,1]) for _ in range(N)]
        i=rng.randrange(0,N)
        d1=delta_E_direct(s,i,H_rows,Jcoeffs,const_term)
        d2=delta_E_localfield(s,i,H_rows,Jcoeffs,var_clauses)
        if abs(d1-d2)>1e-8: return False,{"i":i,"dE_direct":d1,"dE_local":d2}
    return True,{}

# --------------------------- Algorithm-2 Solvers --------------------------
def deterministic_BR(H_rows,Jcoeffs,const_term,max_steps=2000,seed=0):
    N=max((max(row) if row else -1) for row in H_rows)+1 if H_rows else 0
    rng=random.Random(seed)
    s=[rng.choice([-1,1]) for _ in range(N)]
    hist=[]
    for step in range(max_steps):
        improved=False
        order=list(range(N)); rng.shuffle(order)
        for i in order:
            dE=delta_E_direct(s,i,H_rows,Jcoeffs,const_term)
            if dE<-1e-12:
                s[i]=-s[i]
                hist.append((step,i,dE,eval_spin_energy(s,H_rows,Jcoeffs,const_term)))
                improved=True; break
        if not improved: break
    return s, eval_spin_energy(s,H_rows,Jcoeffs,const_term), hist

def greedy_coloring(H_rows,N):
    adj={i:set() for i in range(N)}
    for row in H_rows:
        for a in row:
            for b in row:
                if a!=b: adj[a].add(b); adj[b].add(a)
    colors={}
    for v in sorted(adj.keys(),key=lambda x: len(adj[x]),reverse=True):
        used={colors.get(u) for u in adj[v] if u in colors}
        c=0
        while c in used: c+=1
        colors[v]=c
    R=max(colors.values())+1 if colors else 1
    groups=[[] for _ in range(R)]
    for v,c in colors.items(): groups[c].append(v)
    return groups

def algorithm2_stochastic(H_rows,Jcoeffs,const_term,color_groups,
                           max_iters=5000,seed=0,alpha0=1e-3,beta=1.0,B0=1.0,eps=1e-12,
                           s_init=None):
    N=max((max(row) if row else -1) for row in H_rows)+1 if H_rows else 0
    rng=random.Random(seed)
    s = s_init[:] if (s_init is not None) else [rng.choice([-1,1]) for _ in range(N)]
    M=len(H_rows)
    var_clauses=build_var_clauses(H_rows,N)

    prod_cache=[1]*M; T=[0]*M
    for k,row in enumerate(H_rows):
        prod=1
        for i in row: prod*=s[i]
        prod_cache[k]=prod
        T[k]=1 if (1-prod)//2==1 else 0

    Csum=[0.0]*N
    for k,row in enumerate(H_rows):
        wk=Jcoeffs[k]
        for i in row: Csum[i]+=wk

    best_s=s.copy(); bestE=eval_spin_energy(s,H_rows,Jcoeffs,const_term)
    t=1.0
    for it in range(1,max_iters+1):
        alpha_t=alpha0/(1.0+it/1000.0)
        B_t=B0*(1.0+0.01*it)
        for color in range(len(color_groups)):
            V=color_groups[color]; mV=len(V)
            if mV==0: continue
            u=[rng.random() for _ in range(mV)]
            denom=math.log(1.0+alpha_t*t); denom=denom if denom!=0 else 1e-9
            mu=[(beta*math.log(B_t*uu+eps)/denom) for uu in u]
            eHTT=[0.0]*N
            for k in range(M):
                if T[k]==0: continue
                wk=Jcoeffs[k]
                for i in H_rows[k]: eHTT[i]+=wk
            qcal=[2.0*eHTT[i]-Csum[i] for i in range(N)]
            cand=[V[idx] for idx in range(mV) if qcal[V[idx]]<mu[idx]]
            if cand:
                v=rng.choice(cand); s[v]=-s[v]
                for k in var_clauses[v]:
                    prod_cache[k]=-prod_cache[k]
                    T[k]=1 if (1-prod_cache[k])//2==1 else 0
            t+=1.0
        curE=eval_spin_energy(s,H_rows,Jcoeffs,const_term)
        if curE<bestE-1e-12: bestE=curE; best_s=s.copy()
    return best_s,bestE
# --------------------------- Validation Helpers ---------------------------
def residual_norm(cons):
    """Compute the maximum absolute residual across all constraint types."""
    import math
    r1 = max((abs(v) for v in cons["onehot_res"].values()), default=0.0)
    r2 = max((abs(v) for v in cons["feeder_res"].values()), default=0.0)
    r3 = max((abs(v) for v in cons["budget_res"].values()), default=0.0)
    return max(r1, r2, r3)

def validate_solution_quality(run_results, config):
    """
    Validate that the solution meets the configuration requirements.
    Returns (passed, issues) where issues is a list of validation problems.
    """
    issues = []
    checks = config["checks"]
    
    # Check PUBO == Spin consistency
    if checks["require_pubo_equals_spin"] and not run_results["pubo_spin_equal"]:
        issues.append(f"PUBO==Spin check failed: {run_results['pubo_spin_info']}")
    
    # Check zero residuals
    if checks["require_zero_residuals"]:
        atol = checks["absolute_tolerance"]
        rtol = checks["relative_tolerance"]
        
        # Check one-hot residuals
        for k, v in run_results["onehot_res_det"].items():
            if abs(v) > atol:
                issues.append(f"One-hot residual violation: {k}={v}")
        
        # Check feeder residuals
        for k, v in run_results["feeder_res_det"].items():
            if abs(v) > atol:
                issues.append(f"Feeder residual violation: {k}={v}")
        
        # Check budget residuals
        for k, v in run_results["budget_res_det"].items():
            if abs(v) > atol:
                issues.append(f"Budget residual violation: {k}={v}")
    
    return len(issues) == 0, issues

def compute_scale_aware_penalties(inst, weights):
    """
    Compute scale-aware penalty bounds to ensure constraints dominate soft terms.
    Returns recommended kappa_eq and kappa_1h values.
    """
    decision_idx = inst["decision_idx"]
    y_nom = inst["y_nominal"]
    u = inst["u"]
    T = inst["T"]
    A = inst["num_agents"]
    options = inst["options_per_agent"]
    
    gamma = weights.get("gamma", 1.0)
    kappa_fair = weights.get("kappa_fair", 0.2)
    kappa_sw = weights.get("kappa_sw", 0.5)
    
    # U_max: maximum utility gain
    U_max = sum(gamma * abs(ui) for ui in u)
    
    # C_nom_max: maximum nominal system cost
    C_nom_max = 0.0
    for t in range(T):
        idxs_t = [idx for idx, meta in decision_idx.items() if meta["t"] == t]
        sum_y_t = sum(abs(y_nom[i]) for i in idxs_t)
        C_nom_max += 0.5 * sum_y_t * sum_y_t
    
    # F_max: maximum fairness penalty (conservative bound)
    F_max = kappa_fair * A * T * options * options  # diagonal terms
    
    # S_max: maximum switching penalty
    S_max = kappa_sw * A * (T-1) * options * 2  # adjacent pairs
    
    # Scale-aware bounds
    kappa_eq_recommended = 10.0 * (U_max + C_nom_max + F_max + S_max)
    kappa_1h_recommended = 10.0 * gamma * max(abs(ui) for ui in u)
    
    return {
        "kappa_eq_recommended": kappa_eq_recommended,
        "kappa_1h_recommended": kappa_1h_recommended,
        "U_max": U_max,
        "C_nom_max": C_nom_max,
        "F_max": F_max,
        "S_max": S_max
    }

def check_constraints_on_state(b_vec, inst, meta):
    """
    Compute residuals of constraints for a given binary state b_vec (length N_total).
    Returns dict with residual arrays:
      - onehot_res[(a,t)]: sum_o b_{a,t,o} - 1
      - feeder_res[(r,t)]: sum_{i in t} y_r[i] b_i + S_{r,t} - cap
      - budget_res[a]:     sum_{i in agent a} y_nom[i] b_i + S_a - E_a[a]
    """
    decision_idx = inst["decision_idx"]
    y_nom = inst["y_nominal"]
    scenarios = inst["scenarios"]
    T = inst["T"]
    A = inst["num_agents"]
    options = inst["options_per_agent"]
    slack_indices = meta["slack_indices"]
    slack_specs = meta["slack_specs"]

    # invert helper
    ato_to_idx = {(meta_i["agent"], meta_i["t"], meta_i["opt"]): idx
                  for idx, meta_i in decision_idx.items()}
    def b_index(a,t,o): return ato_to_idx[(a,t,o)]

    # one-hot residuals
    onehot_res = {}
    if options > 1:
        for a in range(A):
            for t in range(T):
                ssum = 0.0
                for o in range(options):
                    ssum += b_vec[b_index(a,t,o)]
                onehot_res[(a,t)] = ssum - 1.0

    # decode slack bits
    slack_val = {}
    for (name, cap, K, kind, info) in slack_specs:
        S = 0
        for k in range(K):
            S += (1<<k) * b_vec[slack_indices[(name,k)]]
        slack_val[name] = (S, cap, kind, info)

    # feeder residuals per scenario/time
    feeder_res = {}
    for (name, cap, K, kind, info) in slack_specs:
        if kind != "feeder": continue
        r = info["r"]; t = info["t"]
        y_r = scenarios[r]["y"]
        ssum = 0.0
        for idx, meta_i in decision_idx.items():
            if meta_i["t"] == t:
                ssum += y_r[idx] * b_vec[idx]
        S, cap0, _, _ = slack_val[name]
        feeder_res[(r,t)] = ssum + S - cap0

    # budget residuals per agent
    budget_res = {}
    for (name, cap, K, kind, info) in slack_specs:
        if kind != "budget": continue
        a = info["a"]
        ssum = 0.0
        for idx, meta_i in decision_idx.items():
            if meta_i["agent"] == a:
                ssum += y_nom[idx] * b_vec[idx]
        S, cap0, _, _ = slack_val[name]
        budget_res[a] = ssum + S - cap0

    return {"onehot_res": onehot_res, "feeder_res": feeder_res, "budget_res": budget_res}

def pubo_spin_consistency_check(PUBO, spin_mono, trials=50, seed=1,
                                atol=1e-8, rtol=1e-9):
    """
    Random s, map to b=(1+s)/2, compare eval_spin == eval_pubo
    using a robust absolute+relative tolerance:
        |E_spin - E_pubo| <= atol + rtol * max(|E_spin|, |E_pubo|)
    Returns (ok, info_if_fail).
    """
    N = 0
    for S in PUBO.keys():
        if S: N = max(N, max(S)+1)
    for S in spin_mono.keys():
        if S: N = max(N, max(S)+1)
    if N == 0:
        return True, {}

    H_rows, J, const = spin_to_Hrows(spin_mono)
    rng = random.Random(seed)

    worst = {"idx": None, "E_spin": None, "E_pubo": None, "absdiff": -1.0}
    for t in range(trials):
        s = [rng.choice([-1,1]) for _ in range(N)]
        b = [(1 if x==1 else 0) for x in s]
        E_spin = eval_spin_energy(s, H_rows, J, const)
        E_pubo = eval_pubo_binary(PUBO, b)
        diff = abs(E_spin - E_pubo)
        tol = atol + rtol * max(abs(E_spin), abs(E_pubo))
        if diff > worst["absdiff"]:
            worst = {"idx": t, "E_spin": E_spin, "E_pubo": E_pubo, "absdiff": diff, "tol": tol}
        if diff > tol:
            return False, worst
    return True, worst

# --------------------------- Experiment Runner ----------------------------
def run_phase4_with_config(outdir, config_name="small_debug", sizes=None, trials=2, seed0=1234):
    """
    Run Phase-4 with a specific configuration profile.
    """
    if config_name not in CONFIG_PROFILES:
        raise ValueError(f"Unknown config profile: {config_name}. Available: {list(CONFIG_PROFILES.keys())}")
    
    config = CONFIG_PROFILES[config_name]
    
    # Use config defaults if sizes not specified
    if sizes is None:
        sizes = (config["instance"]["num_agents"],)
    
    os.makedirs(outdir, exist_ok=True)
    summary = {
        "timestamp": now_str(), 
        "outdir": outdir, 
        "runs": [], 
        "config_name": config_name,
        "config": config
    }

    # Extract parameters from config
    scaling = config["scaling"]
    instance_params = config["instance"]
    penalties = config["penalties"]
    solver_params = config["solver"]
    feasibility_config = config["feasibility_phase"]

    for size in sizes:
        for trial in range(trials):
            seed = seed0 + size*100 + trial
            
            # Generate instance with config parameters
            inst = generate_dr_instance(
                num_agents=size,
                T=instance_params["T"],
                options_per_agent=instance_params["options_per_agent"],
                seed=seed,
                scenario_count=instance_params["scenario_count"],
                include_robustness=True,
                quantize_delta=scaling["delta"],
                robust_perturbation_factor_min=instance_params["robust_perturbation_factor_min"],
                robust_perturbation_factor_max=instance_params["robust_perturbation_factor_max"]
            )
            
            # Auto-scale penalties based on instance characteristics
            penalty_bounds = compute_scale_aware_penalties(inst, penalties)
            w_scaled = dict(penalties)
            w_scaled["kappa_eq"] = max(penalties["kappa_eq"], penalty_bounds["kappa_eq_recommended"])
            w_scaled["kappa_1h"] = max(penalties["kappa_1h"], penalty_bounds["kappa_1h_recommended"])

            # --- Feasibility phase (escalate penalties until residuals==0) ---
            feasible_state = None
            if feasibility_config["enabled"]:
                scales = feasibility_config["penalty_scales"]
                for sc in scales:
                    w_feas = dict(w_scaled)
                    w_feas["kappa_eq"] = w_scaled["kappa_eq"] * sc
                    w_feas["kappa_1h"] = w_scaled["kappa_1h"] * sc

                    PUBO_f, meta_f = build_pubo(inst, w_feas)
                spin_f = pubo_to_spin(PUBO_f)
                H_rows_f, J_f, c_f = spin_to_Hrows(spin_f)

                # deterministic BR to push into feasibility
                s_f, E_f, _ = deterministic_BR(H_rows_f, J_f, c_f, max_steps=solver_params["deterministic_max_steps"], seed=seed+31)
                b_f = b_from_s(s_f)
                cons_f = check_constraints_on_state(b_f, inst, meta_f)
                if residual_norm(cons_f) == 0.0:
                    feasible_state = b_f
                    break

            if feasible_state is None:
                # If still infeasible here, something is off (e.g., one-hot with options=1).
                # Use ultra-conservative fallback
                feasible_state = [0]*meta_f["N_total"]

            # Build intended (target) PUBO and spin once
            PUBO, meta = build_pubo(inst, w_scaled)
            spin = pubo_to_spin(PUBO)

            # PUBO <-> Spin sanity check with config tolerances
            ok_ps, info_ps = pubo_spin_consistency_check(
                PUBO, spin, trials=40, seed=seed+7,
                atol=config["checks"]["absolute_tolerance"],
                rtol=config["checks"]["relative_tolerance"]
            )

            # Build HO-Ising arrays
            H_rows, Jcoeffs, const_term = spin_to_Hrows(spin)
            N_total = meta["N_total"]

            # Coloring
            color_groups = greedy_coloring(H_rows, N_total)

            # Local-field identity (informational only)
            lf_ok, lf_info = lf_identity_check(H_rows, Jcoeffs, const_term, trials=200, seed=seed+11)

            # Build an initial spin from feasible_state
            s_init = [1 if feasible_state[i]==1 else -1 for i in range(meta["N_total"])]

            # Multi-start deterministic BR
            best_det_s = None
            best_det_E = float('inf')
            best_det_hist = None
            
            for restart in range(solver_params["multi_start_restarts"]):
                restart_seed = seed + 13 + restart * 100
                s_det_curr, E_det_curr, hist_det_curr = deterministic_BR(
                    H_rows, Jcoeffs, const_term, 
                    max_steps=solver_params["deterministic_max_steps"], 
                    seed=restart_seed
                )
                if E_det_curr < best_det_E:
                    best_det_s = s_det_curr
                    best_det_E = E_det_curr
                    best_det_hist = hist_det_curr
            
            s_det, E_det, hist_det = best_det_s, best_det_E, best_det_hist
            b_det = b_from_s(s_det)
            cons_det = check_constraints_on_state(b_det, inst, meta)

            # If deterministic didn't preserve feasibility (shouldn't happen with large penalties), fall back to s_init
            if residual_norm(cons_det) != 0.0:
                s_det = s_init[:]
                s_det, E_det, hist_det = deterministic_BR(
                    H_rows, Jcoeffs, const_term, 
                    max_steps=solver_params["deterministic_max_steps"], 
                    seed=seed+13
                )
                b_det = b_from_s(s_det)
                cons_det = check_constraints_on_state(b_det, inst, meta)

            # Stochastic Algorithm-2 (warm-start from deterministic with config parameters)
            s_sto, E_sto = algorithm2_stochastic(
                H_rows, Jcoeffs, const_term, color_groups,
                max_iters=solver_params["stochastic_max_iters"], 
                seed=seed+17,
                alpha0=solver_params["stochastic_alpha0"], 
                beta=solver_params["stochastic_beta"], 
                B0=solver_params["stochastic_B0"], 
                eps=solver_params["stochastic_eps"],
                s_init=s_det if solver_params["warm_start_from_deterministic"] else None
            )
            b_sto = b_from_s(s_sto)
            cons_sto = check_constraints_on_state(b_sto, inst, meta)

            # Validate solution quality
            run_data = {
                "pubo_spin_equal": ok_ps,
                "pubo_spin_info": info_ps,
                "onehot_res_det": cons_det["onehot_res"],
                "feeder_res_det": cons_det["feeder_res"],
                "budget_res_det": cons_det["budget_res"]
            }
            validation_passed, validation_issues = validate_solution_quality(run_data, config)
            
            run = {
                "size": size, "trial": trial, "seed": seed,
                "N_dec": meta["N_dec"], "N_total": meta["N_total"], "M": len(H_rows),
                "pubo_spin_equal": ok_ps, "pubo_spin_info": info_ps,
                "lf_ok": lf_ok, "lf_info": lf_info,
                "det_E": E_det, "det_hist_len": len(hist_det),
                "stoch_E": E_sto,
                "onehot_res_det": cons_det["onehot_res"],
                "feeder_res_det": {str(k): v for k,v in cons_det["feeder_res"].items()},
                "budget_res_det": cons_det["budget_res"],
                "onehot_res_stoch": cons_sto["onehot_res"],
                "feeder_res_stoch": {str(k): v for k,v in cons_sto["feeder_res"].items()},
                "budget_res_stoch": cons_sto["budget_res"],
                "color_groups": [len(g) for g in color_groups],
                "penalty_bounds": penalty_bounds,
                "final_weights": w_scaled,
                "validation_passed": validation_passed,
                "validation_issues": validation_issues
            }
            summary["runs"].append(run)

            base = os.path.join(outdir, f"phase4_size{size}_trial{trial}_seed{seed}")
            write_json(base+"_inst.json", inst)
            write_json(base+"_meta.json", meta)
            write_json(base+"_pubo_terms.json", {"num_terms": len(PUBO)})
            write_json(base+"_spin_terms.json", {"num_terms": len(spin)})
            write_json(base+"_results.json", run)

            print(f"[Size {size} Trial {trial}] detE={E_det:.6g}, stochE={E_sto:.6g}, "
                  f"PUBO==Spin:{ok_ps}, LFok:{lf_ok}, M={len(H_rows)}")

    write_json(os.path.join(outdir, "phase4_summary.json"), summary)
    return summary

# --------------------------- CLI ------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase-4 STRICT DR→PUBO→Spin→HO-Ising pipeline")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="output directory")
    parser.add_argument("--config", type=str, default="small_debug", 
                        choices=list(CONFIG_PROFILES.keys()),
                        help="configuration profile to use")
    parser.add_argument("--sizes", type=int, nargs="+", help="agent sizes to run (overrides config default)")
    parser.add_argument("--trials", type=int, default=2, help="trials per size")
    parser.add_argument("--seed", type=int, default=1234, help="base RNG seed")
    parser.add_argument("--list-configs", action="store_true", help="list available configuration profiles")
    args = parser.parse_args()

    if args.list_configs:
        print("Available configuration profiles:")
        for name, config in CONFIG_PROFILES.items():
            print(f"  {name}: {config['description']}")
        return

    summary = run_phase4_with_config(
        outdir=args.outdir, 
        config_name=args.config, 
        sizes=tuple(args.sizes) if args.sizes else None,
        trials=args.trials,
        seed0=args.seed
    )
    print("Phase-4 STRICT complete. Summary written to:",
          os.path.join(args.outdir, "phase4_summary.json"))

if __name__ == "__main__":
    main()
