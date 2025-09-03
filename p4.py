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
    ,
    "population_500": {
        "description": "Hybrid population warm-start + structured HO-Ising BR for large A.",
        "use_structured": True,
        "scaling": {"delta": 0.1},
        "instance": {
            "num_agents": 500, "T": 4, "options_per_agent": 2, "scenario_count": 2,
            "robust_perturbation_factor_min": 0.9, "robust_perturbation_factor_max": 1.1
        },
        "penalties": {
            "gamma": 1.0, "kappa_eq": 0.0, "kappa_1h": 0.0,
            "kappa_fair": 0.2, "kappa_sw": 0.2
        },
        "solver": {
            "deterministic_max_steps": 50000,
            "stochastic_max_iters": 0,
            "stochastic_alpha0": 0.0, "stochastic_beta": 0.0, "stochastic_B0": 0.0,
            "stochastic_eps": 1e-12, "warm_start_from_deterministic": False, "multi_start_restarts": 1
        },
        "hybrid": {
            "replicator_iters": 20,
            "replicator_eta": 0.1,
            "temperature": 1.0,
            "sample_candidates": 8
        },
        "feasibility_phase": {
            "enabled": False, "penalty_scales": [1.0], "stop_when_residual_zero": True
        },
        "checks": {
            "require_pubo_equals_spin": False, "require_zero_residuals": True,
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
                         include_robustness=True,
                         quantize_delta=0.1,
                         robust_perturbation_factor_min=0.85,
                         robust_perturbation_factor_max=1.15):
    """
    Generate DR instance with quantization to ensure integer compatibility.
    quantize_delta: base unit for quantization (e.g., 0.1 kWh)
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
    quantize_delta = float(inst.get("quantize_delta", 1.0))

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
    # Feeder caps (use float cap in units of quantize_delta; K based on tick capacity)
    for r, sc in enumerate(scenarios):
        for t in range(T):
            cap_val = float(sc["P"][t])
            cap_ticks = max(0, int(round(cap_val / quantize_delta)))
            K = max(1, math.ceil(math.log2(cap_ticks + 1)))
            name = f"slack_feed_r{r}_t{t}"
            slack_specs.append((name, cap_val, K, "feeder", {"r": r, "t": t}))
            for k in range(K):
                slack_indices[(name,k)] = cur; cur += 1
    # Budgets
    for a in range(A):
        cap_val = float(E_a[a])
        cap_ticks = max(0, int(round(cap_val / quantize_delta)))
        K = max(1, math.ceil(math.log2(cap_ticks + 1)))
        name = f"slack_budget_a{a}"
        slack_specs.append((name, cap_val, K, "budget", {"a": a}))
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
                # slack bits represent multiples of quantize_delta
                lin.append((slack_indices[(name,k)], float((1<<k) * quantize_delta)))
            add_squared_equality(lin, cap, kappa_eq)
        elif kind == "budget":
            a = info["a"]
            for idx,meta in decision_idx.items():
                if meta["agent"] == a:
                    lin.append((idx, y_nom[idx]))
            for k in range(K):
                # slack bits represent multiples of quantize_delta
                lin.append((slack_indices[(name,k)], float((1<<k) * quantize_delta)))
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
def deterministic_BR(H_rows,Jcoeffs,const_term,max_steps=2000,seed=0,s_init=None,use_local=False):
    N=max((max(row) if row else -1) for row in H_rows)+1 if H_rows else 0
    rng=random.Random(seed)
    s=s_init[:] if (s_init is not None) else [rng.choice([-1,1]) for _ in range(N)]
    hist=[]
    var_clauses=None
    if use_local:
        var_clauses=build_var_clauses(H_rows,N)
    for step in range(max_steps):
        improved=False
        order=list(range(N)); rng.shuffle(order)
        for i in order:
            dE = (delta_E_localfield(s,i,H_rows,Jcoeffs,var_clauses) if use_local
                  else delta_E_direct(s,i,H_rows,Jcoeffs,const_term))
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
    quantize_delta = float(inst.get("quantize_delta", 1.0))

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
        S_ticks = 0
        for k in range(K):
            S_ticks += (1<<k) * b_vec[slack_indices[(name,k)]]
        slack_val[name] = (S_ticks, cap, kind, info)

    # feeder residuals per scenario/time
    feeder_res = {}
    for (name, cap, K, kind, info) in slack_specs:
        if kind != "feeder": continue
        r = info["r"]; t = info["t"]
        y_r = scenarios[r]["y"]
        # work in integer ticks to avoid float drift
        ssum_ticks = 0
        for idx, meta_i in decision_idx.items():
            if meta_i["t"] == t:
                ssum_ticks += int(round(y_r[idx] / quantize_delta)) * b_vec[idx]
        S_ticks, cap0, _, _ = slack_val[name]
        cap_ticks = int(round(cap0 / quantize_delta))
        resid_ticks = ssum_ticks + S_ticks - cap_ticks
        feeder_res[(r,t)] = resid_ticks * quantize_delta

    # budget residuals per agent
    budget_res = {}
    for (name, cap, K, kind, info) in slack_specs:
        if kind != "budget": continue
        a = info["a"]
        ssum_ticks = 0
        for idx, meta_i in decision_idx.items():
            if meta_i["agent"] == a:
                ssum_ticks += int(round(y_nom[idx] / quantize_delta)) * b_vec[idx]
        S_ticks, cap0, _, _ = slack_val[name]
        cap_ticks = int(round(cap0 / quantize_delta))
        resid_ticks = ssum_ticks + S_ticks - cap_ticks
        budget_res[a] = resid_ticks * quantize_delta

    return {"onehot_res": onehot_res, "feeder_res": feeder_res, "budget_res": budget_res}

def construct_trivially_feasible_state(inst, meta):
    """
    Build a strictly feasible binary vector by:
      - selecting the null option (o=0) for every (agent, t) to satisfy one-hot exactly,
      - setting slack bits to exactly meet feeder/budget equalities in integer ticks.
    This guarantees zero residuals by construction.
    """
    decision_idx = inst["decision_idx"]
    T = inst["T"]
    A = inst["num_agents"]
    options = inst["options_per_agent"]
    quantize_delta = float(inst.get("quantize_delta", 1.0))
    scenarios = inst["scenarios"]

    N_total = meta["N_total"]
    b = [0]*N_total

    ato_to_idx = {(meta_i["agent"], meta_i["t"], meta_i["opt"]): idx
                  for idx, meta_i in decision_idx.items()}
    def b_index(a,t,o): return ato_to_idx[(a,t,o)]

    # Pick null option for each (a,t)
    if options > 0:
        for a in range(A):
            for t in range(T):
                b[b_index(a,t,0)] = 1

    # Set slack bits per equality exactly
    for (name, cap, K, kind, info) in meta["slack_specs"]:
        if kind == "feeder":
            r = info["r"]; t = info["t"]
            y_r = scenarios[r]["y"]
            ssum_ticks = 0
            for idx, meta_i in decision_idx.items():
                if meta_i["t"] == t:
                    ssum_ticks += int(round(y_r[idx] / quantize_delta)) * b[idx]
            cap_ticks = int(round(float(cap) / quantize_delta))
            S_ticks = max(0, cap_ticks - ssum_ticks)
        elif kind == "budget":
            a = info["a"]
            ssum_ticks = 0
            for idx, meta_i in decision_idx.items():
                if meta_i["agent"] == a:
                    ssum_ticks += int(round(inst["y_nominal"][idx] / quantize_delta)) * b[idx]
            cap_ticks = int(round(float(cap) / quantize_delta))
            S_ticks = max(0, cap_ticks - ssum_ticks)
        else:
            continue
        for k in range(K):
            bit = (S_ticks >> k) & 1
            b[meta["slack_indices"][(name,k)]] = bit
    return b

def decode_slacks(b_vec, inst, meta):
    """Decode slack bits into integer ticks and float values for each equality."""
    quantize_delta = float(inst.get("quantize_delta", 1.0))
    decoded = {}
    for (name, cap, K, kind, info) in meta["slack_specs"]:
        ticks = 0
        for k in range(K):
            ticks += (1<<k) * b_vec[meta["slack_indices"][(name,k)]]
        decoded[name] = {
            "ticks": int(ticks),
            "value": float(ticks * quantize_delta),
            "cap": float(cap),
            "kind": kind,
            "info": info
        }
    return decoded

def chosen_options_by_agent_time(b_vec, inst):
    """Return mapping (a,t)->chosen option o from the one-hot block (assumes feasibility)."""
    decision_idx = inst["decision_idx"]
    T = inst["T"]
    A = inst["num_agents"]
    options = inst["options_per_agent"]
    ato_to_idx = {(meta_i["agent"], meta_i["t"], meta_i["opt"]): idx
                  for idx, meta_i in decision_idx.items()}
    def b_index(a,t,o): return ato_to_idx[(a,t,o)]
    chosen = {}
    for a in range(A):
        for t in range(T):
            co = None
            for o in range(options):
                if b_vec[b_index(a,t,o)] == 1:
                    co = o; break
            chosen[(a,t)] = co
    return chosen

def per_time_loads(b_vec, inst):
    """Compute per-time loads for each scenario r: dict r->list over t of total load."""
    decision_idx = inst["decision_idx"]
    T = inst["T"]
    scenarios = inst["scenarios"]
    loads = {}
    for r, sc in enumerate(scenarios):
        y_r = sc["y"]
        L = [0.0]*T
        for t in range(T):
            ssum = 0.0
            for idx, meta_i in decision_idx.items():
                if meta_i["t"] == t:
                    ssum += y_r[idx] * b_vec[idx]
            L[t] = float(ssum)
        loads[r] = L
    return loads

def per_time_reductions_from_caps(decoded_slacks):
    """From decoded feeder slacks, build dict r->list over t of reductions (slack values)."""
    # names are of form slack_feed_r{r}_t{t}
    per_r = {}
    for name, info in decoded_slacks.items():
        if info.get("kind") != "feeder":
            continue
        r = info["info"]["r"]; t = info["info"]["t"]
        per_r.setdefault(r, {})[t] = info["value"]
    # convert inner dicts to lists by t order
    out = {}
    for r, tmap in per_r.items():
        max_t = max(tmap.keys()) if tmap else -1
        out[r] = [float(tmap.get(t, 0.0)) for t in range(max_t+1)]
    return out

def per_agent_loads_and_reductions(b_vec, inst, decoded_slacks):
    """Return per-agent load and reduction (budget slack value)."""
    decision_idx = inst["decision_idx"]
    y_nom = inst["y_nominal"]
    A = inst["num_agents"]
    E_a = inst["E_a"]
    loads = {a: 0.0 for a in range(A)}
    for idx, meta_i in decision_idx.items():
        a = meta_i["agent"]
        loads[a] += y_nom[idx] * b_vec[idx]
    # budget slacks by agent
    reductions = {a: 0.0 for a in range(A)}
    for name, info in decoded_slacks.items():
        if info.get("kind") == "budget":
            a = info["info"]["a"]
            reductions[a] = float(info["value"])
    # sanity: E_a[a] - loads[a] == reductions[a] when feasible
    return loads, reductions

# --------------------------- Structured Large-Scale Path ------------------
def build_x_model(inst):
    """Build arrays for x-formulation: one binary x[a,t]=1 means choose active option (o=1)."""
    A = inst["num_agents"]; T = inst["T"]; options = inst["options_per_agent"]
    decision_idx = inst["decision_idx"]; y_nom = inst["y_nominal"]; u = inst["u"]
    ato_to_idx = {(m["agent"], m["t"], m["opt"]): i for i, m in decision_idx.items()}
    y_act = [[0.0]*T for _ in range(A)]
    u0 = [[0.0]*T for _ in range(A)]
    u1 = [[0.0]*T for _ in range(A)]
    for a in range(A):
        for t in range(T):
            if options > 1:
                i0 = ato_to_idx[(a,t,0)]; i1 = ato_to_idx[(a,t,1)]
                y_act[a][t] = float(y_nom[i1])
                u0[a][t] = float(u[i0])
                u1[a][t] = float(u[i1])
            else:
                i0 = ato_to_idx[(a,t,0)]
                y_act[a][t] = 0.0
                u0[a][t] = float(u[i0])
                u1[a][t] = float(u[i0])
    E_a = {int(k): float(v) for k,v in inst["E_a"].items()} if isinstance(inst["E_a"], dict) else {a: float(inst["E_a"][a]) for a in range(A)}
    quant = float(inst.get("quantize_delta", 1.0))
    return {"A": A, "T": T, "y_act": y_act, "u0": u0, "u1": u1, "E_a": E_a, "quant": quant}

def structured_energy(x, model, weights):
    A = model["A"]; T = model["T"]; y_act = model["y_act"]; u0 = model["u0"]; u1 = model["u1"]
    gamma = weights.get("gamma", 1.0)
    kappa_fair = weights.get("kappa_fair", 0.0)
    kappa_sw = weights.get("kappa_sw", 0.0)
    # Build per-time loads
    S = [0.0]*T
    for t in range(T):
        s = 0.0
        for a in range(A): s += y_act[a][t] * x[a][t]
        S[t] = s
    # System cost
    cost = 0.0
    for t in range(T): cost += 0.5 * S[t] * S[t]
    # Utility
    util = 0.0
    for a in range(A):
        for t in range(T):
            util += (u0[a][t] + (u1[a][t]-u0[a][t]) * x[a][t])
    util_term = -gamma * util
    # Fairness
    pi = [0]*A
    for a in range(A):
        sa = 0
        for t in range(T): sa += x[a][t]
        pi[a] = T - sa
    P = sum(pi)
    fair = kappa_fair * ( sum(p*p for p in pi) - (1.0/A) * (P*P) )
    # Switching
    sw = 0.0
    if kappa_sw != 0.0:
        for a in range(A):
            for t in range(1,T):
                sw += 2.0 * kappa_sw * (x[a][t] + x[a][t-1] - 2*x[a][t]*x[a][t-1])
    return cost + util_term + fair + sw

def structured_delta_flip(x, a, t, model, weights, caches):
    # caches: S[t], pi[a], P
    A = model["A"]; T = model["T"]; y_act = model["y_act"]; u0 = model["u0"]; u1 = model["u1"]
    gamma = weights.get("gamma", 1.0)
    kappa_fair = weights.get("kappa_fair", 0.0)
    kappa_sw = weights.get("kappa_sw", 0.0)
    x_old = x[a][t]
    x_new = 1 - x_old
    flip_dir = 1 if x_old == 0 else -1
    # System cost
    S_t = caches["S"][t]
    y = y_act[a][t]
    deltaS = y * flip_dir
    d_cost = 0.5 * ( (S_t + deltaS)**2 - S_t**2 )
    # Utility
    du = (u1[a][t] - u0[a][t])
    d_util = -gamma * du * flip_dir
    # Fairness
    pi_a = caches["pi"][a]
    Psum = caches["P"]
    D = -flip_dir  # change in pi_a
    d_fair = kappa_fair * ( ((pi_a + D)**2 - pi_a**2) - (1.0/A)*((Psum + D)**2 - Psum**2) )
    # Switching
    d_sw = 0.0
    if kappa_sw != 0.0:
        if t-1 >= 0:
            n = x[a][t-1]
            old = 2.0*kappa_sw*(x_old + n - 2*x_old*n)
            new = 2.0*kappa_sw*(x_new + n - 2*x_new*n)
            d_sw += (new - old)
        if t+1 < T:
            n = x[a][t+1]
            old = 2.0*kappa_sw*(x_old + n - 2*x_old*n)
            new = 2.0*kappa_sw*(x_new + n - 2*x_new*n)
            d_sw += (new - old)
    return d_cost + d_util + d_fair + d_sw

def deterministic_BR_structured(model, weights, x_init, max_steps=10000, seed=0, return_hist=False):
    import random
    A = model["A"]; T = model["T"]; y_act = model["y_act"]
    rng = random.Random(seed)
    x = [row[:] for row in x_init]
    # caches
    S = [0.0]*T
    for t in range(T):
        S[t] = sum(y_act[a][t]*x[a][t] for a in range(A))
    pi = [0]*A
    for a in range(A): pi[a] = T - sum(x[a][t] for t in range(T))
    P = sum(pi)
    caches = {"S": S, "pi": pi, "P": P}
    E_cur = structured_energy(x, model, weights)
    hist_E = []
    # iterate
    for step in range(max_steps):
        improved = False
        indices = [(a,t) for a in range(A) for t in range(T)]
        rng.shuffle(indices)
        for a,t in indices:
            dE = structured_delta_flip(x, a, t, model, weights, caches)
            if dE < -1e-12:
                # flip and update caches
                x_old = x[a][t]
                x_new = 1 - x_old
                x[a][t] = x_new
                # S
                flip_dir = 1 if x_old == 0 else -1
                caches["S"][t] += model["y_act"][a][t] * flip_dir
                # fairness caches
                D = -flip_dir
                caches["pi"][a] += D
                caches["P"] += D
                E_cur += dE
                if return_hist:
                    hist_E.append(E_cur)
                improved = True
                break
        if not improved:
            break
    if return_hist and not hist_E:
        hist_E.append(E_cur)
    return (x, (E_cur, hist_E)) if return_hist else (x, E_cur)

# --------------------------- Visualization & Export -----------------------
def visualize_and_export(inst, meta, run, basepath):
    import os, csv
    # Prepare data
    A = int(inst["num_agents"]); T = int(inst["T"])
    # chosen options matrix
    chosen = run.get("chosen_opt_det", {})
    opt_mat = [[0]*T for _ in range(A)]
    for k, v in chosen.items():
        # keys are like "(a, t)"
        if isinstance(k, str) and k.startswith("("):
            try:
                parts = k.strip("() ").split(",")
                a = int(parts[0].strip()); t = int(parts[1].strip())
            except Exception:
                continue
        elif isinstance(k, (tuple, list)):
            a, t = int(k[0]), int(k[1])
        else:
            continue
        if 0 <= a < A and 0 <= t < T:
            opt_mat[a][t] = int(v if v is not None else 0)

    # Per-time loads and caps (use r0 if present)
    loads_det = run.get("per_time_loads_det", {})
    loads_r0 = loads_det.get("r0") or next(iter(loads_det.values()), [0.0]*T)
    caps_r0 = inst["scenarios"][0]["P"] if inst.get("scenarios") else [0.0]*T
    red_det = run.get("per_time_reduction_from_cap_det", {})
    red_r0 = red_det.get("r0") or [max(0.0, float(caps_r0[t]) - float(loads_r0[t])) for t in range(T)]

    # Per-agent reductions
    red_by_agent = run.get("per_agent_reduction_det", {})
    if isinstance(red_by_agent, dict):
        items = []
        for k, v in red_by_agent.items():
            try:
                a = int(k)
            except Exception:
                continue
            items.append((a, float(v)))
        items.sort(key=lambda x: x[0])
        ag_ids = [a for a,_ in items]
        ag_vals = [v for _,v in items]
    else:
        ag_ids = list(range(len(red_by_agent)))
        ag_vals = [float(v) for v in red_by_agent]

    # CSV exports
    with open(basepath + "_per_agent_reduction.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["agent", "reduction"])
        for a, v in zip(ag_ids, ag_vals): w.writerow([a, v])
    with open(basepath + "_per_time_loads_r0.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "load", "cap", "reduction"])
        for t in range(T): w.writerow([t, float(loads_r0[t]), float(caps_r0[t]), float(red_r0[t])])
    with open(basepath + "_chosen_options.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["agent", "t", "option"])
        for a in range(A):
            for t in range(T): w.writerow([a, t, opt_mat[a][t]])

    # Plots (optional)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.plot(range(T), [float(x) for x in loads_r0], marker='o', label='Load r0')
        plt.plot(range(T), [float(x) for x in caps_r0], marker='s', label='Cap r0')
        plt.bar(range(T), [float(x) for x in red_r0], alpha=0.3, label='Reduction r0')
        plt.xlabel('Time t'); plt.ylabel('Units'); plt.title('Per-time Load vs Cap (r0)'); plt.legend(); plt.tight_layout()
        plt.savefig(basepath + "_loads_caps_r0.png"); plt.close()

        plt.figure(figsize=(8,4))
        plt.hist(ag_vals, bins=30, edgecolor='k', alpha=0.7)
        plt.xlabel('Reduction'); plt.ylabel('Count of agents'); plt.title('Per-agent reductions'); plt.tight_layout()
        plt.savefig(basepath + "_reduction_hist.png"); plt.close()

        # Option counts per time
        counts1 = [sum(opt_mat[a][t] for a in range(A)) for t in range(T)]
        plt.figure(figsize=(6,4))
        plt.bar(range(T), counts1)
        plt.xlabel('Time t'); plt.ylabel('#Agents choosing active'); plt.title('Active option counts by time'); plt.tight_layout()
        plt.savefig(basepath + "_option_counts.png"); plt.close()

        # Heatmap sample (first N agents)
        N = min(A, 100)
        import numpy as np
        H = np.array([opt_mat[a] for a in range(N)])
        plt.figure(figsize=(min(10, T*1.2), 8))
        plt.imshow(H, aspect='auto', interpolation='nearest', cmap='viridis')
        plt.colorbar(label='option (0=null,1=active)')
        plt.xlabel('Time t'); plt.ylabel('Agent (sample)'); plt.title('Chosen options heatmap (sample)'); plt.tight_layout()
        plt.savefig(basepath + "_heatmap_sample.png"); plt.close()

        # Energy trace if available
        histE = run.get("det_hist_E")
        if histE:
            plt.figure(figsize=(7,4))
            plt.plot(range(1, len(histE)+1), histE, marker='.')
            plt.xlabel('Accepted flips'); plt.ylabel('Energy'); plt.title('Deterministic BR energy trace'); plt.tight_layout()
            plt.savefig(basepath + "_det_energy_trace.png"); plt.close()
    except Exception as e:
        # Matplotlib may be unavailable; plots are optional
        pass

def replicator_warmstart(model, weights, iters=20, eta=0.1, temperature=1.0, samples=8, seed=0):
    import random, math
    A = model["A"]; T = model["T"]; y_act = model["y_act"]; u0 = model["u0"]; u1 = model["u1"]
    gamma = weights.get("gamma", 1.0)
    kappa_fair = weights.get("kappa_fair", 0.0)
    kappa_sw = weights.get("kappa_sw", 0.0)
    rng = random.Random(seed)
    # probabilities p[a][t]
    p = [[0.1 for _ in range(T)] for __ in range(A)]
    for _ in range(iters):
        # expected aggregates
        S = [0.0]*T
        for t in range(T): S[t] = sum(y_act[a][t]*p[a][t] for a in range(A))
        pi = [0.0]*A
        for a in range(A): pi[a] = T - sum(p[a][t] for t in range(T))
        P = sum(pi)
        # gradient and update
        for a in range(A):
            for t in range(T):
                g = S[t]*y_act[a][t] - gamma*(u1[a][t]-u0[a][t]) - 2.0*kappa_fair*(pi[a] - P/float(A))
                if kappa_sw != 0.0:
                    m1 = p[a][t-1] if t-1>=0 else 0.0
                    m2 = p[a][t+1] if t+1<T else 0.0
                    g += 4.0*kappa_sw - 4.0*kappa_sw*(m1 + m2)
                p[a][t] = min(1.0, max(0.0, p[a][t] - eta*g))
    # sample candidates and pick best
    best_x = None; best_E = float('inf')
    for _ in range(samples):
        x = [[1 if rng.random() < p[a][t] else 0 for t in range(T)] for a in range(A)]
        E = structured_energy(x, model, weights)
        if E < best_E: best_E = E; best_x = x
    return best_x

def make_slack_meta(inst):
    # mirror minimal slack layout used in PUBO to enable decoding/outputs
    decision_idx = inst["decision_idx"]; T = inst["T"]; A = inst["num_agents"]
    scenarios = inst["scenarios"]; quant = float(inst.get("quantize_delta",1.0))
    slack_specs = []; slack_indices = {}; cur = len(decision_idx)
    # feeders
    for r, sc in enumerate(scenarios):
        for t in range(T):
            cap_val = float(sc["P"][t])
            cap_ticks = max(0, int(round(cap_val/quant)))
            K = max(1, math.ceil(math.log2(cap_ticks+1)))
            name = f"slack_feed_r{r}_t{t}"
            slack_specs.append((name, cap_val, K, "feeder", {"r": r, "t": t}))
            for k in range(K): slack_indices[(name,k)] = cur; cur += 1
    # budgets
    E_a = inst["E_a"]; E_dict = E_a if isinstance(E_a, dict) else {a: E_a[a] for a in range(A)}
    for a in range(A):
        cap_val = float(E_dict[a] if a in E_dict else E_dict[str(a)])
        cap_ticks = max(0, int(round(cap_val/quant)))
        K = max(1, math.ceil(math.log2(cap_ticks+1)))
        name = f"slack_budget_a{a}"
        slack_specs.append((name, cap_val, K, "budget", {"a": a}))
        for k in range(K): slack_indices[(name,k)] = cur; cur += 1
    meta = {"N_dec": len(decision_idx), "N_total": cur, "slack_specs": slack_specs, "slack_indices": slack_indices, "decision_idx": decision_idx}
    return meta

def b_from_x_and_fill_slacks(x, inst, meta):
    # build decision part
    decision_idx = inst["decision_idx"]; A = inst["num_agents"]; T = inst["T"]; options = inst["options_per_agent"]
    ato_to_idx = {(m["agent"], m["t"], m["opt"]): i for i, m in decision_idx.items()}
    N_total = meta["N_total"]; N_dec = meta["N_dec"]
    b = [0]*N_total
    for a in range(A):
        for t in range(T):
            xat = x[a][t]
            if options>1:
                b[ato_to_idx[(a,t,1)]] = xat
                b[ato_to_idx[(a,t,0)]] = 1-xat
            else:
                b[ato_to_idx[(a,t,0)]] = 1
    # set slacks to satisfy equalities exactly
    quant = float(inst.get("quantize_delta",1.0))
    # compute feeder slacks
    for (name, cap, K, kind, info) in meta["slack_specs"]:
        if kind == "feeder":
            r = info["r"]; t = info["t"]; y_r = inst["scenarios"][r]["y"]
            # sum load ticks at time t
            ssum_ticks = 0
            for idx, mi in decision_idx.items():
                if mi["t"]==t: ssum_ticks += int(round(y_r[idx]/quant)) * b[idx]
            cap_ticks = int(round(float(cap)/quant))
            S_ticks = max(0, cap_ticks - ssum_ticks)
        elif kind == "budget":
            a = info["a"]; y_nom = inst["y_nominal"]
            ssum_ticks = 0
            for idx, mi in decision_idx.items():
                if mi["agent"]==a: ssum_ticks += int(round(y_nom[idx]/quant)) * b[idx]
            cap_ticks = int(round(float(cap)/quant))
            S_ticks = max(0, cap_ticks - ssum_ticks)
        else:
            S_ticks = 0
        # write bits
        for k in range(K):
            bit = (S_ticks >> k) & 1
            b[ meta["slack_indices"][(name,k)] ] = bit
    return b

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

            if config.get("use_structured", False):
                # Structured hybrid path for large-scale
                model = build_x_model(inst)
                # Hybrid warm-start via population-like replicator
                hconf = config.get("hybrid", {})
                x0 = replicator_warmstart(
                    model, penalties,
                    iters=hconf.get("replicator_iters", 20),
                    eta=hconf.get("replicator_eta", 0.1),
                    temperature=hconf.get("temperature", 1.0),
                    samples=hconf.get("sample_candidates", 8),
                    seed=seed+5
                )
                # Deterministic BR in structured space
                x_sol, E_det = deterministic_BR_structured(
                    model, penalties, x0,
                    max_steps=solver_params.get("deterministic_max_steps", 20000),
                    seed=seed+9,
                    return_hist=True
                )
                E_det_val, E_hist = E_det
                # Build meta for slacks and b-vector
                meta = make_slack_meta(inst)
                b_det = b_from_x_and_fill_slacks(x_sol, inst, meta)
                # For structured path we set stochastic equal to deterministic (or skip)
                b_sto = list(b_det)
                E_sto = E_det_val
                # Residuals
                cons_det = check_constraints_on_state(b_det, inst, meta)
                cons_sto = cons_det
                # Validate
                run_data = {
                    "pubo_spin_equal": None,
                    "pubo_spin_info": {},
                    "onehot_res_det": cons_det["onehot_res"],
                    "feeder_res_det": cons_det["feeder_res"],
                    "budget_res_det": cons_det["budget_res"]
                }
                validation_passed, validation_issues = validate_solution_quality(run_data, config)
                # Decode artifacts
                slacks_det = decode_slacks(b_det, inst, meta)
                chosen_det = chosen_options_by_agent_time(b_det, inst)
                loads_time_det = per_time_loads(b_det, inst)
                red_time_det = per_time_reductions_from_caps(slacks_det)
                agent_loads_det, agent_red_det = per_agent_loads_and_reductions(b_det, inst, slacks_det)
                # Package run
                run = {
                    "size": size, "trial": trial, "seed": seed,
                    "N_dec": meta["N_dec"], "N_total": meta["N_total"], "M": None,
                    "pubo_spin_equal": None, "pubo_spin_info": {},
                    "lf_ok": None, "lf_info": {},
                    "det_E": E_det_val, "det_hist_len": len(E_hist),
                    "det_hist_E": E_hist,
                    "stoch_E": E_sto,
                    "onehot_res_det": cons_det["onehot_res"],
                    "feeder_res_det": {str(k): v for k,v in cons_det["feeder_res"].items()},
                    "budget_res_det": cons_det["budget_res"],
                    "onehot_res_stoch": cons_sto["onehot_res"],
                    "feeder_res_stoch": {str(k): v for k,v in cons_sto["feeder_res"].items()},
                    "budget_res_stoch": cons_sto["budget_res"],
                    "color_groups": None,
                    "penalty_bounds": None,
                    "final_weights": penalties,
                    "b_det": b_det,
                    "b_sto": b_sto,
                    "slacks_det": slacks_det,
                    "slacks_sto": slacks_det,
                    "chosen_opt_det": {str(k): v for k,v in chosen_det.items()},
                    "chosen_opt_sto": {str(k): v for k,v in chosen_det.items()},
                    "per_time_loads_det": {f"r{r}": v for r,v in loads_time_det.items()},
                    "per_time_loads_sto": {f"r{r}": v for r,v in loads_time_det.items()},
                    "per_time_reduction_from_cap_det": {f"r{r}": v for r,v in red_time_det.items()},
                    "per_time_reduction_from_cap_sto": {f"r{r}": v for r,v in red_time_det.items()},
                    "per_agent_load_det": agent_loads_det,
                    "per_agent_reduction_det": agent_red_det,
                    "per_agent_load_sto": agent_loads_det,
                    "per_agent_reduction_sto": agent_red_det,
                    "validation_passed": validation_passed,
                    "validation_issues": validation_issues
                }
                summary["runs"].append(run)

                base = os.path.join(outdir, f"phase4_size{size}_trial{trial}_seed{seed}")
                write_json(base+"_inst.json", inst)
                write_json(base+"_meta.json", meta)
                write_json(base+"_results.json", run)
                try:
                    visualize_and_export(inst, meta, run, base)
                except Exception as e:
                    print("[viz] Skipped due to:", e)
                print(f"[Structured Size {size} Trial {trial}] detE={E_det_val:.6g}, residual={residual_norm(cons_det):.3g}")
                continue

            # ------------------ Original exact PUBO→Spin path ------------------
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
                    s_f, E_f, _ = deterministic_BR(
                        H_rows_f, J_f, c_f,
                        max_steps=solver_params["deterministic_max_steps"],
                        seed=seed+31,
                        s_init=None,
                        use_local=False
                    )
                    b_f = b_from_s(s_f)
                    cons_f = check_constraints_on_state(b_f, inst, meta_f)
                    if residual_norm(cons_f) == 0.0:
                        feasible_state = b_f
                        break

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
            if feasible_state is None:
                feasible_state = construct_trivially_feasible_state(inst, meta)
            s_init = [1 if feasible_state[i]==1 else -1 for i in range(meta["N_total"])]

            # Multi-start deterministic BR
            best_det_s = None
            best_det_E = float('inf')
            best_det_hist = None
            
            for restart in range(solver_params["multi_start_restarts"]):
                restart_seed = seed + 13 + restart * 100
                # Use the feasible warm start on the first restart to preserve constraints
                warm = s_init if restart == 0 else None
                s_det_curr, E_det_curr, hist_det_curr = deterministic_BR(
                    H_rows, Jcoeffs, const_term,
                    max_steps=solver_params["deterministic_max_steps"],
                    seed=restart_seed,
                    s_init=warm,
                    use_local=lf_ok
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
                    seed=seed+13,
                    s_init=s_det,
                    use_local=lf_ok
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

            # Decode analysis artifacts for deterministic solution
            slacks_det = decode_slacks(b_det, inst, meta)
            chosen_det = chosen_options_by_agent_time(b_det, inst)
            loads_time_det = per_time_loads(b_det, inst)
            red_time_det = per_time_reductions_from_caps(slacks_det)
            agent_loads_det, agent_red_det = per_agent_loads_and_reductions(b_det, inst, slacks_det)

            # Decode analysis artifacts for stochastic solution
            slacks_sto = decode_slacks(b_sto, inst, meta)
            chosen_sto = chosen_options_by_agent_time(b_sto, inst)
            loads_time_sto = per_time_loads(b_sto, inst)
            red_time_sto = per_time_reductions_from_caps(slacks_sto)
            agent_loads_sto, agent_red_sto = per_agent_loads_and_reductions(b_sto, inst, slacks_sto)

            run = {
                "size": size, "trial": trial, "seed": seed,
                "N_dec": meta["N_dec"], "N_total": meta["N_total"], "M": len(H_rows),
                "pubo_spin_equal": ok_ps, "pubo_spin_info": info_ps,
                "lf_ok": lf_ok, "lf_info": lf_info,
                "det_E": E_det, "det_hist_len": len(hist_det),
                "det_hist_E": [h[3] for h in (hist_det or [])],
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
                # Added analysis fields
                "b_det": b_det,
                "b_sto": b_sto,
                "slacks_det": slacks_det,
                "slacks_sto": slacks_sto,
                "chosen_opt_det": {str(k): v for k,v in chosen_det.items()},
                "chosen_opt_sto": {str(k): v for k,v in chosen_sto.items()},
                "per_time_loads_det": {f"r{r}": v for r,v in loads_time_det.items()},
                "per_time_loads_sto": {f"r{r}": v for r,v in loads_time_sto.items()},
                "per_time_reduction_from_cap_det": {f"r{r}": v for r,v in red_time_det.items()},
                "per_time_reduction_from_cap_sto": {f"r{r}": v for r,v in red_time_sto.items()},
                "per_agent_load_det": agent_loads_det,
                "per_agent_reduction_det": agent_red_det,
                "per_agent_load_sto": agent_loads_sto,
                "per_agent_reduction_sto": agent_red_sto,
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
            try:
                visualize_and_export(inst, meta, run, base)
            except Exception as e:
                print("[viz] Skipped due to:", e)
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
