import argparse, json, os
from phase4_core.config import DEFAULT_OUTDIR, CONFIG_PROFILES, now_str
from phase4_core.instance import generate_dr_instance
from phase4_core.pubo import build_pubo
from phase4_core.spin import pubo_to_spin, spin_to_Hrows, eval_spin_energy, b_from_s
from phase4_core.solver import deterministic_BR, greedy_coloring, algorithm2_stochastic
from phase4_core.validate import (pubo_spin_consistency_check, check_constraints_on_state,
                                  residual_norm, compute_scale_aware_penalties)

def write_json(path, obj):
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2, sort_keys=True)

def run_phase4_with_config(outdir, config_name="small_debug", sizes=None, trials=2, seed0=1234):
    if config_name not in CONFIG_PROFILES:
        raise ValueError(f"Unknown config profile: {config_name}")
    config = CONFIG_PROFILES[config_name]
    sizes = (config["instance"]["num_agents"],) if sizes is None else tuple(sizes)

    summary = {"timestamp": now_str(), "outdir": outdir, "runs": [], "config_name": config_name, "config": config}
    scaling = config["scaling"]; inst_cfg = config["instance"]; base_w = config["penalties"]
    solver = config["solver"]; feas = config["feasibility_phase"]; checks = config["checks"]

    for size in sizes:
        for trial in range(trials):
            seed = seed0 + size*100 + trial

            # Instance
            inst = generate_dr_instance(num_agents=size, T=inst_cfg["T"], options_per_agent=inst_cfg["options_per_agent"],
                                        seed=seed, scenario_count=inst_cfg["scenario_count"], include_robustness=True,
                                        delta=scaling["delta"])

            # Penalty auto-scaling (lower bounds)
            bounds = compute_scale_aware_penalties(inst, base_w)
            w_scaled = dict(base_w)
            w_scaled["kappa_eq"] = max(base_w["kappa_eq"], bounds["kappa_eq_recommended"])
            w_scaled["kappa_1h"] = max(base_w["kappa_1h"], bounds["kappa_1h_recommended"])

            # -------- Feasibility phase (correctly indented) --------
            feasible_state = None
            if feas["enabled"]:
                for sc in feas["penalty_scales"]:
                    w_feas = dict(w_scaled)
                    w_feas["kappa_eq"] = w_scaled["kappa_eq"] * sc
                    w_feas["kappa_1h"] = w_scaled["kappa_1h"] * sc

                    PUBO_f, meta_f = build_pubo(inst, w_feas)
                    spin_f = pubo_to_spin(PUBO_f)
                    H_rows_f, J_f, c_f = spin_to_Hrows(spin_f)

                    # BR warm-start from random, push to feasibility
                    s_f, E_f, _ = deterministic_BR(H_rows_f, J_f, c_f, max_steps=solver["deterministic_max_steps"], seed=seed+31)
                    b_f = b_from_s(s_f)
                    cons_f = check_constraints_on_state(b_f, inst, meta_f)
                    if residual_norm(cons_f) == 0.0:
                        feasible_state = b_f
                        break

            if feasible_state is None:
                # ultra-conservative fallback: all zero (still feasible for feeder/budget; one-hot penalizes)
                PUBO_f, meta_f = build_pubo(inst, w_scaled)  # ensure meta_f exists
                feasible_state = [0] * meta_f["N_total"]

            # -------- Target run (intended weights) --------
            PUBO, meta = build_pubo(inst, w_scaled)
            spin = pubo_to_spin(PUBO)

            ok_ps, info_ps = pubo_spin_consistency_check(PUBO, spin,
                                                         trials=40, seed=seed+7,
                                                         atol=checks["absolute_tolerance"],
                                                         rtol=checks["relative_tolerance"])

            H_rows, J, c0 = spin_to_Hrows(spin)
            N_total = meta["N_total"]
            color_groups = greedy_coloring(H_rows, N_total)

            # include feasibility warm-start as a deterministic candidate
            s0 = [1 if feasible_state[i]==1 else -1 for i in range(N_total)]
            best_det_s, best_det_E = s0[:], eval_spin_energy(s0, H_rows, J, c0)
            best_det_hist = []

            # random restarts
            for r in range(solver["multi_start_restarts"]):
                s_det_r, E_det_r, hist_r = deterministic_BR(H_rows, J, c0,
                                                            max_steps=solver["deterministic_max_steps"],
                                                            seed=seed+13+100*r)
                if E_det_r < best_det_E:
                    best_det_s, best_det_E, best_det_hist = s_det_r, E_det_r, hist_r

            s_det, E_det, hist_det = best_det_s, best_det_E, best_det_hist
            b_det = b_from_s(s_det)
            cons_det = check_constraints_on_state(b_det, inst, meta)

            # If deterministic violates residuals (unexpected), revert to s0 and rerun once
            if residual_norm(cons_det) != 0.0:
                s_det, E_det, hist_det = deterministic_BR(H_rows, J, c0,
                                                          max_steps=solver["deterministic_max_steps"],
                                                          seed=seed+13, s_init=s0)
                b_det = b_from_s(s_det)
                cons_det = check_constraints_on_state(b_det, inst, meta)

            # Stochastic Algorithm-2, warm-start (faithful)
            s_sto, E_sto = algorithm2_stochastic(H_rows, J, c0, color_groups,
                                                 max_iters=solver["stochastic_max_iters"],
                                                 seed=seed+17,
                                                 alpha0=solver["stochastic_alpha0"],
                                                 beta=solver["stochastic_beta"],
                                                 B0=solver["stochastic_B0"],
                                                 eps=solver["stochastic_eps"],
                                                 s_init=(s_det if solver["warm_start_from_deterministic"] else None))
            b_sto = b_from_s(s_sto)
            cons_sto = check_constraints_on_state(b_sto, inst, meta)

            # Validation summary
            run = {
                "size": size, "trial": trial, "seed": seed,
                "N_dec": meta["N_dec"], "N_total": N_total, "M": len(H_rows),
                "pubo_spin_equal": ok_ps, "pubo_spin_info": info_ps,
                "det_E": E_det, "det_hist_len": len(hist_det),
                "stoch_E": E_sto,
                "onehot_res_det": cons_det["onehot_res"],
                "feeder_res_det": {str(k): v for k, v in cons_det["feeder_res"].items()},
                "budget_res_det": cons_det["budget_res"],
                "onehot_res_stoch": cons_sto["onehot_res"],
                "feeder_res_stoch": {str(k): v for k, v in cons_sto["feeder_res"].items()},
                "budget_res_stoch": cons_sto["budget_res"],
                "color_groups": [len(g) for g in color_groups],
                "penalty_bounds": bounds,
                "final_weights": w_scaled
            }

            base = os.path.join(outdir, f"phase4_size{size}_trial{trial}_seed{seed}")
            write_json(base+"_inst.json", inst)
            write_json(base+"_meta.json", meta)
            write_json(base+"_pubo_terms.json", {"num_terms": len(PUBO)})
            write_json(base+"_spin_terms.json", {"num_terms": len(spin)})
            write_json(base+"_results.json", run)

            print(f"[Size {size} Trial {trial}] detE={E_det:.6g}, stochE={E_sto:.6g}, "
                  f"PUBO==Spin:{ok_ps}, M={len(H_rows)}")

            summary["runs"].append(run)

    write_json(os.path.join(outdir, "phase4_summary.json"), summary)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Phase-4 STRICT DR→PUBO→Spin→HO-Ising pipeline (modular)")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--config", type=str, default="small_debug", choices=list(CONFIG_PROFILES.keys()))
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--list-configs", action="store_true")
    args = parser.parse_args()

    if args.list_configs:
        for k,v in CONFIG_PROFILES.items():
            print(f"{k}: {v['description']}")
        return

    run_phase4_with_config(outdir=args.outdir,
                           config_name=args.config,
                           sizes=args.sizes if args.sizes else None,
                           trials=args.trials, seed0=args.seed)

if __name__ == "__main__":
    main()
