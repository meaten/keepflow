import os
import argparse
from yacs.config import CfgNode
import json
import numpy as np
import torch

from keepflow.utils import load_config, Timer, kde
from keepflow.data import build_dataloader
from keepflow.models import build_model
from keepflow.metrics import build_metrics
from keepflow.visualization import build_visualizer

from train import evaluate_model, aggregate, dict_to_device


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    if args.kde_n_sample is not None:
        cfg.defrost()
        cfg.TEST.KDE_N_SAMPLE = args.kde_n_sample
        cfg.freeze()

    # Override the ODE solver at inference time only. The checkpoint location is
    # derived from the config path, so the same trained model can be evaluated
    # under different integration budgets (accuracy-vs-NFE trade-off).
    if args.cnf_solver is not None:
        cfg.defrost()
        cfg.MODEL.FLOW.CNF_SOLVER = args.cnf_solver
        cfg.MODEL.FLOW.NUM_STEP = args.num_step
        cfg.freeze()

    # The update gate (apply the update only when deviation_norm <= threshold)
    # is enabled for ETH alone -- every other config leaves it at the default
    # 1000.0, which disables it -- and the manuscript does not mention it at
    # all. Overriding it here makes the two settings directly comparable.
    if args.update_threshold is not None:
        cfg.defrost()
        cfg.MODEL.FLOW.UPDATE_THRESHOLD = args.update_threshold
        cfg.freeze()
    if args.ess_threshold is not None:
        cfg.defrost()
        cfg.MODEL.FLOW.ESS_THRESHOLD = args.ess_threshold
        cfg.freeze()
    if args.force_kde:
        # Evaluate FlowChain's own particles with the same estimator the KDE
        # baselines are scored with, so the density comparison is like for like.
        cfg.defrost()
        cfg.TEST.KDE = True
        cfg.freeze()
    print(f"UPDATE_THRESHOLD = {cfg.MODEL.FLOW.UPDATE_THRESHOLD}  "
          f"ESS_THRESHOLD = {cfg.MODEL.FLOW.ESS_THRESHOLD}")

    test(cfg, de_sample_size=args.de_sample_size,
         update_timesteps=args.update_timesteps, tag=args.tag,
         scene_sampling=args.scene_sampling, analytic_logprob=args.analytic_logprob,
         chain_logprob=args.chain_logprob)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch testing code for task-agnostic time-series prediction")
    parser.add_argument("--config_file", type=str, default='',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--de_sample_size", type=int, default=None)
    parser.add_argument("--update_timesteps", type=int, nargs="*", default=[])
    parser.add_argument("--scene_sampling", type=str, default="head",
                        choices=["head", "even", "per_agent"],
                        help="how density_estimation() picks which test-set entries "
                             "to evaluate. 'head' (default) reproduces the legacy "
                             "behavior of taking the first de_sample_size entries in "
                             "dataset order (consecutive entries can be the same agent "
                             "at consecutive timesteps). 'even' spreads de_sample_size "
                             "entries evenly across the whole test set via "
                             "np.linspace, deduplicated. 'per_agent' ignores "
                             "de_sample_size and evaluates exactly one snapshot per "
                             "unique agent (the entry at that agent's median index, "
                             "chosen deterministically) so every agent is counted once.")
    parser.add_argument("--kde_n_sample", type=int, default=None,
                        help="override TEST.KDE_N_SAMPLE (sample budget for density estimation)")
    parser.add_argument("--cnf_solver", type=str, default=None,
                        choices=['euler', 'midpoint', 'rk4', 'dopri5', 'tsit5'],
                        help="override MODEL.FLOW.CNF_SOLVER at inference time; the "
                             "checkpoint location is unaffected")
    parser.add_argument("--num_step", type=int, default=None,
                        help="override MODEL.FLOW.NUM_STEP (fixed-step solvers only; "
                             "must be omitted for dopri5/tsit5)")
    parser.add_argument("--force_kde", action="store_true", default=False,
                        help="turn TEST.KDE on regardless of the config, so a model "
                             "with an analytic density is also scored by a KDE over "
                             "its own particles")
    parser.add_argument("--ess_threshold", type=float, default=None,
                        help="override MODEL.FLOW.ESS_THRESHOLD: the minimum "
                             "effective sample size, as a fraction of the particle "
                             "count, for the update to be applied. Below it the "
                             "previous forecast is kept. 0 disables the check.")
    parser.add_argument("--update_threshold", type=float, default=None,
                        help="override MODEL.FLOW.UPDATE_THRESHOLD, the gate that "
                             "suppresses the density update when the prediction has "
                             "already drifted from the new observation. Only the ETH "
                             "config sets it (0.4); everything else inherits the "
                             "default 1000.0, which disables the gate. Pass a large "
                             "value to disable it explicitly.")
    parser.add_argument("--tag", type=str, default=None,
                        help="suffix for the results filename (metrics_<tag>.json) so that "
                             "sweep runs do not overwrite each other; the checkpoint "
                             "location is unaffected")
    parser.add_argument("--chain_logprob", action="store_true", default=False,
                        help="evaluate the density by propagating the base Gaussian "
                             "through the chain (Eqs. 7-9) rather than one step from "
                             "the previous ground-truth position, and do so for the "
                             "update steps as well as the initial forecast")
    parser.add_argument("--analytic_logprob", action="store_true", default=False,
                        help="for models with an analytic density (the FlowChain family: "
                             "FlowChainTraj and its ContinuousFlowChain / FlowMatchingChain "
                             "subclasses), evaluate gt_traj_log_prob directly via "
                             "model.analytic_log_prob() instead of rasterising the particle "
                             "cloud onto a grid and interpolating at the GT position "
                             "(TrajVisualizer.prob_on_grid's LinearNDInterpolator path, which "
                             "clamps to -100 outside the particle cloud's convex hull). Has no "
                             "effect on KDE-based baselines or models without a native flow "
                             "log_prob (e.g. Trajectron++, which already reports its own "
                             "gt_traj_log_prob). Default False keeps the legacy grid path.")

    return parser.parse_args()


def test(cfg: CfgNode, de_sample_size: int=None, update_timesteps: list=None,
         tag: str=None, scene_sampling: str="head", analytic_logprob: bool=False,
         chain_logprob: bool=False) -> None:
    result_metrics = {}
    # scene_sampling != "head" (e.g. per_agent) evaluates density estimation even
    # when de_sample_size was not given on the command line.
    if de_sample_size is not None or scene_sampling != "head":
        model = build_model(cfg)
        try:
            model.load()
        except FileNotFoundError:
            print("no model saved")
        result_metrics.update(density_estimation(cfg, model, de_sample_size, update_timesteps,
                                                   scene_sampling=scene_sampling,
                                                   analytic_logprob=analytic_logprob,
                                                   chain_logprob=chain_logprob))

    model = build_model(cfg)
    try:
        model.load()
    except FileNotFoundError:
        print("no model saved")
    
    data_loader = build_dataloader(cfg, rand=False, split="test")    
    result_metrics.update(evaluate_model(cfg, model, data_loader))
    np.set_printoptions(precision=4)
    print(result_metrics)
    result_metrics = {k: str(v) for k, v in result_metrics.items()}
    fname = "metrics.json" if tag is None else f"metrics_{tag}.json"
    with open(os.path.join(cfg.SAVE_DIR, fname), "w") as fp:
        json.dump(result_metrics, fp)


def density_estimation(cfg, model: torch.nn.Module, de_sample_size: int = None, update_timesteps: list=None,
                        scene_sampling: str = "head", analytic_logprob: bool = False,
                        chain_logprob: bool = False) -> dict:
    metrics = build_metrics(cfg)
    model.eval()

    # Only models with an analytic density (the FlowChain family; see
    # FlowChainTraj.analytic_log_prob) expose this method. KDE-based baselines and
    # models that already produce their own gt_traj_log_prob (e.g. Trajectron++,
    # AR-flow) are unaffected either way -- prob_on_grid / kde() already set
    # ("gt_traj_log_prob", 0) for those before this flag would ever be consulted.
    use_analytic_logprob = analytic_logprob and hasattr(model, "analytic_log_prob")
    if analytic_logprob and not use_analytic_logprob:
        print(f"--analytic_logprob was requested but {type(model).__name__} has no "
              f"analytic_log_prob(); falling back to the grid-interpolation path.")
    
    timer = Timer(cfg.DEVICE)

    grid_size = 100
    
    result_metrics = {}
    
    run_times = {0: []}
    run_times.update({t: [] for t in update_timesteps})
    
    visualizer = build_visualizer(cfg)
    with torch.no_grad():
        result_list = []
        print("timing the computation, evaluating probability map... ")
        data_loader = build_dataloader(cfg, rand=False, split = "test", batch_size=1)
        count = 0
        compiled_model = torch.compile(model)
        if de_sample_size == -1:
            de_sample_size = len(data_loader.dataset)

        # selected_positions is None for the legacy "head" mode, in which case the
        # loop below falls back exactly to the original count-based break (including
        # its off-by-one: requesting N entries yields N-1). For "even" / "per_agent"
        # we precompute the exact dataset positions to keep and evaluate exactly
        # that many entries, with no off-by-one.
        selected_positions = None
        if scene_sampling == "even":
            n_total = len(data_loader.dataset)
            n = de_sample_size if de_sample_size is not None else n_total
            n = min(n, n_total)
            raw = np.linspace(0, n_total - 1, n).round().astype(int)
            selected_positions = sorted(set(int(x) for x in raw))
            if len(selected_positions) < n:
                print(f"[scene_sampling=even] requested {n} samples but only "
                      f"{len(selected_positions)} unique positions remain after "
                      f"dedup (dataset has {n_total} entries)")
        elif scene_sampling == "per_agent":
            if de_sample_size is not None:
                print("[scene_sampling=per_agent] de_sample_size is ignored; "
                      "evaluating exactly one snapshot per unique agent")
            agent_positions = {}
            for pos, scan_dict in enumerate(data_loader):
                scene_id, t_step, agent_key = scan_dict["index"][0]
                agent_positions.setdefault((scene_id, agent_key), []).append(pos)
            # deterministic: take the entry at each agent's median occurrence index
            # (the very first snapshot of an agent's track can be uninformative).
            selected_positions = sorted(
                idxs[len(idxs) // 2] for idxs in agent_positions.values()
            )
            print(f"[scene_sampling=per_agent] found {len(agent_positions)} unique "
                  f"agents across {len(data_loader.dataset)} entries; evaluating "
                  f"{len(selected_positions)} snapshots (one per agent)")
        elif scene_sampling != "head":
            raise ValueError(f"unknown scene_sampling mode: {scene_sampling}")

        selected_set = set(selected_positions) if selected_positions is not None else None
        target_count = len(selected_positions) if selected_positions is not None else de_sample_size

        for pos, data_dict in enumerate(data_loader):
            if selected_set is not None and pos not in selected_set:
                continue
            data_dict = dict_to_device(data_dict, cfg.DEVICE)

            count += 1
            # if count != 219:
            #     continue
            print(f"Density Estimation {count} / {target_count}")

            result_dict = compiled_model.predict(data_dict, return_prob=True)  # warm-up
            timer.start()
            result_dict = compiled_model.predict(data_dict, return_prob=True)
            timer.end()
            curr_run_time = timer.elapsed_time()
            run_times[0].append(curr_run_time)
            
            for t in update_timesteps:
                timer.start()
                result_dict = compiled_model.update_from_new_obs(result_dict, t)
                timer.end()
                curr_run_time = timer.elapsed_time()
                run_times[t].append(curr_run_time)

            # Must run BEFORE metrics.denormalize(), which mutates result_dict['gt']
            # in place (state_v: integrates it into position space; state_p: adds
            # back the standardization offset) -- analytic_log_prob needs the raw,
            # pre-denormalize ground truth because that's the space the flow (and
            # therefore its log_prob) is defined in. See
            # FlowChainTraj.analytic_log_prob's docstring for the state_v Jacobian
            # discussion.
            analytic_gt_log_prob = None
            if use_analytic_logprob:
                if chain_logprob and hasattr(model, "analytic_log_prob_chain"):
                    # One entry per number of absorbed updates, so that the
                    # update rows are evaluated on the same footing as the
                    # initial forecast instead of falling back to the grid.
                    analytic_gt_log_prob = model.analytic_log_prob_chain(
                        result_dict, [0] + list(update_timesteps))
                else:
                    analytic_gt_log_prob = model.analytic_log_prob(result_dict).detach()

            result_dict = metrics.denormalize(result_dict)  # denormalize the output
            if cfg.TEST.KDE:
                timer.start()
                result_dict = kde(result_dict)
                timer.end()
                run_times[0][-1] += timer.elapsed_time()
            result_dict = visualizer.prob_on_grid(result_dict, grid_size=grid_size,
                                                   analytic_gt_log_prob=analytic_gt_log_prob)
            result_list.append(metrics(result_dict))
            
            # visualizer(result_dict)
            if selected_set is None:
                # legacy "head" behavior, off-by-one preserved intentionally
                if count == de_sample_size - 1:
                    break
            else:
                if count == target_count:
                    break
            
        result_metrics.update(aggregate(result_list))
        
    print(f"execution time: {np.mean(run_times[0]):.2f} " + u"\u00B1" + f"{np.std(run_times[0]):.2f} [ms]")
    if len(update_timesteps) != 0:
        print(f"execution time: {np.mean(run_times[1]):.2f} " + u"\u00B1" + f"{np.std(run_times[1]):.2f} [ms]")
    result_metrics.update({"execution time": np.mean(run_times[0]), "time std": np.std(run_times[0])})
    
    model.train()
    
    return result_metrics

if __name__ == "__main__":
    main()