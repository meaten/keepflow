"""HPD-coverage calibration of the predicted densities: per-case HPD levels u.

For each test case we draw N samples from the predictive distribution, evaluate
the log-density of every sample and of the ground truth with *one* call to the
same density function, and record

    u = mean_i 1[log p(x_i) > log p(x*)] + 0.5 * mean_i 1[log p(x_i) == log p(x*)],

the probability mass of the predictive that is more probable than the ground
truth. Under calibration u ~ U(0, 1). `scripts/calib_summary.py` turns the u
values into reliability diagrams and ECE.

The samples and the ground truth are scored in a single batch, so the
stochastic parts of the estimator (the CIF latent w drawn from r(w|x), the
Hutchinson trace noise) are drawn i.i.d. per row by the same code path for
both. This is what makes u valid: the score of x* and the scores of the x_i are
exchangeable whenever x* is a draw from the predictive, whatever the score
function. The score only decides how much the test can see; scoring with
log p makes the levels HPD levels.

Methods
-------
cnf        FlowChain-CNF (configs/traj/FlowMatchingChain_onestep/CIF_spline).
           Samples: the model's own sampler (`sample_with_log_prob`).
           Density: the per-step marginal p(x_{t+n} | o_t) obtained by taking
           each point back through the whole chain to one base Gaussian
           (`chain_preimages` + `log_prob_chain`, start_step=0), i.e. exactly
           `FlowChainTraj.analytic_log_prob_chain`, the function behind the
           paper's log-probability column. One u per step n.
           No joint: the chain is a deterministic map of (z_0, w), so the
           12-step trajectory lies on an 18-dimensional manifold of R^24 and
           has no density there.
nf         FlowChain-NF (configs/traj/FlowChain/CIF_separate_cond_v).
           Density: joint log p(x_{1:T}|o) = sum_n log p(x_n | x_{<n}, o), the
           teacher-forced one-step conditionals of `log_prob_separate` (the
           function `analytic_log_prob` and the training loss use). One u per
           case (n = -1). Velocity space; the -2 log dt shift is common to all
           rows and cancels in u.
           Samples: ancestral sampling of that same Markov model, x_n =
           f_n(z_n), z_n ~ N(x_{n-1}, sigma_n^2). This is the distribution the
           joint density belongs to.
nf-native  FlowChain-NF scored with the same joint density, but sampled with
           the model's own sampler, which draws base noise once and then maps
           x_n = f_n(x_{n-1}) with no fresh base noise. That sampler does not
           draw from the density above. The ranking is still a valid
           calibration test of *that* sampler, though not an HPD test.
trajectron Trajectron++ (configs/traj/Trajectron). Density: the per-step
           position-space GMM the table scores GT with (`predict(return_prob=
           True)`: the latent-averaged mixture, logsumexp - log S), with the
           ground truth integrated to positions over time (the table's call
           integrates over the x/y axis instead; see trajectron_step_logp). Samples:
           exact draws from that mixture (latent slot uniform, then GMM2D
           rsample). One u per step n.
gt         Exact ground-truth posterior, to validate the pipeline. Uses the
           CNF config only for the data loader. unicycle: closed-form per-step
           density (`keepflow/data/traj/analytic_density.py`) and exact
           samples; fork/branch: the generating GMM, per step and joint.

Usage
-----
    pyenv exec python scripts/calib_hpd.py --method cnf --dataset eth --device cuda:0
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from keepflow.utils import load_config
from keepflow.data import build_dataloader
from keepflow.models import build_model
from keepflow.metrics import build_metrics
from train import dict_to_device

CONFIG_DIRS = {
    "cnf": "configs/traj/FlowMatchingChain_onestep/CIF_spline",
    "nf": "configs/traj/FlowChain/CIF_separate_cond_v",
    "nf-native": "configs/traj/FlowChain/CIF_separate_cond_v",
    "gt": "configs/traj/FlowMatchingChain_onestep/CIF_spline",
    "trajectron": "configs/traj/Trajectron",
}

FORK_NOISE_STD = 0.3  # process_data.py, simfork/simbranch generator

FIELDS = ["case_id", "scene", "agent", "t", "n", "u", "ess_frac", "logp_gt", "tie_frac",
          "nonfinite_frac"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=sorted(CONFIG_DIRS))
    p.add_argument("--dataset", required=True)
    p.add_argument("--config_root", default="",
                   help="prefix for the seed copies, e.g. configs_seed/s2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--sampling", choices=["per_agent", "all"], default="per_agent",
                   help="per_agent: one snapshot per agent at its median index, as "
                        "test.py --scene_sampling per_agent (for SimTrajDensity every "
                        "trajectory is its own agent, so this is the whole test set)")
    p.add_argument("--max_cases", type=int, default=-1,
                   help="evenly thin the selected cases to at most this many")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--out_dir", default="results/calib")
    p.add_argument("--tag", default="", help="suffix for the output file name")
    args = p.parse_args()
    args.config_file = os.path.join(args.config_root, CONFIG_DIRS[args.method],
                                    f"{args.dataset}.yml")
    return args


# --------------------------------------------------------------------------- #
# case selection
# --------------------------------------------------------------------------- #

def select_cases(cfg, dataset, sampling, max_cases):
    n = len(dataset)
    if sampling == "all":
        idx = list(range(n))
    elif "sim" in cfg.DATA.DATASET_NAME:
        groups = {}
        for i, (scene, t, node) in enumerate(dataset.index):
            groups.setdefault((scene.name, node.id), []).append(i)
        idx = sorted(v[len(v) // 2] for v in groups.values())
    else:
        groups = {}
        for i in range(n):
            scene_path, agent_id, _ = dataset._data_index[i]
            groups.setdefault((str(scene_path), agent_id), []).append(i)
        idx = sorted(v[len(v) // 2] for v in groups.values())
    if 0 < max_cases < len(idx):
        keep = np.linspace(0, len(idx) - 1, max_cases).round().astype(int)
        idx = [idx[k] for k in sorted(set(keep.tolist()))]
    return idx


# --------------------------------------------------------------------------- #
# HPD level
# --------------------------------------------------------------------------- #

def hpd_level(lp_samples, lp_gt):
    """lp_samples: (N, ...) ; lp_gt: (...). Returns (u, tie_frac), each (...)."""
    lp_samples = torch.as_tensor(lp_samples, dtype=torch.float64)
    lp_gt = torch.as_tensor(lp_gt, dtype=torch.float64)
    gt = lp_gt[None]
    above = (lp_samples > gt).double().mean(0)
    tie = (lp_samples == gt).double().mean(0)
    return above + 0.5 * tie, tie


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

def encode(model, data_dict):
    node = model.get_node_type(data_dict)
    cond = model._encode(data_dict, node)
    base_pos = model._get_base_pos(data_dict)
    return model.flow[node], cond, base_pos


def native_samples(flow, cond, base_pos, n):
    """The models' own sampler, as in FlowChainTraj._predict_inverse_prob."""
    seq, _, _, _ = flow.sample_with_log_prob(
        base_pos[:, None].expand(-1, n, -1), cond=cond[:, None].expand(-1, n, -1))
    return seq[0]                                                    # (n, T, 2)


def nf_ancestral_samples(flow, cond, base_pos, n):
    """Ancestral draw from the Markov model whose density is log_prob_separate:
    z_n ~ N(x_{n-1}, sigma_n^2), x_n = f_n(z_n | o, x_{<n})."""
    c = cond[:, None].expand(-1, n, -1)                              # (1, n, C)
    prev = base_pos[:, None].expand(-1, n, -1)                       # (1, n, 2)
    seq = []
    for step in range(flow.pred_len):
        z = flow.base_dist(prev, step=step).sample()
        if step == 0:
            cc = c
        else:
            cc = torch.cat([c, torch.stack(seq, dim=2).reshape(1, n, -1)], dim=-1)
        x, _ = flow.net[step].inverse(z, cc)
        seq.append(x)
        prev = x
    return torch.stack(seq, dim=2)[0]                                # (n, T, 2)


def cnf_marginal_logp(flow, cond, base_pos, seq):
    """Per-step marginal log p(x_{t+n}|o_t): analytic_log_prob_chain at k = 0."""
    B = seq.shape[0]
    c = cond.expand(B, -1)
    bp = base_pos.expand(B, -1)
    cache = flow.chain_preimages(seq, c)
    base_positions = torch.cat([bp[:, None], seq[:, :-1]], dim=1)
    return flow.log_prob_chain(base_positions, cache, start_step=0)  # (B, T)


def nf_joint_logp(flow, cond, base_pos, seq):
    """sum_n log p(x_n | x_{<n}, o): the per-step terms of analytic_log_prob.

    A few ancestral draws diverge (the chain is applied to its own, off-
    distribution outputs). Such a draw is still a draw from the model; it is
    scored log p = -inf, i.e. less probable than any finite ground truth, and
    its share is reported as nonfinite_frac.
    """
    B = seq.shape[0]
    bad = ~torch.isfinite(seq).all(dim=-1).all(dim=-1)
    seq = torch.where(bad[:, None, None], torch.zeros_like(seq), seq)
    lp = flow.log_prob(base_pos.expand(B, -1), seq, cond.expand(B, -1)).sum(dim=1)
    lp = torch.where(bad | ~torch.isfinite(lp), torch.full_like(lp, -float("inf")), lp)
    return lp


def trajectron_step_logp(model, data_dict, n):
    """Samples from, and per-step log density of, Trajectron++'s predictive GMM."""
    gt = data_dict["gt"].clone()
    data_dict = model.predict(data_dict, return_prob=True)
    y_dist = data_dict[("prob", 0)]                                   # GMM2D [S, 1, T, K]
    from trajdata import AgentType
    # same lookup as Trajectron.predict
    node_type = str(AgentType(data_dict["agent_type"][0].item())).lstrip("AgentType.")
    inner = model.node_models_dict[node_type]
    # integrate_samples cumsums over dim 2 and so expects [S, bs, T, 2];
    # Trajectron.predict passes [bs, T, 2], which sums x and y instead of time
    # (so the table's Trajectron++ GT log-prob is taken at the wrong points).
    # The leading axis added here makes dim 2 the time axis.
    gt_pos = inner.dynamic.integrate_samples(gt[None])[0]             # [1, T, 2]
    S = y_dist.log_pis.shape[0]
    draws = y_dist.rsample((n,))                                      # [n, S, 1, T, 2]
    slot = torch.randint(S, (n,), device=draws.device)
    x = draws[torch.arange(n, device=draws.device), slot]             # [n, 1, T, 2]
    seq = torch.cat([x, gt_pos[None]], dim=0)                         # [n+1, 1, T, 2]
    lp = y_dist.log_prob(seq[:, None])                                # [n+1, S, 1, T]
    lp = torch.logsumexp(lp, dim=1) - np.log(S)
    return x[:, 0], lp[:, 0].cpu()


# --------------------------------------------------------------------------- #
# exact ground truth (SimTrajDensity)
# --------------------------------------------------------------------------- #

class GroundTruth:
    def __init__(self, cfg):
        name = cfg.DATA.DATASET_NAME
        self.name = name
        if name == "simunicycle":
            from keepflow.data.traj.analytic_density import build_unicycle_gt
            self.dens = build_unicycle_gt(cfg.DATA.PREDICT_LENGTH)
        elif name in ("simfork", "simbranch"):
            import dill
            env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "processed_data" / f"{name}_test.pkl"
            with open(env_path, "rb") as f:
                env = dill.load(f, encoding="latin1")
            assert env.gt_dist_type == "gmm", env.gt_dist_type
            # Only the mode means are taken from the stored GMM. Its scale is
            # std**2 = 0.09 (process_data.py multiplies `scale`, already equal
            # to std, by std again), whereas the generator adds N(0, 0.3^2) to
            # every one of the 20 positions, observed ones included; and the
            # evaluation frame is centred on the *noisy* last observation. The
            # exact posterior is therefore derived here from the generator.
            comp = env.gt_dist.component_distribution.base_dist
            self.mu = comp.loc.double()                              # (T, K, 2)
            self.logit = env.gt_dist.mixture_distribution.logits.double()[0]  # (K,)
            self.sigma = FORK_NOISE_STD
            self.obs_len = cfg.DATA.OBSERVE_LENGTH
            self.shift = None
        else:
            raise ValueError(f"no exact ground truth for {name}")

    def condition(self, obs_xy):
        """Condition the fork/branch posterior on the observed window.

        In the frame centred on the last observation, x_n = mu_{n,k} + e_n - e_L,
        with e_* i.i.d. N(0, s^2 I) and e_L the last observation's noise. The
        observed history fixes d_j = e_j - e_L (the deviation of observation j
        from the noise-free straight line, which ends at the origin), and with
        m = obs_len - 1 such deviations, e_L | d ~ N(-sum(d)/(m+1), s^2/(m+1)).
        Hence

            x | o, k ~ N(mu_k + sum(d)/(m+1), s^2 I_T + s^2/(m+1) 11^T)

        per coordinate; the mode is not identified by the (shared) history.
        """
        if self.name == "simunicycle":
            return
        obs_xy = obs_xy.double().cpu()                               # (L, 2)
        L = obs_xy.shape[0]
        line = torch.stack([torch.arange(-(L - 1), 1, dtype=torch.float64),
                            torch.zeros(L, dtype=torch.float64)], dim=1)
        d = (obs_xy - line)[:-1]                                     # (L-1, 2)
        self.shift = d.sum(0) / L                                    # E[-e_L | d]
        T = self.mu.shape[0]
        s2 = self.sigma ** 2
        cov_t = s2 * torch.eye(T, dtype=torch.float64) + s2 / L * torch.ones(T, T, dtype=torch.float64)
        self.cov_t = cov_t
        self.chol_t = torch.linalg.cholesky(cov_t)
        self.step_sd = (s2 + s2 / L) ** 0.5

    def sample(self, n, rng):
        if self.name == "simunicycle":
            from keepflow.data.traj.analytic_density import sample_unicycle
            return torch.as_tensor(
                sample_unicycle(n, pred_len=len(self.dens), seed=int(rng.integers(2**31))))
        T, K, _ = self.mu.shape
        w = torch.softmax(self.logit, 0).numpy()
        k = rng.choice(K, size=n, p=w)
        eps = torch.as_tensor(rng.standard_normal((n, T, 2)))
        noise = torch.einsum("ts,nsd->ntd", self.chol_t, eps)
        return self.mu[:, k].transpose(0, 1) + self.shift + noise

    def step_logp(self, seq):
        """(B, T, 2) positions -> (B, T) per-step marginal log density."""
        seq = seq.double()
        if self.name == "simunicycle":
            out = []
            for n, d in enumerate(self.dens):
                with np.errstate(divide="ignore"):
                    out.append(torch.as_tensor(np.log(d(seq[:, n].numpy()))))
            return torch.stack(out, dim=1)
        logn = torch.distributions.Normal(self.mu[None] + self.shift, self.step_sd).log_prob(
            seq[:, :, None]).sum(-1)                                  # (B, T, K)
        return torch.logsumexp(logn + torch.log_softmax(self.logit, -1), dim=-1)

    def joint_logp(self, seq):
        if self.name == "simunicycle":
            return None
        seq = seq.double()
        mvn = torch.distributions.MultivariateNormal(
            torch.zeros(seq.shape[1], dtype=torch.float64), scale_tril=self.chol_t)
        r = seq[:, None] - (self.mu.transpose(0, 1)[None] + self.shift)  # (B, K, T, 2)
        logn = mvn.log_prob(r.transpose(-1, -2)).sum(-1)              # (B, K)
        return torch.logsumexp(logn + torch.log_softmax(self.logit, -1), dim=-1)


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #

def main():
    # Diverged ancestral NF draws are non-finite; let them through the base
    # Gaussian of the next step instead of raising (they are scored -inf).
    torch.distributions.Distribution.set_default_validate_args(False)
    args = parse_args()
    cfg = load_config(args)
    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)
    dataset, collate_fn = loader.dataset, loader.collate_fn

    cases = select_cases(cfg, dataset, args.sampling, args.max_cases)
    n_cases_total = len(cases)
    if args.num_shards > 1:
        cases = cases[args.shard::args.num_shards]
    print(f"{args.method}/{args.dataset}: {len(cases)} cases "
          f"(of {n_cases_total}), N = {args.n_samples}", flush=True)

    if args.method == "gt":
        gt_model = GroundTruth(cfg)
        metrics = build_metrics(cfg)
    else:
        model = build_model(cfg)
        model.load()
        model.eval()

    stem = f"u_{args.method}_{args.dataset}{args.tag}"
    if args.num_shards > 1:
        stem += f".shard{args.shard}of{args.num_shards}"
    out_path = Path(args.out_dir) / f"{stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(out_path, "w", newline="") as fp, torch.no_grad():
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        for count, idx in enumerate(cases, 1):
            # One seed per case, shared by every method: the samples and the
            # ground truth are then scored under identical random streams.
            torch.manual_seed(cfg.SEED + idx)
            np.random.seed(cfg.SEED + idx)
            rng = np.random.default_rng(cfg.SEED + idx)

            data_dict = dict_to_device(collate_fn([dataset[idx]]), cfg.DEVICE)
            scene, t, agent = data_dict["index"][0]
            gt = data_dict["gt"]                                     # (1, T, 2), model space
            if torch.isnan(gt).any():
                print(f"  skip case {idx}: NaN in ground truth", flush=True)
                continue
            base = {"case_id": idx, "scene": scene, "agent": agent, "t": int(t),
                    "ess_frac": "", "nonfinite_frac": ""}

            if args.method == "gt":
                data_dict = metrics.denormalize(data_dict)
                gt_world = data_dict["gt"][0].double().cpu()          # (T, 2) world
                gt_model.condition(data_dict["obs"][0, :, :2])
                samples = gt_model.sample(args.n_samples, rng)       # (N, T, 2) world
                seq = torch.cat([samples, gt_world[None]], dim=0)
                lp_step = gt_model.step_logp(seq)
                lp_joint = gt_model.joint_logp(seq)
            elif args.method == "trajectron":
                samples, lp_step = trajectron_step_logp(model, data_dict, args.n_samples)
                lp_joint = None
            elif args.method == "cnf":
                flow, cond, base_pos = encode(model, data_dict)
                samples = native_samples(flow, cond, base_pos, args.n_samples)
                seq = torch.cat([samples, gt], dim=0)
                lp_step = cnf_marginal_logp(flow, cond, base_pos, seq).cpu()
                lp_joint = None
            else:
                flow, cond, base_pos = encode(model, data_dict)
                if args.method == "nf":
                    samples = nf_ancestral_samples(flow, cond, base_pos, args.n_samples)
                else:
                    samples = native_samples(flow, cond, base_pos, args.n_samples)
                seq = torch.cat([samples, gt], dim=0)
                lp_step = None
                lp_joint = nf_joint_logp(flow, cond, base_pos, seq).cpu()

            if lp_step is not None:
                u, tie = hpd_level(lp_step[:-1], lp_step[-1])
                for n in range(lp_step.shape[1]):
                    writer.writerow({**base, "n": n + 1, "u": f"{u[n].item():.6f}",
                                     "logp_gt": f"{lp_step[-1, n].item():.6f}",
                                     "tie_frac": f"{tie[n].item():.6f}"})
            if lp_joint is not None:
                if not torch.isfinite(lp_joint[-1]):
                    print(f"  case {idx}: non-finite GT log-prob", flush=True)
                base["nonfinite_frac"] = f"{(~torch.isfinite(lp_joint[:-1])).double().mean().item():.6f}"
                u, tie = hpd_level(lp_joint[:-1], lp_joint[-1])
                writer.writerow({**base, "n": -1, "u": f"{u.item():.6f}",
                                 "logp_gt": f"{lp_joint[-1].item():.6f}",
                                 "tie_frac": f"{tie.item():.6f}"})
            fp.flush()
            if count % 10 == 0 or count == len(cases):
                el = time.time() - t0
                print(f"  {count}/{len(cases)}  {el:.0f}s elapsed, "
                      f"{el / count:.2f}s/case", flush=True)

    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
