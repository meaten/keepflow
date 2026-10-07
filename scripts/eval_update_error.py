"""E3-1: approximation error of FlowChain's fast density-update mechanism.

Error analysis of the "fast update"
(`FlowChainTraj.update_from_new_obs`, `keepflow/models/traj/flowchain.py:218-249`).
That function does **not** recompute the flow. It re-evaluates the cached base
Gaussian -- re-centred on the new observation -- at the particle positions and
cumulative log-det-Jacobians cached from the *initial* prediction, renormalizes
with logsumexp, and gates the whole thing off (falls back to the untouched
cached density) when the observed deviation exceeds
`cfg.MODEL.FLOW.UPDATE_THRESHOLD`. This script quantifies how far that
approximation drifts from a full recomputation ("oracle") as the number of
elapsed update steps k grows.

Per test scene, for k in `--update_timesteps`:
  (a) approximate update : `model.update_from_new_obs(result, k)`
  (b) oracle             : a *fresh* `model.predict(..., return_prob=True)` run
      on the observation window that actually ends k steps later, i.e. the
      window an online system would have used had it restarted prediction
      from scratch at time t+k instead of reusing t's cache.

How the oracle observation window is obtained
----------------------------------------------
`test.py` builds its loader as `build_dataloader(cfg, rand=False, split="test",
batch_size=1)`; with `shuffle=False` iteration order equals dataset index
order. So `loader.dataset[i]` run through `loader.collate_fn` reproduces
exactly the i-th `data_dict` a plain test loop would see (see
`keepflow/data/build_dataloader.py`, `keepflow/data/traj/build_traj_dataset.py`).

* ETH/UCY, SDD, nuScenes (`keepflow/data/traj/build_traj_dataset.py`, the
  `trajdata.UnifiedDataset` branch, `centric="agent"`): we do **not** use
  `dataset[idx + k]`. An earlier version of this script did, and it was
  *correct* as far as it went -- trajdata's `AgentDataIndex` (see
  `.../site-packages/trajdata/data_structures/data_index.py`) really does lay
  each agent's queryable timesteps out as one contiguous index block, so
  `idx + k` genuinely is the same (scene, agent) k steps later whenever it
  exists. The problem is that "whenever it exists" is much narrower than it
  needs to be: `UnifiedDataset._get_data_index_agent()` ->
  `trajdata/filtering/filters.py:get_valid_ts()` only emits an index entry
  for agent-timestep `ts` if the agent has a *full* `future_sec` window
  (== `DATA.PREDICT_LENGTH` steps, both train and test -- see
  `params_ethucy()`/`params_sim()` in `build_traj_dataset.py`, whose
  `future_sec=(4.8, 4.8)` fixes min *and* max) beyond `ts`. That is a
  correctness requirement for *ordinary* dataset items (which need a full
  prediction horizon of ground truth to be evaluated), but it is needlessly
  strict for our oracle at `t + k`: the overlap this script actually scores
  is only the first `pred_len - k` steps (see "Time alignment" below), and
  the *original* query at `t` already had to satisfy the standard
  `future_sec` bound, i.e. the agent's `last_timestep >= t + pred_len ==
  (t + k) + (pred_len - k)`. So the real data on disk always has enough
  future for the comparison we need; trajdata's index just doesn't expose a
  query object for it because it additionally demands `pred_len - (pred_len
  - k) = k` more steps than that, for no reason relevant to us. Empirically
  this is why the naive `idx + k` lookup skipped 12/18 (scene, k) records in
  a 3-scene smoke test: those weren't index-arithmetic failures, they were
  short-track agents for which trajdata's conservative index simply never
  materializes a `t + k` query, even though the ground truth we need exists.
  We therefore build the oracle `AgentBatchElement` for `(scene, agent,
  t + k)` *directly*, bypassing `dataset._data_index`, via `_agent_item_at()`
  below. `trajdata`'s own cache methods
  (`trajdata/caching/df_cache.py:DataFrameCache.get_agent_history/future`)
  already do not re-check the min-future/min-history bound themselves --
  their code comments say so explicitly ("We don't have to check the mins
  here because our data_index filtering in dataset.py already took care of
  it") -- they just return whatever real, recorded data exists up to
  `future_sec[1]` (the max), which may be shorter than a full `pred_len` near
  the end of a track. So this is not fabricating or padding anything: it is
  reading the same on-disk trajectory the standard index would read, just
  without discarding query points purely because *more* future than we need
  is unavailable. If the agent has truly left the scene by `t + k` (not
  present in `scene.agent_presence[t + k]`) or `t + k` exceeds the scene
  length, `_agent_item_at()` returns `None` and the record is skipped with
  `skip_reason="shift_unavailable"` -- a genuine data-absence case, not a
  lookup artifact. After construction we still run the item through the exact same
  `collate_fn` as every other item and re-verify the returned
  `("scene_id", "scene_ts", "agent_name")` tuple via `indices_consistent()`
  before trusting it, so a bug in this reasoning would show up as
  `skip_reason="shift_mismatch"` rather than silently corrupting a record.
* `sim*` datasets (`keepflow/data/traj/trajectron_dataset.py`,
  `NodeTypeDataset`): `dataset.index` is an explicit
  `[(scene, t, node), ...]` list, but consecutive entries are *not*
  guaranteed to be the same node advanced by one timestep (the outer loop is
  over t, the inner loop over whichever nodes are present at that t). We
  therefore build an explicit `(scene.name, node.id, t) -> index` lookup
  once and query it directly, so no ordering assumption is needed. Note:
  `NodeTypeDataset.index_env()` -> `Scene.present_nodes(...,
  min_future_timesteps=pred_len)` (`keepflow/data/traj/environment/scene.py`)
  applies the *exact same* kind of full-`pred_len`-future filter as
  trajdata's `AgentDataIndex`, so `sim*` is in principle susceptible to the
  same conservative-index under-coverage described above. We did not
  observe it in practice (SimTrajDensity tracks span a fixed, long episode
  per scene, so `t + k` is essentially always still within the indexed
  range) and have not implemented the equivalent direct-construction bypass
  for it here -- if a future run of this script on `sim*` shows a non-trivial
  `shift_unavailable`/`shift_mismatch` rate, the same fix idea applies:
  `Node.get(...)` (unlike trajdata's cache) would need to be checked for
  whether it also tolerates a partial future window before reusing this
  approach.

Coordinate frames -- important correctness point
--------------------------------------------------
Both the model's predictions and the ground truth live in a frame relative to
the *query time*'s own position (`standardize_data=True` in
`build_traj_dataset.py`, or the `rel_state` centring in
`keepflow/data/traj/preprocessing.py`). The update's cached tensors are
relative to the *original* time t, while a fresh oracle prediction at t+k is
relative to t+k. Comparing raw tensor values across the two would silently
compare positions expressed in two different frames. `test.py`'s
density-estimation path calls `metrics.denormalize()` (==
`TrajMetrics.output_to_trajectories`) before touching probabilities on a
grid, which adds back each dict's own `standardize_offset` -- but that is a
**translation only**. For the ETH/UCY/SDD/nuScenes branch
(`trajdata.UnifiedDataset` with `standardize_data=True`), each query's local
frame is *also rotated* to align with the agent's heading at that query's
own reference time
(`site-packages/trajdata/data_structures/batch_element.py:57-80`,
`AgentBatchElement.agent_from_world_tf`). That rotation is irrelevant for
every existing single-query use (ADE/FDE, visualization, grid density against
ground truth) because it cancels out consistently within one result dict --
but it does NOT cancel across two different queries with two different
headings, which is exactly our case. Empirically this is not a rounding
error: on the eth smoke test, using translation-only alignment gave a
ground-truth-position mismatch between the update and oracle branches that
*grew with k* (0.9 m at k=1, up to 2.4 m at k=3) even though both nominally
refer to the same real-world point. `align_to_world()` below fixes this by
reading each query's own `agent_from_world_tf` off the *raw, uncollated*
dataset item (`dataset[idx]`, before `CollateWrapper`/`get_collate_fn`
discards the rotation) and applying the missing rotation on top of
`metrics.denormalize()`'s translation, so both dicts truly land in one shared
world frame. For `sim*` datasets, `Environment.standardize`
(`keepflow/data/traj/environment/environment.py:43-50`) is translate/scale
only (no rotation), so the correction is the identity there and
`metrics.denormalize()` alone is already correct.

Time alignment
---------------
`update_from_new_obs(data_dict, k)` only ever truncates the *original*
pred_len-long forecast (it keeps indices k..pred_len-1 of the initial
particle set); it can never extend the prediction horizon. So for update
step k, absolute times t+k+1 .. t+pred_len are available (pred_len - k
steps). An oracle prediction from t+k covers t+k+1 .. t+k+pred_len (pred_len
steps). The two overlap on exactly t+k+1 .. t+pred_len, i.e. the first
`pred_len - k` steps of each. All error metrics below are computed on that
overlap only, and a `gt_consistency_max_abs_diff` sanity value (should be
~0) is recorded per record to confirm the alignment: the ground-truth
position at a given absolute time must be identical whether read off the
update's or the oracle's own `gt` tensor.

Grid / metrics
---------------
KL divergence and Sinkhorn distance are divergences *between* the update and
the oracle, so they need one shared support: the particle-weighted density is
rasterized onto a shared grid using the same routine as everywhere else in
the codebase (`TrajVisualizer.griddata_on_cluster`, see also
`TrajVisualizer.prob_on_grid` and its use in `test.py`'s
`density_estimation`), with the grid extent fixed once per scene (from the
*initial* prediction's `get_minmax`, i.e. exactly what `test.py` would use
since it also accumulates every update's keys into one dict before a single
`prob_on_grid` call) and reused for both the update and the oracle -- see
`kl_divergence`/`sinkhorn_distance` below.

GT log-probability is different: it is a property of *one* density (how much
mass it puts on the true future point), not a divergence between two, so it
does not need a shared support -- and forcing one is actively wrong. The
oracle's forecast horizon is *k* steps closer to its own observation than the
update's kept tail, so (for any reasonable flow/diffusion-style predictor)
its density is far more concentrated in the region that is actually scored.
Earlier revisions of this script rasterized the oracle onto the shared grid
above -- sized for the initial, 12-step-out prediction's much broader spread
-- which under-resolves the oracle's tighter density: `griddata_on_cluster`'s
kNN-distance mask (`threshold_pixels=0.5` grid cells) then rejects most of
the grid as extrapolation, `LinearNDInterpolator` returns 0 or NaN at the GT
point far more often, and the `-100` floor silently absorbs the result. That
produced a spurious *monotone collapse* of the oracle's GT log-probability
with k (while the update's, whose kept tail is not concentrating, stayed
flat) -- i.e. it made full recomputation look worse than the cheap
approximation it is supposed to be a ground truth for, which is backwards.
See `own_minmax()` below: for GT log-probability only, each side gets its
*own* grid, extent derived the same way `TrajVisualizer.get_minmax` derives
it in `test.py`'s real per-scene call (obs + gt + a particle-sample bounding
box, then the same 10%-pad-or-5m-box shrink/pad rule), but scoped to the
`n_overlap`-step window actually being scored instead of the whole
`pred_len` horizon -- so the number is computed by the same procedure as
every other log-probability reported in the paper, just sized to the density
actually being read off. The `-100` floor is kept (mirrors
`TrajVisualizer.prob_on_grid`'s `gt_traj_log_prob`), but each record also
carries a `logprob_gt_{update,oracle}_floor_hits` count (out of
`n_overlap_steps`) so any record where flooring still occurs can be flagged
rather than silently averaged in with the rest.

Usage
-----
    pyenv exec python scripts/eval_update_error.py \\
        --config_file configs/traj/FlowMatchingChain_onestep/CIF_spline/eth.yml \\
        --device cuda:2 --n_scenes 3 --update_timesteps 1 2 3 4 5 6 \\
        --out experiments/.../update_error.json
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import ot
from scipy.interpolate import LinearNDInterpolator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from keepflow.utils import load_config
from keepflow.data import build_dataloader
from keepflow.models import build_model
from keepflow.metrics import build_metrics
from keepflow.visualization import build_visualizer
from keepflow.visualization.traj.traj_visualizer import get_grid

from train import dict_to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config_file", type=str, required=True, metavar="FILE",
                    help="path to config file (same convention as train.py/test.py)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--sampling", choices=["even", "head"], default="even",
                   help="'even' spreads the sampled indices over the whole dataset "
                        "(needed for agent/scene diversity: consecutive dataset "
                        "entries are the same agent at consecutive timesteps); "
                        "'head' takes the first n_scenes entries (legacy behaviour)")
    p.add_argument("--num_shards", type=int, default=1,
                   help="split the selected scenes across this many processes "
                        "(one per GPU); concatenating the shards' outputs "
                        "reproduces the unsharded run exactly")
    p.add_argument("--shard", type=int, default=0,
                   help="which shard this process handles, in [0, num_shards)")
    p.add_argument("--n_scenes", type=int, default=-1,
                    help="number of test-set scenes to process, in dataset order "
                         "(-1 = all)")
    p.add_argument("--update_timesteps", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6],
                    help="update steps k to evaluate (each must satisfy "
                         "0 < k < DATA.PREDICT_LENGTH)")
    p.add_argument("--grid_size", type=int, default=100,
                    help="side length of the square evaluation grid (matches the "
                         "100x100 grid used in test.py's density_estimation)")
    p.add_argument("--update_threshold", type=float, default=None,
                   help="override MODEL.FLOW.UPDATE_THRESHOLD (the update gate: the "
                        "update is applied only when deviation_norm <= threshold). "
                        "Configs set this inconsistently -- flagship ETH uses 0.4, the "
                        "ablation configs inherit the default 1000.0, which disables "
                        "the gate -- so it must be equalised for any arm-vs-arm "
                        "comparison. Pass a large value to disable the gate.")
    p.add_argument("--kernel_sigma", choices=["fixed", "cloud"], default="fixed",
                   help="width of the Gaussian that reweights the cached particles. "
                        "'fixed' is the current behaviour: the base variance "
                        "logvar[k], which for FlowChain-CNF never leaves its "
                        "initialisation of 0.1 because the flow-matching objective "
                        "does not touch it. 'cloud' instead uses the empirical "
                        "spread of the cached particles at that step, so the kernel "
                        "follows the forecast it is filtering. No new "
                        "hyperparameter and no oracle: the spread is already in the "
                        "cache.")
    p.add_argument("--kernel_scale", type=float, default=1.0,
                   help="multiplies the width chosen by --kernel_sigma")
    p.add_argument("--out", type=str, default=None,
                    help="output .json or .csv path (extension selects the format); "
                         "defaults to <SAVE_DIR>/update_error.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    # The update gate threshold is NOT set uniformly across configs: the
    # flagship ETH configs set MODEL.FLOW.UPDATE_THRESHOLD to 0.4 while the
    # ablation configs leave it at the default 1000.0, which disables the gate
    # entirely. Comparing two arms without equalising this compares "gated" to
    # "ungated", not the models. Override it explicitly to make the comparison
    # meaningful (and to isolate raw update quality by disabling the gate).
    if args.update_threshold is not None:
        cfg.defrost()
        cfg.MODEL.FLOW.UPDATE_THRESHOLD = args.update_threshold
        cfg.freeze()
    print(f"UPDATE_THRESHOLD = {cfg.MODEL.FLOW.UPDATE_THRESHOLD}")

    out_path = args.out or os.path.join(cfg.SAVE_DIR, "update_error.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    records = run(cfg, args)
    write_records(records, out_path)

    n_ok = sum(1 for r in records if r["status"] == "ok")
    n_skip = len(records) - n_ok
    print(f"wrote {len(records)} records ({n_ok} ok, {n_skip} skipped) to {out_path}")


# --------------------------------------------------------------------------- #
# dataset-index bookkeeping: given the index of the sample used for the
# *initial* prediction, find the dataset *item* that observes the same
# (scene, agent) k timesteps later. Returns a function `(idx, k) ->
# Optional[item]`; `item` is whatever `dataset[i]` normally returns (i.e.
# something `collate_fn([item])` accepts), so the caller doesn't need to
# know how it was constructed.
# --------------------------------------------------------------------------- #

def build_shift_lookup(cfg, dataset):
    if "sim" in cfg.DATA.DATASET_NAME:
        lookup = {}
        for i, (scene, t, node) in enumerate(dataset.index):
            lookup[(scene.name, node.id, int(t))] = i

        def shift_item(idx: int, k: int):
            scene, t, node = dataset.index[idx]
            j = lookup.get((scene.name, node.id, int(t) + k))
            return None if j is None else dataset[j]

        return shift_item

    if cfg.DATA.DATASET_NAME == "jrdb":
        raise NotImplementedError(
            "JRDB_Dataset (keepflow/data/traj/jrdb_dataset.py) was not audited for "
            "this script's index-shift assumptions; refusing to guess. Extend "
            "build_shift_lookup() after checking its indexing scheme.")

    # trajdata.UnifiedDataset, centric="agent": see the module docstring's "How
    # the oracle observation window is obtained" section for why we do *not*
    # use `dataset[idx + k]` here (it under-covers: trajdata's own
    # `AgentDataIndex` only emits a query for `t + k` if a *full* `pred_len`
    # of future exists beyond it, which is stricter than the `pred_len - k`
    # steps this script actually needs). Instead we construct the
    # `AgentBatchElement` for `(scene, agent, t + k)` directly from the raw,
    # cached trajdata scene, which returns however much real future data
    # exists (possibly less than `pred_len`, never fabricated).
    from trajdata.caching.env_cache import EnvCache
    from trajdata.data_structures.scene import SceneTimeAgent
    from trajdata.data_structures.batch_element import AgentBatchElement
    from trajdata.utils import scene_utils

    def _agent_item_at(scene_path: str, agent_id: str, ts: int):
        scene = EnvCache.load(scene_path)
        scene_utils.enforce_desired_dt(scene, dataset.desired_dt)
        if ts < 0 or ts >= len(scene.agent_presence):
            return None

        cache = dataset.cache_class(dataset.cache_path, scene, dataset.augmentations)
        cache.set_obs_format(dataset.obs_format)

        scene_time_agent = SceneTimeAgent.from_cache(
            scene, ts, agent_id, cache,
            only_types=dataset.only_types, no_types=dataset.no_types,
            incl_robot_future=dataset.incl_robot_future,
        )
        if scene_time_agent.agent is None:
            # agent not present at ts: either its track truly ended before
            # ts, or there's a genuine data gap. Either way, no fabrication.
            return None

        return AgentBatchElement(
            cache, -1, scene_time_agent,
            dataset.history_sec, dataset.future_sec,
            dataset.agent_interaction_distances,
            dataset.incl_robot_future, dataset.incl_raster_map,
            dataset.raster_map_params, dataset._map_api,
            dataset.vector_map_params, dataset.state_format,
            dataset.standardize_data, dataset.standardize_derivatives,
            dataset.max_neighbor_num,
        )

    def shift_item(idx: int, k: int):
        scene_path, agent_id, ts = dataset._data_index[idx]
        return _agent_item_at(scene_path, agent_id, ts + k)

    return shift_item


def indices_consistent(orig_index: Tuple, shifted_index: Tuple, k: int) -> bool:
    """orig_index / shifted_index: (scene_id, scene_ts, agent_name) triples as
    produced by CollateWrapper / dict_collate's "index" field."""
    same_scene = orig_index[0] == shifted_index[0]
    same_agent = orig_index[2] == shifted_index[2]
    try:
        same_time = int(shifted_index[1]) == int(orig_index[1]) + k
    except (TypeError, ValueError):
        same_time = False
    return bool(same_scene and same_agent and same_time)


# --------------------------------------------------------------------------- #
# world-frame alignment (see the "Coordinate frames" note in the module
# docstring): metrics.denormalize() only translates; for the trajdata branch
# we additionally undo each query's own heading-alignment rotation so that
# the *update* result (relative to t) and the *oracle* result (relative to
# t+k) land in one shared world frame and can be compared directly.
# --------------------------------------------------------------------------- #

def rotation_for_item(cfg, item, device: str) -> torch.Tensor:
    if "sim" in cfg.DATA.DATASET_NAME:
        return torch.eye(2, dtype=torch.float32, device=device)
    if hasattr(item, "agent_from_world_tf"):
        world_from_agent = np.linalg.inv(item.agent_from_world_tf)
        return torch.tensor(world_from_agent[:2, :2], dtype=torch.float32, device=device)
    raise NotImplementedError(
        f"don't know how to recover the world-frame rotation for dataset item "
        f"type {type(item)}; align_to_world() needs extending before this "
        f"dataset can be used for cross-query comparison.")


def _rotate(pos: torch.Tensor, R: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    local = pos - origin
    rotated = torch.einsum("ij,...j->...i", R, local)
    return rotated + origin


def align_to_world(result_dict: Dict, R: torch.Tensor) -> Dict:
    """Call after metrics.denormalize(). Rotates every position-valued tensor
    about its own standardize_offset by R, undoing the per-query heading
    rotation that denormalize() leaves in place (see module docstring)."""
    origin = result_dict["standardize_offset"][..., 0:2]
    result_dict["gt"] = _rotate(result_dict["gt"], R, origin)
    result_dict["obs"] = _rotate(result_dict["obs"], R, origin)

    if isinstance(result_dict["neighbors"], torch.Tensor) and result_dict["neighbors"].numel() > 0:
        origin_n = result_dict["standardize_offset"][:, None, :, 0:2]
        result_dict["neighbors"] = _rotate(result_dict["neighbors"], R, origin_n)
        result_dict["neighbors_gt"] = _rotate(result_dict["neighbors_gt"], R, origin_n)

    for key in list(result_dict.keys()):
        if isinstance(key, tuple) and key[0] == "prob" and isinstance(result_dict[key], torch.Tensor):
            origin_p = result_dict["standardize_offset"][:, None, :, 0:2]
            xy = _rotate(result_dict[key][..., :2], R, origin_p)
            result_dict[key] = torch.cat([xy, result_dict[key][..., 2:]], dim=-1)

    return result_dict


# --------------------------------------------------------------------------- #
# per-scene motion statistics used for later stratification (E3-2)
# --------------------------------------------------------------------------- #

def compute_motion_stats(data_dict: Dict) -> Dict:
    # obs channels follow DATA.TRAJ.STATE == 'state_pva': [pos(2), vel(2), acc(2)]
    obs = data_dict["obs"][0]  # (obs_len, 6)
    pos = obs[:, 0:2]
    vel = obs[:, 2:4]

    v_t, v_tm1 = vel[-1], vel[-2]
    speed_t, speed_tm1 = v_t.norm().item(), v_tm1.norm().item()

    if speed_t > 1e-6 and speed_tm1 > 1e-6:
        cos_angle = torch.clamp(torch.dot(v_t, v_tm1) / (speed_t * speed_tm1), -1.0, 1.0)
        turn_angle_deg = torch.rad2deg(torch.arccos(cos_angle)).item()
    else:
        turn_angle_deg = float("nan")

    speed_change_ratio = abs(speed_t - speed_tm1) / speed_tm1 if speed_tm1 > 1e-6 else float("nan")
    displacement = (pos[-1] - pos[-2]).norm().item()

    neighbors = data_dict["neighbors"]
    n_close = None
    if isinstance(neighbors, torch.Tensor) and neighbors.numel() > 0:
        neigh_pos_t = neighbors[0, :, -1, 0:2]
        valid = ~torch.isnan(neigh_pos_t).any(dim=-1)
        dist = neigh_pos_t.norm(dim=-1)
        n_close = int(((dist < 2.0) & valid).sum().item())
    elif isinstance(neighbors, torch.Tensor):
        n_close = 0
    elif isinstance(neighbors, dict):
        # 'sim' branch (dict_collate): dict[edge_type] -> [tensor per batch item].
        # Not exercised by the eth smoke test; best-effort, not separately verified.
        n_close = 0
        for tensor_list in neighbors.values():
            if len(tensor_list) == 0:
                continue
            t = tensor_list[0]
            pos_t = t[:, -1, 0:2]
            valid = ~torch.isnan(pos_t).any(dim=-1)
            dist = pos_t.norm(dim=-1)
            n_close += int(((dist < 2.0) & valid).sum().item())

    return {
        "turn_angle_deg": turn_angle_deg,
        "speed_change_ratio": speed_change_ratio,
        "displacement": displacement,
        "n_neighbors_within_2m": n_close,
    }


# --------------------------------------------------------------------------- #
# shared-grid rasterization + divergence metrics
# --------------------------------------------------------------------------- #

def rasterize(visualizer, prob_tensor: torch.Tensor, xx, yy, min_pos, max_pos) -> np.ndarray:
    """prob_tensor: (1, N_particles, T, 3) with last dim [x, y, density].
    Returns (T, grid, grid) raw (non-negative-clipped, un-normalized) density,
    using the exact same per-timestep routine as
    TrajVisualizer.prob_on_grid / TrajVisualizer.griddata_on_cluster."""
    prob_np = visualizer.to_numpy(prob_tensor)
    _, _, timesteps, _ = prob_np.shape
    zz = [visualizer.griddata_on_cluster(0, prob_np, xx, yy, max_pos, min_pos, j)
          for j in range(timesteps)]
    return np.stack(zz, axis=0)


def to_pmf(zz: np.ndarray) -> np.ndarray:
    """(T, grid, grid) raw density -> (T, grid*grid) normalized pmf. Mirrors the
    normalization in TrajMetrics.emd (clip, +1e-12, sum-normalize)."""
    zz = np.clip(zz, a_min=0.0, a_max=None)
    flat = zz.reshape(zz.shape[0], -1) + 1e-12
    return flat / flat.sum(axis=-1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)


def sinkhorn_distance(p: np.ndarray, q: np.ndarray, M: torch.Tensor, device: str) -> np.ndarray:
    p_t = torch.as_tensor(p, dtype=torch.float32, device=device)
    q_t = torch.as_tensor(q, dtype=torch.float32, device=device)
    out = []
    for t in range(p_t.shape[0]):
        # identical call to keepflow/metrics/traj/traj_metrics.py:TrajMetrics.emd
        out.append(ot.sinkhorn2(p_t[t], q_t[t], M, 1.0, warn=False).item())
    return np.array(out)


def gt_log_prob(zz: np.ndarray, xx: np.ndarray, yy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    """log density assigned to the true future position, per timestep. Mirrors
    TrajVisualizer.prob_on_grid's gt_traj_log_prob computation (LinearNDInterpolator
    + a -100 floor for non-finite results). Returns the (possibly floored)
    log-prob per timestep; the caller checks `== -100.0` to count flooring
    (see `logprob_gt_{update,oracle}_floor_hits` below) rather than silently
    averaging contaminated timesteps in with the rest."""
    points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    out = np.full(zz.shape[0], np.nan, dtype=np.float64)
    for t in range(zz.shape[0]):
        interp = LinearNDInterpolator(points, zz[t].flatten())
        val = interp(gt_xy[t, 0], gt_xy[t, 1])
        out[t] = float(val) if val is not None else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_out = np.log(out)
    log_out[~np.isfinite(log_out)] = -100.0
    return log_out


def own_minmax(visualizer, result_dict: Dict, prob_xy: torch.Tensor,
               gt_slice: torch.Tensor, gt_start: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Grid extent for GT log-probability, derived the same way
    `TrajVisualizer.get_minmax` derives it for every other log-probability in
    the paper (`test.py`'s density_estimation -> `prob_on_grid` ->
    `get_minmax`) -- obs + gt + a particle-sample bounding box, then the same
    10%-pad-or-5m-box shrink/pad rule -- except scoped to the `n_overlap`-step
    window actually being scored (`gt_slice`, `prob_xy`) instead of the whole
    `pred_len` horizon, so a since-concentrated (or since-widened) density
    gets a grid sized to *its own* spread rather than inheriting whatever
    extent the *other* side's full-horizon forecast happened to need.

    `prob_xy`/`gt_slice` must already be in world-frame coordinates (i.e.
    read off `result_dict` post `align_to_world`), since `get_minmax` has no
    notion of the update/oracle rotation split -- passing pre-alignment
    tensors here would silently mix frames and produce a meaningless box.

    prob_xy:  (1, N_particles, n_overlap, 2) -- the same tensor `rasterize()`
              is called on for this side.
    gt_slice: (1, n_overlap, 2).
    """
    # get_minmax concatenates the agent's [obs | gt] with the neighbours'
    # [neighbors | neighbors_gt] along the agent axis, so both must span the
    # same number of timesteps. `gt_slice` is only `n_overlap` long, so the
    # neighbours' future has to be cut to the *same* window (`gt_start` is the
    # offset of `gt_slice` within the full pred_len horizon: `k` for the
    # update side, 0 for the oracle side). Without this the concat raises
    # "Sizes of tensors must match" as soon as k > 0.
    neighbors = result_dict["neighbors"]
    neighbors_gt = result_dict["neighbors_gt"]
    if len(neighbors_gt) != 0:
        n_overlap = gt_slice.shape[1]
        neighbors_gt = neighbors_gt[:, :, gt_start:gt_start + n_overlap]

    sub = {
        "index": result_dict["index"],  # only read by get_minmax's 'sim' branch
        "obs": result_dict["obs"],
        "gt": gt_slice,
        "neighbors": neighbors,
        "neighbors_gt": neighbors_gt,
        ("pred", 0): prob_xy,
    }
    return visualizer.get_minmax(sub, visualize=False)


def cost_matrix(xx: np.ndarray, yy: np.ndarray, device: str) -> torch.Tensor:
    coords = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1),
                           dtype=torch.float32, device=device)
    coords_sqr = (coords ** 2).sum(dim=1)
    M = coords_sqr[:, None] + coords_sqr[None, :] - 2 * coords @ coords.T
    return M.clamp(min=0).sqrt()


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #

def effective_sample_size(model, result0: Dict, k: int) -> Dict:
    """Kish effective sample size of the update's importance weights.

    `update_from_new_obs` is self-normalized importance reweighting: the cached
    particles stay put and only their weights change, the weight of particle i
    being the new base density N(x_hat_{t+k}, sigma_k^2) evaluated at that
    particle's position at step k-1. The weights therefore already exist inside
    the update; ESS = 1 / sum_i w_i^2 costs one extra reduction and no network
    evaluation at all.

    ESS measures directly what the current `deviation_norm` gate only proxies:
    how many of the cached particles still carry usable mass after the new
    observation arrives. A collapsed ESS means the update is extrapolating from
    a handful of particles, which is exactly when it should fall back to a full
    recomputation.

    Recomputed here rather than returned by `update_from_new_obs` so that the
    model's behaviour is left untouched by the measurement.
    """
    with torch.no_grad():
        node = model.get_node_type(result0)
        particles = result0[("prob", 0)][:, :, k - 1, :2]          # (B, N, 2)
        log_w = model.flow[node].base_dist(
            result0["gt"][:, k - 1], step=k).log_prob(particles).sum(dim=-1)  # (B, N)
        log_w = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
        ess = torch.exp(-torch.logsumexp(2.0 * log_w, dim=1))       # (B,)
        n_particles = particles.shape[1]
        return {"ess": ess.mean().item(),
                "ess_frac": (ess.mean() / n_particles).item(),
                "n_particles": n_particles,
                "max_weight": torch.exp(log_w.max()).item()}


def run(cfg, args) -> List[Dict]:
    model = build_model(cfg)
    try:
        model.load()
    except FileNotFoundError:
        raise SystemExit(f"no checkpoint found at {model.model_path}; train the model first")
    model.eval()

    metrics = build_metrics(cfg)
    visualizer = build_visualizer(cfg)

    loader = build_dataloader(cfg, rand=False, split="test", batch_size=1)
    dataset = loader.dataset
    collate_fn = loader.collate_fn
    shift_item = build_shift_lookup(cfg, dataset)

    pred_len = cfg.DATA.PREDICT_LENGTH
    ks = sorted(k for k in args.update_timesteps if 0 < k < pred_len)
    dropped = sorted(set(args.update_timesteps) - set(ks))
    if dropped:
        print(f"WARNING: dropping update_timesteps outside (0, {pred_len}): {dropped}")

    n_scenes = len(dataset) if args.n_scenes in (None, -1) else min(args.n_scenes, len(dataset))

    # The dataset is indexed by (scene, agent, timestep) with consecutive
    # entries being the SAME agent at consecutive times, so taking the first
    # n_scenes entries yields a handful of agents sampled densely in time
    # rather than a cross-section of the data. Spreading the indices evenly
    # over the whole dataset gives agent/scene diversity for the same cost.
    if args.sampling == "even" and n_scenes < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, n_scenes).round().astype(int).tolist()
        indices = sorted(set(indices))
    else:
        indices = list(range(n_scenes))

    # Optional sharding so one sweep can be split across several GPUs. The
    # scene list is built first and *then* strided, so the union of all shards
    # is exactly the unsharded run -- concatenating the per-shard JSON files
    # reproduces it record for record.
    if args.num_shards > 1:
        indices = indices[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(indices)} scenes",
              flush=True)

    records: List[Dict] = []
    with torch.no_grad():
        for count, idx in enumerate(indices, 1):
            # flush: this script is normally run with stdout redirected to a
            # log file, where block buffering would hide progress until exit.
            print(f"scene {count}/{len(indices)} (dataset idx {idx})", flush=True)
            item = dataset[idx]
            data_dict = dict_to_device(collate_fn([item]), cfg.DEVICE)
            index0 = data_dict["index"][0]

            motion_stats = compute_motion_stats(data_dict)

            result0 = model.predict(data_dict, return_prob=True)

            # Compute the (approximate) updates *and* the gate decision for
            # every k before touching the oracle path: model.predict() below
            # (for the oracle) overwrites model.ldjs_tmp /
            # model.base_prob_normalizer_tmp in place, which
            # update_from_new_obs relies on.
            gate_info = {}
            _node = model.get_node_type(result0)
            _flow = model.flow[_node]
            _saved_logvar = _flow.logvar.data.clone()
            if args.kernel_sigma == "cloud":
                # Particle spread at each step, measured from the cache. The kernel
                # for update k weights the particles at step k-1, so shift by one.
                _spread = result0[("prob", 0)][0, :, :, :2].std(dim=0)   # (T, dim)
                _w = _spread.clone()
                _w[1:] = _spread[:-1]
                _flow.logvar.data = (args.kernel_scale * _w).clamp(min=1e-3).log()
            for k in ks:
                deviation = torch.norm(
                    result0[("prob", 0)][:, :, k - 1, :2] - result0["gt"][:, k - 1], dim=-1)
                base_pos = model._get_base_pos(result0)
                gt_norm = torch.norm(result0["gt"][:, k - 1] - base_pos, dim=-1)
                if gt_norm.item() < 0.01:
                    gate_info[k] = {"gate_active": False, "deviation_norm": None,
                                     "gate_reason": "near_zero_gt_norm"}
                else:
                    deviation_norm = (deviation / gt_norm).mean().item()
                    gate_info[k] = {
                        "gate_active": deviation_norm <= cfg.MODEL.FLOW.UPDATE_THRESHOLD,
                        "deviation_norm": deviation_norm, "gate_reason": None}
                gate_info[k].update(effective_sample_size(model, result0, k))
                result0 = model.update_from_new_obs(result0, k)

            # Restore before the oracle predictions below, which sample from the
            # base distribution and must not see the modified width.
            _flow.logvar.data = _saved_logvar

            result0 = metrics.denormalize(result0)
            result0 = align_to_world(result0, rotation_for_item(cfg, item, cfg.DEVICE))
            min_pos, max_pos = visualizer.get_minmax(result0, visualize=False)
            xx, yy = get_grid(args.grid_size, min_pos, max_pos)
            M = cost_matrix(xx, yy, cfg.DEVICE)

            for k in ks:
                base_record = {
                    "scene_idx": idx, "scene_id": index0[0], "agent": index0[2],
                    "t": int(index0[1]), "k": k, **motion_stats, **gate_info[k],
                }

                oracle_item = shift_item(idx, k)
                if oracle_item is None:
                    records.append({**base_record, "status": "skipped",
                                     "skip_reason": "shift_unavailable"})
                    continue

                oracle_dict = dict_to_device(collate_fn([oracle_item]), cfg.DEVICE)
                if not indices_consistent(index0, oracle_dict["index"][0], k):
                    records.append({**base_record, "status": "skipped",
                                     "skip_reason": "shift_mismatch"})
                    continue

                oracle_result = model.predict(oracle_dict, return_prob=True)
                oracle_result = metrics.denormalize(oracle_result)
                oracle_result = align_to_world(
                    oracle_result, rotation_for_item(cfg, oracle_item, cfg.DEVICE))

                n_overlap = pred_len - k
                upd_prob = result0[("prob", k)][:, :, :n_overlap]
                ora_prob = oracle_result[("prob", 0)][:, :, :n_overlap]

                # Shared-grid path: KL/Sinkhorn are divergences *between* the
                # two densities, so they need one common support (`xx`/`yy`,
                # fixed once per scene above from the initial prediction).
                zz_upd = rasterize(visualizer, upd_prob, xx, yy, min_pos, max_pos)
                zz_ora = rasterize(visualizer, ora_prob, xx, yy, min_pos, max_pos)
                pmf_upd, pmf_ora = to_pmf(zz_upd), to_pmf(zz_ora)

                kl_ou = kl_divergence(pmf_ora, pmf_upd)  # KL(oracle || update)
                kl_uo = kl_divergence(pmf_upd, pmf_ora)  # KL(update || oracle)
                sinkhorn = sinkhorn_distance(pmf_upd, pmf_ora, M, cfg.DEVICE)

                gt_upd_t = result0["gt"][:, k:k + n_overlap]     # (1, n_overlap, 2), world frame
                gt_ora_t = oracle_result["gt"][:, :n_overlap]    # (1, n_overlap, 2), world frame
                gt_upd = gt_upd_t[0].cpu().numpy()
                gt_ora = gt_ora_t[0].cpu().numpy()

                # Own-grid path: GT log-probability scores a single density
                # against the truth, not a divergence between the two, so it
                # does not need (and must not use) the shared grid above --
                # see own_minmax()/the module docstring for why reusing the
                # shared grid here spuriously collapses the oracle's score as
                # k grows. Each side gets its own extent, derived the same
                # way test.py's density-estimation path derives one.
                min_pos_u, max_pos_u = own_minmax(visualizer, result0, upd_prob[..., :2],
                                                  gt_upd_t, gt_start=k)
                xx_u, yy_u = get_grid(args.grid_size, min_pos_u, max_pos_u)
                zz_upd_own = rasterize(visualizer, upd_prob, xx_u, yy_u, min_pos_u, max_pos_u)
                logp_upd = gt_log_prob(zz_upd_own, xx_u, yy_u, gt_upd)

                min_pos_o, max_pos_o = own_minmax(visualizer, oracle_result, ora_prob[..., :2], gt_ora_t)
                xx_o, yy_o = get_grid(args.grid_size, min_pos_o, max_pos_o)
                zz_ora_own = rasterize(visualizer, ora_prob, xx_o, yy_o, min_pos_o, max_pos_o)
                logp_ora = gt_log_prob(zz_ora_own, xx_o, yy_o, gt_ora)

                records.append({
                    **base_record,
                    "status": "ok", "skip_reason": None,
                    "n_overlap_steps": int(n_overlap),
                    "kl_oracle_to_update_mean": float(np.nanmean(kl_ou)),
                    "kl_update_to_oracle_mean": float(np.nanmean(kl_uo)),
                    "sinkhorn_mean": float(np.nanmean(sinkhorn)),
                    "logprob_gt_update_mean": float(np.nanmean(logp_upd)),
                    "logprob_gt_oracle_mean": float(np.nanmean(logp_ora)),
                    "logprob_gt_diff_mean": float(np.nanmean(logp_upd - logp_ora)),
                    # diagnostic: how many of the n_overlap timesteps hit the
                    # -100 floor (non-finite interpolated density) on each
                    # side's own grid, so contaminated records can be
                    # identified rather than silently averaged in.
                    "logprob_gt_update_floor_hits": int(np.sum(logp_upd <= -100.0)),
                    "logprob_gt_oracle_floor_hits": int(np.sum(logp_ora <= -100.0)),
                    "gt_consistency_max_abs_diff": float(np.nanmax(np.abs(gt_upd - gt_ora))),
                })

    return records


def write_records(records: List[Dict], out_path: str) -> None:
    if out_path.endswith(".csv"):
        fieldnames = sorted({k for r in records for k in r.keys()})
        with open(out_path, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    else:
        with open(out_path, "w") as fp:
            json.dump(records, fp, indent=2)


if __name__ == "__main__":
    main()
