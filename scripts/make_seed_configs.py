"""Generate per-seed copies of a config directory for multi-seed experiments.

Rationale
---------
`SEED` is a config key (`keepflow/utils/default_params.py`), and since the fix in
`keepflow/utils/param_utils.py` an override in the YAML actually takes effect.
Because `SAVE_DIR` is derived from the config path, a seed variant must live at a
distinct path so that checkpoints and metrics do not collide.

Two traps this script handles:

1. `LOAD_TUNED` (default True) makes `load_config` pull hyperparameters from
   `<SAVE_DIR>/optuna.db` when one exists. A freshly generated seed directory has
   no such database, so the seed runs would silently use *different*
   hyperparameters from the ones behind the published numbers. This script copies
   the tuned values into the generated config and sets `LOAD_TUNED: False`, so the
   settings are explicit and reproducible from the config file alone.

2. `load_tuned_params` runs *after* `merge_from_file` and assigns
   unconditionally, so a tuned value **overrides** the YAML value (e.g. a config
   saying `WEIGHT_DECAY: 0.0` still runs with the tuned 4.57e-7). This script
   reproduces that precedence exactly.

Usage
-----
    python scripts/make_seed_configs.py \
        --config_dir configs/traj/FlowMatchingChain_onestep/CIF_spline \
        --datasets eth hotel univ zara1 zara2 \
        --seeds 2 3 \
        --out_root configs_seed
"""
import argparse
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keys recovered from optuna.db that still exist in the current schema.
# Anything else in the study (e.g. the retired MODEL.FLOW.CONDITIONING_LENGTH)
# is reported and skipped rather than written into the config.
KNOWN_KEYS = {
    "MODEL.FLOW.N_HIDDEN",
    "MODEL.FLOW.HIDDEN_SIZE",
    "MODEL.FLOW.COND_SIZE",
    "MODEL.FLOW.N_BLOCKS",
    "MODEL.FLOW.BASE_SIGMA",
    "SOLVER.LR",
    "SOLVER.WEIGHT_DECAY",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config_dir", required=True,
                   help="source config directory, e.g. configs/traj/.../CIF_spline")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--out_root", default="configs_seed")
    p.add_argument("--save_dir", default="experiments",
                   help="must match SAVE_DIR used for the original runs")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def tuned_params(save_dir: Path, config_rel: Path) -> dict:
    """Best-trial params from <save_dir>/<config without suffix>/optuna.db, if any."""
    db = save_dir / config_rel.with_suffix("") / "optuna.db"
    if not db.exists():
        return {}
    import optuna
    study = optuna.load_study(storage="sqlite:///" + str(db), study_name="my_opt")
    return dict(study.best_trial.params)


def set_nested(tree: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = tree
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def get_nested(tree: dict, dotted: str):
    node = tree
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def main() -> None:
    args = parse_args()
    src_dir = Path(args.config_dir)
    save_dir = Path(args.save_dir)

    written, skipped = 0, []
    for ds in args.datasets:
        src = src_dir / f"{ds}.yml"
        if not src.exists():
            skipped.append(f"missing config: {src}")
            continue

        cfg = yaml.safe_load(src.read_text()) or {}
        rel = src.relative_to(src.parts[0]) if src.is_absolute() else src
        tuned = tuned_params(save_dir, Path(*src.parts))

        applied, ignored = {}, {}
        for k, v in tuned.items():
            if k not in KNOWN_KEYS:
                ignored[k] = v
                continue
            # load_tuned_params assigns after merge_from_file, so the tuned
            # value wins even when the YAML sets the key explicitly.
            applied[k] = v

        if tuned:
            print(f"[{src}] optuna.db found: applying {applied}"
                  + (f"; ignoring retired keys {ignored}" if ignored else ""))

        for seed in args.seeds:
            out = Path(args.out_root) / f"s{seed}" / src_dir / f"{ds}.yml"
            new = yaml.safe_load(src.read_text()) or {}
            for k, v in applied.items():
                set_nested(new, k, v)
            new["SEED"] = seed
            # Pin hyperparameters: the generated SAVE_DIR has no optuna.db, so
            # leaving LOAD_TUNED on would silently change the configuration.
            new["LOAD_TUNED"] = False

            print(f"  -> {out}")
            if not args.dry_run:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(yaml.safe_dump(new, sort_keys=False,
                                              default_flow_style=False))
            written += 1

    print(f"\n{written} config(s) written under {args.out_root}/")
    for s in skipped:
        print(f"  WARNING: {s}")


if __name__ == "__main__":
    main()
