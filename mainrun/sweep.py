#!/usr/bin/env python3
"""
Resumable, concurrent sweep runner for MainRun.

WHAT: Executes randomized hyperparameter trials with concurrency control,
      resumes safely by skipping completed runs, and logs all results.
WHY: Explore a targeted space to minimize final validation loss in exactly 7 epochs.

Outputs:
- logs/large_sweep_plan.json        (planned trials)
- logs/large_sweep.jsonl            (one JSON per trial result)
- logs/large_sweep_summary.csv      (sorted leaderboard by final_val, rebound)
"""

import argparse
import concurrent.futures as futures
import itertools
import json
import os
import random
import string
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "mainrun" / "runs"
LOGS = ROOT / "mainrun" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


def short_id(length: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choice(alphabet) for _ in range(length))


def build_trials() -> list[dict]:
    # Core scheduler priors (focus near proven best)
    lr_list = [0.0039, 0.0042, 0.0045, 0.0048]
    eta_min_list = [0.0, 0.0025, 0.005, 0.0075, 0.01]
    warmup_list = [0.20, 0.25, 0.30]
    tail_pct_list = [0.08, 0.10, 0.12, 0.15]
    beta2_tail_list = [None, 0.99, 0.995]
    tail_beta2_start_list = [0.20, 0.30, 0.35]
    # Fixed
    grad_accum_list = [2]

    # Low-risk toggles (sparsely sampled later)
    residual_scale_list = [False, True]
    mlp_activation_list = ["gelu", "swiglu"]
    pack_tokens_list = [False, True]
    dropout_list = [0.05, 0.10]
    weight_decay_list = [0.08, 0.10, 0.12]

    # Build base grid of core scheduler params
    grid = list(itertools.product(
        lr_list, eta_min_list, warmup_list, tail_pct_list,
        beta2_tail_list, tail_beta2_start_list, grad_accum_list
    ))

    # Randomly sample ~100 from grid (focus near center by biasing sampling)
    random.shuffle(grid)
    base_trials = []
    for (lr, eta_min, warmup, tail_pct, beta2_tail, beta2_start, accum) in grid[:110]:
        base_trials.append({
            "lr": lr,
            "eta_min_factor": eta_min,
            "warmup_pct": warmup,
            "tail_squeeze": True,
            "tail_squeeze_pct": tail_pct,
            "beta2_tail": beta2_tail,
            "tail_beta2_start_pct": beta2_start,
            "grad_accum_steps": accum,
            # defaults for toggles (off)
            "residual_scale": False,
            "mlp_activation": "gelu",
            "pack_tokens": False,
            "dropout": None,
            "weight_decay": None,
        })

    # Add micro-branches (architecture/regularization toggles) for 40 trials sampled from base
    micro_base = random.sample(base_trials, k=min(40, len(base_trials)))
    micro_trials = []
    for t in micro_base:
        micro_trials.append({**t, "residual_scale": True})
        micro_trials.append({**t, "mlp_activation": "swiglu"})
        micro_trials.append({**t, "pack_tokens": True})
        micro_trials.append({**t, "dropout": random.choice([0.05, 0.10])})
        micro_trials.append({**t, "weight_decay": random.choice([0.08, 0.10, 0.12])})
    # Cap total trials to ~160
    trials = base_trials + micro_trials
    trials = trials[:160]

    # Save plan
    (LOGS / "large_sweep_plan.json").write_text(json.dumps(trials, indent=2))
    return trials


def trial_run(args: dict, max_seconds: float | None = None) -> dict:
    run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + "_" + short_id(5)
    run_dir = RUNS / run_id
    if run_dir.exists():
        # ensure uniqueness
        run_dir = RUNS / (run_id + "_" + short_id(3))
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        sys.executable, str(ROOT / "mainrun" / "train.py"),
        "--lr", str(args["lr"]),
        "--eta_min_factor", str(args["eta_min_factor"]),
        "--warmup_pct", str(args["warmup_pct"]),
        "--grad_accum_steps", str(args["grad_accum_steps"]),
        "--tail_squeeze",
        "--tail_squeeze_pct", str(args["tail_squeeze_pct"]),
    ]
    if args.get("beta2_tail") is not None:
        cmd += ["--beta2_tail", str(args["beta2_tail"])]
    cmd += ["--tail_beta2_start_pct", str(args["tail_beta2_start_pct"])]
    if args.get("residual_scale"): cmd += ["--residual_scale"]
    if args.get("mlp_activation"): cmd += ["--mlp_activation", str(args["mlp_activation"])]
    if args.get("pack_tokens"): cmd += ["--pack_tokens"]
    if args.get("dropout") is not None: cmd += ["--dropout", str(args["dropout"])]
    if args.get("weight_decay") is not None: cmd += ["--weight_decay", str(args["weight_decay"])]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        duration = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"fail({proc.returncode})"
        # try parse result.json from latest symlink or by locating most recent run
        result_path = run_dir / "result.json"
        if not result_path.exists():
            # fallback: try latest symlink if our run_dir naming diverged
            latest = RUNS / "latest"
            if latest.is_symlink():
                lp = latest / "result.json"
                if lp.exists():
                    result_path = lp
        rec = {
            "run_id": run_dir.name,
            "cmd": " ".join(cmd),
            "lr": args["lr"],
            "eta_min_factor": args["eta_min_factor"],
            "warmup_pct": args["warmup_pct"],
            "tail_squeeze": True,
            "tail_squeeze_pct": args["tail_squeeze_pct"],
            "grad_accum_steps": args["grad_accum_steps"],
            "beta2_tail": args.get("beta2_tail"),
            "tail_beta2_start_pct": args["tail_beta2_start_pct"],
            "residual_scale": args.get("residual_scale", False),
            "mlp_activation": args.get("mlp_activation", "gelu"),
            "pack_tokens": args.get("pack_tokens", False),
            "dropout": args.get("dropout"),
            "weight_decay": args.get("weight_decay"),
            "duration_sec": duration,
            "status": status,
        }
        final_val = best_val = rebound = skipped = None
        lr_min=lr_max=None; last5=None
        if result_path.exists():
            try:
                data = json.loads(result_path.read_text())
                final_val = data.get("final_val_loss")
                best_val = data.get("best_val_loss")
                rebound = data.get("rebound")
                skipped = data.get("skipped_updates")
                lr_min = data.get("lr_min"); lr_max = data.get("lr_max"); last5 = data.get("last5_lrs")
            except Exception:
                pass
        rec.update({
            "final_val": final_val,
            "best_val": best_val,
            "rebound": None if (final_val is None or best_val is None) else final_val - best_val if rebound is None else rebound,
            "skipped_updates": skipped,
            "lr_min": lr_min, "lr_max": lr_max, "last5_lrs": last5,
        })
        with open(LOGS / "large_sweep.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
    except KeyboardInterrupt:
        raise
    except Exception as e:
        duration = time.time() - t0
        rec = {"run_id": run_dir.name, "cmd": " ".join(cmd), "status": f"exception: {e}", "duration_sec": duration}
        with open(LOGS / "large_sweep.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_concurrent", type=int, default=2)
    args = ap.parse_args()

    trials = build_trials()
    random.shuffle(trials)

    # Resume: skip trials that have matching result in jsonl by cmd signature
    done_cmds = set()
    jl = LOGS / "large_sweep.jsonl"
    if jl.exists():
        with open(jl) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "cmd" in obj:
                        done_cmds.add(obj["cmd"])
                except Exception:
                    pass

    pending = []
    for t in trials:
        # pre-build command string to dedupe resume
        cmd_sig = " ".join([
            "python3", str(ROOT / "mainrun" / "train.py"),
            "--lr", str(t["lr"]),
            "--eta_min_factor", str(t["eta_min_factor"]),
            "--warmup_pct", str(t["warmup_pct"]),
            "--grad_accum_steps", str(t["grad_accum_steps"]),
            "--tail_squeeze", "--tail_squeeze_pct", str(t["tail_squeeze_pct"]),
            *( ["--beta2_tail", str(t["beta2_tail"])] if t.get("beta2_tail") is not None else [] ),
            "--tail_beta2_start_pct", str(t["tail_beta2_start_pct"]),
            *( ["--residual_scale"] if t.get("residual_scale") else [] ),
            "--mlp_activation", str(t.get("mlp_activation","gelu")),
            *( ["--pack_tokens"] if t.get("pack_tokens") else [] ),
            *( ["--dropout", str(t["dropout"]) ] if t.get("dropout") is not None else [] ),
            *( ["--weight_decay", str(t["weight_decay"]) ] if t.get("weight_decay") is not None else [] ),
        ])
        if cmd_sig in done_cmds:
            continue
        pending.append(t)

    print(f"Planned trials: {len(trials)} | Pending after resume: {len(pending)}")

    results = []
    try:
        with futures.ThreadPoolExecutor(max_workers=max(1, args.max_concurrent)) as pool:
            futs = [pool.submit(trial_run, t) for t in pending]
            for fut in futures.as_completed(futs):
                results.append(fut.result())
    except KeyboardInterrupt:
        print("\nSweep interrupted. Generating partial summary...")

    # Write summary CSV
    import csv
    rows = []
    if jl.exists():
        with open(jl) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    # sort by final_val then rebound
    def sort_key(r):
        fv = r.get("final_val")
        rb = r.get("rebound")
        return (float('inf') if fv is None else fv, float('inf') if rb is None else rb)
    rows_sorted = sorted(rows, key=sort_key)
    csv_path = LOGS / "large_sweep_summary.csv"
    cols = [
        "run_id","final_val","best_val","rebound","lr","eta_min_factor","warmup_pct",
        "tail_squeeze_pct","beta2_tail","tail_beta2_start_pct","grad_accum_steps",
        "residual_scale","mlp_activation","pack_tokens","dropout","weight_decay",
        "lr_min","lr_max","last5_lrs","duration_sec","status"
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_sorted:
            w.writerow({c: r.get(c) for c in cols})
    print(f"Summary written: {csv_path}")

    # Print top 15
    print("\nTop 15 trials (by final_val then rebound):")
    for r in rows_sorted[:15]:
        print(json.dumps({k: r.get(k) for k in cols}, ensure_ascii=False))


if __name__ == "__main__":
    main()


