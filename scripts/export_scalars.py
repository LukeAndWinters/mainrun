from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse, pathlib, matplotlib.pyplot as plt, pandas as pd
from datetime import datetime
import os

def plot_series(tag, ea, out_png):
    steps, vals = [], []
    for e in ea.Scalars(tag): steps.append(e.step); vals.append(e.value)
    if not steps: return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.plot(steps, vals)
    plt.xlabel("step"); plt.ylabel(tag); plt.title(tag); plt.tight_layout()
    plt.savefig(out_png, dpi=180); plt.close(); return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default="docs/figures")
    ap.add_argument("--run_name", default=None, help="Optional run name for file identification")
    args = ap.parse_args()
    run = pathlib.Path(args.run_dir); out = pathlib.Path(args.out_dir)
    
    # Generate run identifier: use run_name if provided, otherwise use timestamp of event file
    if args.run_name:
        run_id = args.run_name
    else:
        # Find the TensorBoard event file and use its timestamp
        event_files = list(run.glob("events.out.tfevents.*"))
        if event_files:
            # Extract timestamp from event file
            event_file = event_files[0]
            mtime = os.path.getmtime(event_file)
            run_id = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
        else:
            # Fallback to current timestamp
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    ea = EventAccumulator(str(run)); ea.Reload()
    for tag in ["loss/train","loss/val","lr","perf/tokens_per_sec","metrics/perplexity"]:
        plot_series(tag, ea, out / f"{run_id}_{tag.replace('/','_')}.png")
    rows=[]
    for tag in ea.Tags()["scalars"]:
        for s in ea.Scalars(tag):
            rows.append({"tag": tag, "step": s.step, "value": s.value})
    if rows: pd.DataFrame(rows).to_csv(out / f"{run_id}_scalars.csv", index=False)
    print(f"Exported snapshots with run ID: {run_id}")

if __name__ == "__main__": main()
