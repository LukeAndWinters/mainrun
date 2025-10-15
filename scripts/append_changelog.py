import argparse, pathlib, subprocess
TEMPLATE = """
## {exp_name} — {title}
- **Commit:** `{commit}`
- **Change:** {change}
- **Rationale:** {rationale}
- **Key settings:** `{settings}`
- **Result:** best val loss = **{best_val:.4f}**
- **Figures:**  
  ![val loss](figures/{file_prefix}_loss_val.png)  
  ![train loss](figures/{file_prefix}_loss_train.png)
"""
def git_commit_hash():
    return subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--change", required=True)
    ap.add_argument("--rationale", required=True)
    ap.add_argument("--settings", default="")
    ap.add_argument("--best_val", type=float, required=True)
    args=ap.parse_args()
    md = pathlib.Path("docs/CHANGELOG.md")
    if not md.exists(): md.write_text("# Experiment ChangeLog\n\n")
    entry=TEMPLATE.format(
        exp_name=args.exp_name, title=args.title, change=args.change,
        rationale=args.rationale, settings=args.settings, best_val=args.best_val,
        file_prefix=args.exp_name, commit=git_commit_hash()
    )
    with md.open("a") as f: f.write(entry+"\n")
if __name__=="__main__": main()