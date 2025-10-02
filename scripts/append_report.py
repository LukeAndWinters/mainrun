#!/usr/bin/env python3
"""
Append a standardized experiment section to mainrun/report.md without overwriting
previous content.

Usage:
  python3 scripts/append_report.py \
    --exp_name EXP \
    --title "Title" \
    --change "What changed" \
    --rationale "Why" \
    --settings "k=v,..." \
    --best_val 1.2345
"""
import argparse, pathlib

TEMPLATE = """
## {exp_name} — {title}
- Change: {change}
- Rationale: {rationale}
- Settings: `{settings}`
- Best val loss: **{best_val:.4f}**

Figures:
![Validation Loss](../docs/figures/{file_prefix}_loss_val.png)
![Training Loss](../docs/figures/{file_prefix}_loss_train.png)
![Learning Rate](../docs/figures/{file_prefix}_lr.png)
![Tokens/sec](../docs/figures/{file_prefix}_perf_tokens_per_sec.png)
![Perplexity](../docs/figures/{file_prefix}_metrics_perplexity.png)

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--change", required=True)
    ap.add_argument("--rationale", required=True)
    ap.add_argument("--settings", default="")
    ap.add_argument("--best_val", type=float, required=True)
    args = ap.parse_args()

    report_md = pathlib.Path("mainrun/report.md")
    if not report_md.exists():
        report_md.write_text("MainRun Report\n\n")

    section = TEMPLATE.format(
        exp_name=args.exp_name,
        title=args.title,
        change=args.change,
        rationale=args.rationale,
        settings=args.settings,
        best_val=args.best_val,
        file_prefix=args.exp_name,
    )
    with report_md.open("a", encoding="utf-8") as f:
        f.write(section)


if __name__ == "__main__":
    main()


