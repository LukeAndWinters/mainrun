#!/usr/bin/env python3
"""
Update image paths in report.md to use new subdirectory structure.
"""
import re
import pathlib

def update_report_paths():
    report_path = pathlib.Path("mainrun/report.md")
    if not report_path.exists():
        print("Report file not found")
        return
    
    content = report_path.read_text()
    
    # Define path mappings
    path_mappings = {
        r'../docs/figures/20251002_083200_': '../docs/figures/baseline_20251002_083200/',
        r'../docs/figures/adamw_warmup_01_': '../docs/figures/adamw_warmup_01/',
        r'../docs/figures/amp_02_': '../docs/figures/amp_02/',
        r'../docs/figures/baseline_20251002_': '../docs/figures/baseline_20251002/',
        r'../docs/figures/baseline_v1_': '../docs/figures/baseline_v1/',
        r'../docs/figures/logs_': '../docs/figures/logs_baseline/',
    }
    
    # Apply mappings
    for old_pattern, new_pattern in path_mappings.items():
        content = re.sub(old_pattern, new_pattern, content)
    
    # Write updated content
    report_path.write_text(content)
    print("Updated report.md with new image paths")

if __name__ == "__main__":
    update_report_paths()
