#!/usr/bin/env python3
"""
Add date prefixes to existing figure files and update report paths.
"""
import pathlib
import shutil
import re
from datetime import datetime

def add_dates_to_existing():
    figures_dir = pathlib.Path("docs/figures")
    if not figures_dir.exists():
        print("No figures directory found")
        return
    
    # Define experiment date mappings (you can adjust these dates as needed)
    experiment_dates = {
        "baseline_20251002_083200": "20251002",
        "adamw_warmup_01": "20251002", 
        "amp_02": "20251015",
        "baseline_20251002": "20251002",
        "baseline_v1": "20251002",
        "logs_baseline": "20251002"
    }
    
    # Process each experiment directory
    for exp_dir in figures_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name in experiment_dates:
            date_prefix = experiment_dates[exp_dir.name]
            print(f"Processing {exp_dir.name} with date {date_prefix}")
            
            # Rename files in this directory
            for file_path in exp_dir.iterdir():
                if file_path.is_file():
                    filename = file_path.name
                    # Skip if already has date prefix
                    if filename.startswith(date_prefix + "_"):
                        continue
                    
                    # Add date prefix
                    new_filename = f"{date_prefix}_{filename}"
                    new_path = exp_dir / new_filename
                    print(f"  Renaming {filename} -> {new_filename}")
                    shutil.move(str(file_path), str(new_path))
    
    print("Date prefixes added to existing files")

def update_report_paths_with_dates():
    """Update report.md to use new dated filenames"""
    report_path = pathlib.Path("mainrun/report.md")
    if not report_path.exists():
        print("Report file not found")
        return
    
    content = report_path.read_text()
    
    # Define path mappings with dates
    path_mappings = {
        r'../docs/figures/baseline_20251002_083200/loss_val\.png': '../docs/figures/baseline_20251002_083200/20251002_loss_val.png',
        r'../docs/figures/baseline_20251002_083200/loss_train\.png': '../docs/figures/baseline_20251002_083200/20251002_loss_train.png',
        r'../docs/figures/baseline_20251002_083200/lr\.png': '../docs/figures/baseline_20251002_083200/20251002_lr.png',
        r'../docs/figures/baseline_20251002_083200/metrics_perplexity\.png': '../docs/figures/baseline_20251002_083200/20251002_metrics_perplexity.png',
        r'../docs/figures/baseline_20251002_083200/perf_tokens_per_sec\.png': '../docs/figures/baseline_20251002_083200/20251002_perf_tokens_per_sec.png',
        
        r'../docs/figures/adamw_warmup_01/loss_val\.png': '../docs/figures/adamw_warmup_01/20251002_loss_val.png',
        r'../docs/figures/adamw_warmup_01/loss_train\.png': '../docs/figures/adamw_warmup_01/20251002_loss_train.png',
        r'../docs/figures/adamw_warmup_01/lr\.png': '../docs/figures/adamw_warmup_01/20251002_lr.png',
        r'../docs/figures/adamw_warmup_01/metrics_perplexity\.png': '../docs/figures/adamw_warmup_01/20251002_metrics_perplexity.png',
        r'../docs/figures/adamw_warmup_01/perf_tokens_per_sec\.png': '../docs/figures/adamw_warmup_01/20251002_perf_tokens_per_sec.png',
        
        r'../docs/figures/amp_02/loss_val\.png': '../docs/figures/amp_02/20251015_loss_val.png',
        r'../docs/figures/amp_02/loss_train\.png': '../docs/figures/amp_02/20251015_loss_train.png',
        r'../docs/figures/amp_02/lr\.png': '../docs/figures/amp_02/20251015_lr.png',
        r'../docs/figures/amp_02/metrics_perplexity\.png': '../docs/figures/amp_02/20251015_metrics_perplexity.png',
        r'../docs/figures/amp_02/perf_tokens_per_sec\.png': '../docs/figures/amp_02/20251015_perf_tokens_per_sec.png',
    }
    
    # Apply mappings
    for old_pattern, new_pattern in path_mappings.items():
        content = re.sub(old_pattern, new_pattern, content)
    
    # Write updated content
    report_path.write_text(content)
    print("Updated report.md with dated image paths")

if __name__ == "__main__":
    add_dates_to_existing()
    update_report_paths_with_dates()
    print("All files updated with dates!")
