#!/usr/bin/env python3
"""
Reorganize existing figures into subdirectories by experiment name.
"""
import pathlib
import shutil
import re

def reorganize_figures():
    figures_dir = pathlib.Path("docs/figures")
    if not figures_dir.exists():
        print("No figures directory found")
        return
    
    # Define experiment patterns and their target directories
    experiments = {
        "20251002_083200": "baseline_20251002_083200",
        "adamw_warmup_01": "adamw_warmup_01", 
        "amp_02": "amp_02",
        "baseline_20251002": "baseline_20251002",
        "baseline_v1": "baseline_v1",
        "logs": "logs_baseline"
    }
    
    for file_path in figures_dir.iterdir():
        if file_path.is_file() and file_path.suffix in ['.png', '.csv']:
            # Extract experiment name from filename
            filename = file_path.name
            experiment_name = None
            
            for pattern, target_dir in experiments.items():
                if filename.startswith(pattern):
                    experiment_name = target_dir
                    break
            
            if experiment_name:
                # Create subdirectory
                exp_dir = figures_dir / experiment_name
                exp_dir.mkdir(exist_ok=True)
                
                # Move file to subdirectory
                new_name = filename
                if experiment_name != "logs_baseline":  # Keep original names for experiments
                    # Remove experiment prefix from filename
                    new_name = re.sub(rf"^{pattern}_", "", filename)
                
                new_path = exp_dir / new_name
                print(f"Moving {filename} -> {experiment_name}/{new_name}")
                shutil.move(str(file_path), str(new_path))
            else:
                print(f"Skipping {filename} - no matching experiment pattern")

if __name__ == "__main__":
    reorganize_figures()
    print("Reorganization complete!")
