#!/usr/bin/env python3
"""
LR sweep script for GPT-2 training optimization.
WHAT: Run multiple training runs with different LR and accumulation settings
WHY: Find optimal hyperparameters for best validation loss
IMPACT: Systematic optimization of learning rate and batch size
"""
import subprocess
import json
import sys
from pathlib import Path

def run_experiment(lr, eta_min_factor, grad_accum_steps, base_lr=6e-3):
    """Run a single training experiment with given parameters."""
    lr_actual = base_lr * lr
    eta_min = lr_actual * eta_min_factor
    
    print(f"\n{'='*60}")
    print(f"Running: LR={lr_actual:.2e}, EtaMin={eta_min:.2e}, Accum={grad_accum_steps}")
    print(f"{'='*60}")
    
    cmd = [
        "python3", "mainrun/train.py",
        "--lr", str(lr_actual),
        "--eta_min_factor", str(eta_min_factor),
        "--grad_accum_steps", str(grad_accum_steps),
        "--sweep_mode"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Run LR sweep experiments."""
    print("🚀 Starting LR Sweep for GPT-2 Training Optimization")
    print("Target: Minimize validation loss, prevent late-epoch rebounds")
    
    # Clear previous sweep log
    sweep_log = Path("logs/lr_sweep.jsonl")
    if sweep_log.exists():
        sweep_log.unlink()
    
    # Experiment configurations
    experiments = [
        # (lr_multiplier, eta_min_factor, grad_accum_steps)
        (0.8, 0.2, 4),   # Lower LR, same accumulation
        (1.0, 0.2, 4),   # Baseline LR
        (1.1, 0.2, 4),   # Slightly higher LR
        (1.2, 0.2, 4),   # Higher LR
        (1.2, 0.2, 2),   # Higher LR, less accumulation
    ]
    
    results = []
    base_lr = 6e-3
    
    for i, (lr_mult, eta_min_factor, grad_accum) in enumerate(experiments, 1):
        print(f"\n🔬 Experiment {i}/{len(experiments)}")
        success = run_experiment(lr_mult, eta_min_factor, grad_accum, base_lr)
        
        if success and sweep_log.exists():
            # Read the latest result
            with open(sweep_log, "r") as f:
                lines = f.readlines()
                if lines:
                    latest_result = json.loads(lines[-1].strip())
                    results.append(latest_result)
    
    # Print summary table
    if results:
        print(f"\n{'='*80}")
        print("📊 LR SWEEP RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"{'LR':<12} {'EtaMin':<12} {'Accum':<6} {'Final Val':<12} {'Best Val':<12} {'Skipped':<8}")
        print(f"{'-'*80}")
        
        for r in results:
            print(f"{r['base_lr']:<12.2e} {r['eta_min']:<12.2e} {r['accum']:<6} "
                  f"{r['final_val']:<12.6f} {r['best_val']:<12.6f} {r['skipped_updates']:<8}")
        
        # Find best configuration
        best_result = min(results, key=lambda x: x['final_val'])
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   LR: {best_result['base_lr']:.2e}")
        print(f"   EtaMin: {best_result['eta_min']:.2e}")
        print(f"   Accum: {best_result['accum']}")
        print(f"   Final Val Loss: {best_result['final_val']:.6f}")
        print(f"   Best Val Loss: {best_result['best_val']:.6f}")
        
        # Check acceptance criteria
        print(f"\n✅ ACCEPTANCE CRITERIA CHECK:")
        print(f"   Skipped Updates = 0: {'✅' if all(r['skipped_updates'] == 0 for r in results) else '❌'}")
        print(f"   Final Val < 1.445: {'✅' if best_result['final_val'] < 1.445 else '❌'}")
        print(f"   Target ≤ 1.38: {'✅' if best_result['final_val'] <= 1.38 else '❌'}")
        
    else:
        print("❌ No successful experiments completed")

if __name__ == "__main__":
    main()