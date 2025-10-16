#!/usr/bin/env python3
"""
Enhanced Tail Squeeze Parameter Sweep Script

WHAT: Systematic exploration of beta2 damping + tail squeeze parameters
WHY: Find optimal configuration for final validation loss ≤ 1.33 with rebound ≤ 0.01
IMPACT: Automated parameter optimization with comprehensive logging

Usage: python3 scripts/tail_squeeze_sweep.py
"""

import subprocess
import json
import pathlib
import time
from datetime import datetime

def run_experiment(config, experiment_num, total_experiments):
    """Run a single training experiment with given configuration."""
    print(f"\n=== Experiment {experiment_num}/{total_experiments} ===")
    print(f"Config: {config}")
    
    # Build command
    cmd = [
        "python3", "mainrun/train.py",
        "--lr", str(config["lr"]),
        "--eta_min_factor", str(config["eta_min_factor"]),
        "--warmup_pct", str(config["warmup_pct"]),
        "--grad_accum_steps", str(config["accum"]),
        "--tail_squeeze_pct", str(config["tail_pct"]),
    ]
    
    # Add optional flags
    if config.get("tail_squeeze", True):
        cmd.append("--tail_squeeze")
    
    if config["beta2_tail"] is not None:
        cmd.extend(["--beta2_tail", str(config["beta2_tail"])])
        cmd.extend(["--tail_beta2_start_pct", str(config["beta2_start"])])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run training
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr}")
        return None
    
    # Parse results from latest run
    result_path = pathlib.Path("mainrun/runs/latest/result.json")
    if not result_path.exists():
        print(f"❌ No result.json found at {result_path}")
        return None
    
    with open(result_path) as f:
        run_result = json.load(f)
    
    final_val = run_result["final_val_loss"]
    best_val = run_result["best_val_loss"]
    rebound = final_val - best_val
    
    print(f"✅ Completed in {elapsed:.1f}s: Final={final_val:.6f}, Best={best_val:.6f}, Rebound={rebound:.6f}")
    
    return {
        **config,
        "final_val": final_val,
        "best_val": best_val,
        "rebound": rebound,
        "elapsed_time": elapsed,
        "timestamp": datetime.now().isoformat()
    }

def main():
    """Run the enhanced tail squeeze parameter sweep."""
    
    # Define 8-config parameter grid as specified in plan
    configs = [
        # Test eta_min variations with beta2 damping
        {"lr": 4.5e-3, "eta_min_factor": 0.01, "warmup_pct": 0.25, "tail_pct": 0.10, 
         "accum": 2, "beta2_tail": 0.98, "beta2_start": 0.30, "tail_squeeze": True},
        
        {"lr": 4.5e-3, "eta_min_factor": 0.00, "warmup_pct": 0.25, "tail_pct": 0.10, 
         "accum": 2, "beta2_tail": 0.98, "beta2_start": 0.30, "tail_squeeze": True},
        
        # Test tail_pct variations
        {"lr": 4.5e-3, "eta_min_factor": 0.01, "warmup_pct": 0.25, "tail_pct": 0.08, 
         "accum": 2, "beta2_tail": 0.98, "beta2_start": 0.30, "tail_squeeze": True},
        
        {"lr": 4.5e-3, "eta_min_factor": 0.01, "warmup_pct": 0.25, "tail_pct": 0.15, 
         "accum": 2, "beta2_tail": 0.98, "beta2_start": 0.30, "tail_squeeze": True},
        
        # Test different beta2_tail values
        {"lr": 4.5e-3, "eta_min_factor": 0.01, "warmup_pct": 0.25, "tail_pct": 0.10, 
         "accum": 2, "beta2_tail": 0.99, "beta2_start": 0.30, "tail_squeeze": True},
        
        # Test warmup variation
        {"lr": 4.5e-3, "eta_min_factor": 0.01, "warmup_pct": 0.30, "tail_pct": 0.10, 
         "accum": 2, "beta2_tail": 0.98, "beta2_start": 0.30, "tail_squeeze": True},
        
        # Test with accum=4 for comparison
        {"lr": 4.5e-3, "eta_min_factor": 0.01, "warmup_pct": 0.25, "tail_pct": 0.10, 
         "accum": 4, "beta2_tail": 0.98, "beta2_start": 0.30, "tail_squeeze": True},
        
        # Baseline control (no beta2 damping, no tail squeeze)
        {"lr": 4.5e-3, "eta_min_factor": 0.02, "warmup_pct": 0.25, "tail_pct": 0.10, 
         "accum": 2, "beta2_tail": None, "beta2_start": 0.30, "tail_squeeze": False},
    ]
    
    print("🔍 Starting Enhanced Tail Squeeze Parameter Sweep")
    print("=" * 60)
    print(f"Total experiments: {len(configs)}")
    print(f"Estimated time: {len(configs) * 4:.0f} minutes")
    print("=" * 60)
    
    # Run all experiments
    results = []
    for i, config in enumerate(configs, 1):
        result = run_experiment(config, i, len(configs))
        if result:
            results.append(result)
        
        # Small delay between runs
        if i < len(configs):
            time.sleep(2)
    
    if not results:
        print("❌ No experiments completed successfully!")
        return
    
    # Print comprehensive results table
    print("\n" + "=" * 80)
    print("📊 ENHANCED TAIL SQUEEZE SWEEP RESULTS")
    print("=" * 80)
    print(f"{'Exp':<3} {'LR':<8} {'Eta':<6} {'Warm':<6} {'Tail%':<7} {'Acc':<4} {'β2':<5} {'Final':<8} {'Best':<8} {'Reb':<8} {'Time':<6}")
    print(f"{'-'*3:<3} {'-'*8:<8} {'-'*6:<6} {'-'*6:<6} {'-'*7:<7} {'-'*4:<4} {'-'*5:<5} {'-'*8:<8} {'-'*8:<8} {'-'*8:<8} {'-'*6:<6}")
    
    for i, r in enumerate(results, 1):
        beta2 = f"{r['beta2_tail']:.2f}" if r['beta2_tail'] else "None"
        print(f"{i:<3} {r['lr']:.1e}  {r['eta_min_factor']:.2f}   {r['warmup_pct']:.2f}   "
              f"{r['tail_pct']:.2f}    {r['accum']:<4} {beta2:<5} "
              f"{r['final_val']:.4f}   {r['best_val']:.4f}   {r['rebound']:.4f}   {r['elapsed_time']:.1f}s")
    
    # Find best configurations
    best_final = min(results, key=lambda x: x['final_val'])
    best_rebound = min(results, key=lambda x: x['rebound'])
    
    print("\n🏆 BEST CONFIGURATIONS:")
    print(f"   Best Final Val:  {best_final['final_val']:.6f} (Exp {results.index(best_final)+1})")
    print(f"   Best Rebound:    {best_rebound['rebound']:.6f} (Exp {results.index(best_rebound)+1})")
    
    # Check success criteria
    print("\n📋 SUCCESS CRITERIA CHECK:")
    success_count = 0
    for r in results:
        if r['final_val'] <= 1.33 and r['rebound'] <= 0.01:
            success_count += 1
    
    print(f"   Configs meeting target (final ≤ 1.33, rebound ≤ 0.01): {success_count}/{len(results)}")
    
    if best_final['final_val'] <= 1.33:
        print("   ✅ Target final validation loss achieved!")
    else:
        print(f"   ⚠️  Target final validation loss not met (best: {best_final['final_val']:.6f})")
    
    if best_rebound['rebound'] <= 0.01:
        print("   ✅ Target rebound achieved!")
    else:
        print(f"   ⚠️  Target rebound not met (best: {best_rebound['rebound']:.6f})")
    
    # Save results to file
    sweep_log_path = pathlib.Path("logs/tail_squeeze_sweep.jsonl")
    sweep_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(sweep_log_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\n💾 Results saved to: {sweep_log_path}")
    print("✅ Enhanced tail squeeze sweep completed!")
    
    # Return best config for potential Phase 2
    return best_final

if __name__ == "__main__":
    main()
