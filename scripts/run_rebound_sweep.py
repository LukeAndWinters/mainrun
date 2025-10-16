#!/usr/bin/env python3
"""
Rebound Fix Sweep Script

WHAT: Automated testing of different LR schedule configurations to eliminate late-epoch rebound
WHY: Systematic evaluation of warmup percentage, eta_min_factor, and tail squeeze options
IMPACT: Find optimal configuration with rebound ≤ 0.03 and final_val ≈ best_val
"""

import subprocess
import json
import pathlib
import time

def run_experiment(lr, eta_min_factor, warmup_pct, grad_accum_steps, tail_squeeze=False):
    """Run a single training experiment and return results."""
    cmd = [
        "python3", "mainrun/train.py",
        "--lr", str(lr),
        "--eta_min_factor", str(eta_min_factor),
        "--warmup_pct", str(warmup_pct),
        "--grad_accum_steps", str(grad_accum_steps)
    ]
    if tail_squeeze:
        cmd.append("--tail_squeeze")
    
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"Error running experiment: {process.stderr}")
        return None
    
    # Parse result from latest run
    result_path = pathlib.Path("mainrun/runs/latest/result.json")
    if result_path.exists():
        with open(result_path) as f:
            result = json.load(f)
        
        best_val = result.get("best_val_loss", 0)
        final_val = result.get("final_val_loss", 0)
        rebound = final_val - best_val
        
        return {
            "lr": lr,
            "eta_min_factor": eta_min_factor,
            "warmup_pct": warmup_pct,
            "accum": grad_accum_steps,
            "tail_squeeze": tail_squeeze,
            "best_val": best_val,
            "final_val": final_val,
            "rebound": rebound
        }
    return None

def main():
    """Run the rebound fix sweep with 4 different configurations."""
    print("🔍 Starting Rebound Fix Sweep")
    print("=" * 50)
    
    experiments = [
        (5e-3, 0.05, 0.25, 2, False),    # Lower LR, lower eta_min, higher warmup
        (4.5e-3, 0.05, 0.25, 2, False),  # Even lower LR
        (4.5e-3, 0.02, 0.25, 2, False),  # Very low eta_min
        (4.5e-3, 0.05, 0.25, 2, True),   # With tail squeeze
    ]
    
    results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n--- Experiment {i}/4 ---")
        result = run_experiment(*exp)
        if result:
            results.append(result)
            print(f"✅ Completed: Best={result['best_val']:.6f}, Final={result['final_val']:.6f}, Rebound={result['rebound']:.6f}")
        else:
            print(f"❌ Failed: {exp}")
        
        # Small delay between experiments
        if i < len(experiments):
            time.sleep(2)
    
    print("\n" + "=" * 50)
    print("📊 REBOUND FIX SWEEP RESULTS")
    print("=" * 50)
    print(f"{'LR':<10} {'EtaMin':<10} {'Warmup':<10} {'Accum':<7} {'Tail':<6} {'Final':<10} {'Best':<10} {'Rebound':<10}")
    print("-" * 80)
    
    for r in results:
        tail_str = "Yes" if r['tail_squeeze'] else "No"
        print(f"{r['lr']:.2e}   {r['eta_min_factor']:.2f}       {r['warmup_pct']:.2f}       {r['accum']:<7} {tail_str:<6} {r['final_val']:.6f}  {r['best_val']:.6f}  {r['rebound']:.6f}")
    
    # Find best configuration
    if results:
        best_config = min(results, key=lambda x: x['rebound'])
        print(f"\n🎯 BEST CONFIGURATION (Lowest Rebound):")
        print(f"   LR: {best_config['lr']:.2e}")
        print(f"   Eta Min Factor: {best_config['eta_min_factor']:.2f}")
        print(f"   Warmup %: {best_config['warmup_pct']:.2f}")
        print(f"   Grad Accum: {best_config['accum']}")
        print(f"   Tail Squeeze: {'Yes' if best_config['tail_squeeze'] else 'No'}")
        print(f"   Rebound: {best_config['rebound']:.6f}")
        print(f"   Final Val: {best_config['final_val']:.6f}")
        print(f"   Best Val: {best_config['best_val']:.6f}")
        
        # Check if we achieved the target
        if best_config['rebound'] <= 0.03:
            print(f"✅ SUCCESS: Rebound ≤ 0.03 achieved!")
        else:
            print(f"⚠️  PARTIAL: Rebound = {best_config['rebound']:.6f} (target: ≤ 0.03)")
    
    # Save results
    logs_dir = pathlib.Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    with open("logs/rebound_sweep.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\n💾 Results saved to: logs/rebound_sweep.jsonl")
    print("✅ Rebound fix sweep completed!")

if __name__ == "__main__":
    main()
