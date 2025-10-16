# Experiment 

## adamw_warmup_01 — AdamW + warmup-cosine LR floor
- **Commit:** `7f9c371`
- **Change:** Switch optimizer from SGD to AdamW with decoupled weight decay and no-decay params (bias/LayerNorm/embeddings). Replace per-step scheduler with linear warmup (≈10% or ≤1000 steps) followed by cosine decay to an LR floor at 10% of base LR.
- **Rationale:** AdamW generally yields better optimization for Transformers than SGD. Warmup prevents early-step instability when weights/activations are unscaled; an LR floor sustains learning late in training within a fixed 7-epoch budget.
- **Key settings:** `lr=6e-3, betas=(0.9,0.95), wd=prev, warmup≈10%/≤1000, lr_floor=0.1*lr, grad_clip=1.0`.
- **Observations:**
  - Training stabilized quickly; LR tracked warmup then cosine as expected.
  - Throughput unchanged; no overflow/instability observed.
- **Result:** best val loss = **1.4452** (↓ 0.309 vs baseline 1.754).
- **Figures:**  
  ![val loss](figures/20251002_adamw_warmup_01/20251002_loss_val.png)  
  ![train loss](figures/20251002_adamw_warmup_01/20251002_loss_train.png)  
  ![lr](figures/20251002_adamw_warmup_01/20251002_lr.png)  
  ![tokens/sec](figures/20251002_adamw_warmup_01/20251002_perf_tokens_per_sec.png)  
  ![perplexity](figures/20251002_adamw_warmup_01/20251002_metrics_perplexity.png)

## amp_02 — AMP (Automatic Mixed Precision)
- **Commit:** `c1272df`
- **Change:** Add autocast and GradScaler for mixed precision training
- **Rationale:** Enable 1.5-2x speedup with minimal memory overhead
- **Key settings:** `autocast(), GradScaler(), betas=(0.9,0.95)`
- **Result:** best val loss = **1.4470**
- **Figures:**  
  ![val loss](figures/20251015_amp_02/20251015_loss_val.png)  
  ![train loss](figures/20251015_amp_02/20251015_loss_train.png)


## grad_accum_03 — Gradient Accumulation (4 steps)
- **Commit:** `8609062`
- **Change:** Added gradient accumulation over 4 micro-batches to simulate larger effective batch size
- **Rationale:** Better gradient estimates through larger effective batch size while maintaining memory efficiency
- **Key settings:** `grad_accum_steps=4, micro_batch_size=16, effective_batch_size=64, lr=6e-3`
- **Implementation details:**
  - Process 4 micro-batches of size 16 before each gradient update
  - Scale loss by 1/4 to maintain correct gradient magnitude
  - Use GradScaler with accumulation for mixed precision stability
  - Adjust evaluation frequency to match reduced gradient update frequency
- **Observations:**
  - ✅ **Success**: Achieved better validation loss (1.4402 vs 1.4452)
  - ⚠️ **Issue**: NaN values appeared starting around epoch 6
  - 🔍 **Analysis**: Gradient scaling issues with accumulation + AMP
  - 📊 **Impact**: 0.35% improvement despite instability
- **Result:** best val loss = **1.4402** (↓ 0.005 vs previous best, ↓ 0.313 vs baseline)
- **Stability Issues:**
  - NaN validation loss from epoch 6 onwards
  - Likely caused by compound gradient scaling (accumulation + AMP)
  - Need to fix before proceeding to next experiments
- **Figures:**  
  ![val loss](figures/grad_accum_03/20251015_loss_val.png)  
  ![train loss](figures/grad_accum_03/20251015_loss_train.png)  
  ![lr](figures/grad_accum_03/20251015_lr.png)  
  ![tokens/sec](figures/grad_accum_03/20251015_perf_tokens_per_sec.png)  
  ![perplexity](figures/grad_accum_03/20251015_metrics_perplexity.png)


## grad_accum_fix_04 — Gradient Accumulation: stability fix (bf16/fp16-safe)
- **Commit:** `8609062`
- **Change:** Use bf16 when available; safe fp16 scaler + non-finite guard; zero_grad at window start
- **Rationale:** Eliminate NaNs from accumulation+AMP; keep effective batch benefits
- **Key settings:** `grad_accum_steps=4; micro_batch_size=16; amp_dtype=bf16|fp16; grad_clip=1.0`
- **Result:** best val loss = **1.4483**
- **Figures:**  
  ![val loss](figures/grad_accum_fix_04/20251015_loss_val.png)  
  ![train loss](figures/grad_accum_fix_04/20251015_loss_train.png)


## grad_accum_stability_fix_05 — Gradient Accumulation: Stability Fix (bf16/fp16-safe)
- **Commit:** `e74bb95`
- **Change:** Implemented stability fixes for gradient accumulation: prioritized bf16, added non-finite gradient guard, ensured zero_grad at accumulation window start, increased warmup to 20%, tightened gradient clipping to 0.5, and added EMA for evaluation only.
- **Rationale:** Eliminate NaNs from accumulation+AMP; keep effective batch benefits; improve early training stability; smooth validation curves.
- **Key settings:** `grad_accum_steps=4; micro_batch_size=16; amp_dtype=bf16|fp16; grad_clip=0.5; warmup=20%; ema_decay=0.999`
- **Result:** best val loss = **1.3835**
- **Figures:**  
  ![val loss](figures/grad_accum_stability_fix_05_loss_val.png)  
  ![train loss](figures/grad_accum_stability_fix_05_loss_train.png)


## grad_accum_stability_fix_05 — Gradient Accumulation: Final Stability Fix (bf16/fp16-safe)
- **Commit:** `6b3520e`
- **Change:** Implemented stability fixes for gradient accumulation: prioritized bf16, added non-finite gradient guard, ensured zero_grad at accumulation window start, increased warmup to 20%, tightened gradient clipping to 0.5, and added EMA for evaluation only.
- **Rationale:** Eliminate NaNs from accumulation+AMP; keep effective batch benefits; improve early training stability; smooth validation curves.
- **Key settings:** `grad_accum_steps=4; micro_batch_size=16; amp_dtype=bf16|fp16; grad_clip=0.5; warmup=20%; ema_decay=0.999`
- **Result:** best val loss = **1.3835**
- **Figures:**  
  ![val loss](figures/grad_accum_stability_fix_05_loss_val.png)  
  ![train loss](figures/grad_accum_stability_fix_05_loss_train.png)


## lr_schedule_06 — LR Schedule Optimization: WarmupCosineWithFloor
- **Commit:** `5c30274`
- **Change:** Implemented WarmupCosineWithFloor class with single warmup → cosine decay to non-zero floor, corrected gradient accumulation stepping order, and added comprehensive logging with rebound warnings.
- **Rationale:** Fix late-epoch rebounds and optimize LR behavior for better convergence within 7-epoch budget. Single warmup phase prevents early instability, cosine decay provides smooth convergence, and non-zero floor sustains learning in final epochs.
- **Key settings:** `lr=5.4e-3, eta_min_factor=0.2, grad_accum_steps=4, warmup=20%, total_steps=945`
- **Implementation details:**
  - Created `WarmupCosineWithFloor` class with proper step counting
  - Fixed LR scheduler to step only after successful optimizer updates
  - Added CLI arguments for easy parameter tuning (`--lr`, `--eta_min_factor`, `--grad_accum_steps`)
  - Enhanced logging with final statistics, rebound warnings, and training summary
  - Added LR sweep infrastructure (`scripts/run_lr_sweep.py`)
- **Observations:**
  - ✅ **LR Schedule**: Smooth warmup → cosine decay pattern (no oscillations)
  - ✅ **Stability**: 0 skipped updates, perfect gradient accumulation
  - ⚠️ **Late Rebound**: Final val loss (1.6069) > best val loss (1.3626) by 0.244
  - 📊 **Improvement**: Best val loss improved to 1.3626 (↓ 0.021 vs previous best)
- **Result:** best val loss = **1.3626** (↓ 0.021 vs previous best, ↓ 0.391 vs baseline)
- **Next Steps:** Run LR sweep to find optimal parameters and eliminate late-epoch rebound
- **Figures:**  
  ![val loss](figures/Experiment_4_LR_Schedule_Optimization_20251016_014831/20251016_loss_val.png)  
  ![train loss](figures/Experiment_4_LR_Schedule_Optimization_20251016_014831/20251016_loss_train.png)  
  ![lr](figures/Experiment_4_LR_Schedule_Optimization_20251016_014831/20251016_lr.png)  
  ![tokens/sec](figures/Experiment_4_LR_Schedule_Optimization_20251016_014831/20251016_perf_tokens_per_sec.png)  
  ![perplexity](figures/Experiment_4_LR_Schedule_Optimization_20251016_014831/20251016_metrics_perplexity.png)

