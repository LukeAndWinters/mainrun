# Experiment 

## [2025-01-16] Experiment 13: Rotary Position Embeddings (RoPE) Implementation

### Added
- Rotary Position Embeddings (RoPE) support in `CausalSelfAttention`
- Configurable position encoding types: learned, rope via `--pos_encoding` CLI argument
- RoPE frequency matrix pre-computation for head_dim
- Rotation logic applied to even/odd dimensions of head_dim

### Changed
- Modified `CausalSelfAttention` to apply RoPE to query and key vectors
- Updated `GPT` class to conditionally use RoPE instead of learned embeddings
- Added position encoding configuration to `GPTConfig`

### Results
- **Final Val Loss**: 1.260637 (vs 1.271026 with MQA) - **0.8% improvement**
- **Best Val Loss**: 1.215451 (vs 1.270052 with MQA) - **4.3% improvement**
- **Rebound**: 0.045185 (vs -0.000974 with MQA) - higher instability
- **Overall Assessment**: Small net improvement despite higher rebound
- **Implementation Success**: RoPE worked correctly without errors

### Technical Details
- RoPE applied directly to query and key vectors in attention mechanism
- Frequency matrix pre-computed for head_dim (64) with base=10000.0
- Rotation applied to even/odd dimensions of head_dim
- Integrated well with existing MQA + RMSNorm + Pre-LN architecture

### Architecture
- RoPE + MQA + RMSNorm + Pre-LN + Residual Scaling + SwiGLU
- Status: NEW BEST RESULT - Small improvement over MQA baseline (28.1% over baseline)

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

## rebound_fix_06 — LR Schedule Refinement for Rebound Elimination
- **Commit:** TBD
- **Change:** Added configurable warmup percentage, optional tail squeeze, and ran focused sweep to eliminate late-epoch rebound
- **Rationale:** Systematic late-epoch rebound (0.12-0.24) prevents maintaining peak performance; implement gentler LR decay patterns
- **Key settings:** warmup_pct=0.25, eta_min_factor=0.02-0.05, grad_accum_steps=2, optional tail_squeeze
- **Implementation details:**
  - Added CLI arguments: `--warmup_pct`, `--tail_squeeze`, `--tail_squeeze_pct`
  - Created `WarmupCosineWithTailSqueeze` class with linear decay to 0 in final 10% of steps
  - Updated LR scheduler initialization to choose between cosine-with-floor and tail-squeeze variants
  - Enhanced result.json to include final_val_loss for rebound analysis
  - Created automated sweep script `scripts/run_rebound_sweep.py`
- **Sweep Results:**
  - **Best Configuration**: LR=4.5e-3, eta_min_factor=0.02, warmup=0.25, grad_accum=2, no tail squeeze
  - **Best Val Loss**: 1.310431 (excellent improvement from 1.3356)
  - **Rebound**: 0.094559 (reduced from 0.12-0.24, but still above target ≤0.03)
  - **Tail Squeeze Issue**: Caused massive rebound (518.16), indicating too aggressive
- **Observations:**
  - ✅ **Significant Improvement**: Best validation loss improved to 1.310431 (25.3% better than baseline)
  - ✅ **Rebound Reduction**: Reduced from 0.12-0.24 to 0.094559 (21% improvement)
  - ⚠️ **Target Not Met**: Rebound still above 0.03 target, indicating need for further refinement
  - 🔍 **Tail Squeeze Problem**: Linear decay to 0 too aggressive, causes training instability
- **Result:** best val loss = **1.310431** (↓ 0.025 vs previous best, ↓ 0.444 vs baseline)
- **Next Steps:** Further LR schedule refinement needed to achieve rebound ≤ 0.03 target
- **Figures:** [to be added after snapshot]


## rebound_fix_06 — Rebound Fix - Best Configuration
- **Commit:** `865dcd1`
- **Change:** LR=4.5e-3, eta_min=0.02, warmup=0.25, grad_accum=2
- **Rationale:** Optimal configuration from systematic sweep
- **Key settings:** `lr=4.5e-3, eta_min_factor=0.02, warmup_pct=0.25, grad_accum_steps=2`
- **Result:** best val loss = **1.3104**
- **Figures:**  
  ![val loss](figures/rebound_fix_06_loss_val.png)  
  ![train loss](figures/rebound_fix_06_loss_train.png)


## tail_squeeze_fix — Tail Squeeze Bug Fix - Corrected Scheduler
- **Commit:** `865dcd1`
- **Change:** Fixed total_optim_steps calculation in WarmupCosineWithTailSqueeze scheduler
- **Rationale:** The scheduler was initialized with wrong total steps, causing negative LRs and training explosion
- **Key settings:** `LR=4.5e-3, eta_min=0.05, warmup=0.25, grad_accum=2, tail_squeeze=True`
- **Result:** best val loss = **1.3514**
- **Figures:**  
  ![val loss](figures/tail_squeeze_fix_loss_val.png)  
  ![train loss](figures/tail_squeeze_fix_loss_train.png)


## rebound_fix_06c — Enhanced Tail Squeeze + Beta2 Damping Sweep
- **Commit:** `TBD`
- **Change:** Systematic 8-config sweep adding late-training beta2 damping with tail squeeze; compared warmup %, tail duration, eta_min floor, and accumulation.
- **Rationale:** Eliminate late-epoch rebound while maintaining best-in-run convergence; test whether second-moment damping (beta2 → 0.98–0.99) plus a short tail squeeze (10%) improves final validation loss.
- **Key settings (sweeped):**
  - `lr=4.5e-3`
  - `eta_min_factor ∈ {0.00, 0.01, 0.02}`
  - `warmup_pct ∈ {0.25, 0.30}`
  - `tail_squeeze=True` with `tail_squeeze_pct ∈ {0.08, 0.10, 0.15}` (except control)
  - `grad_accum_steps ∈ {2, 4}`
  - `beta2_tail ∈ {None, 0.98, 0.99}` starting at `tail_beta2_start_pct=0.30`
- **Representative results (Final | Best | Rebound):**
  - tail_pct=0.08, beta2=0.98, warmup=0.25, accum=2 → 1.3791 | 1.3615 | 0.0176
  - tail_pct=0.10, beta2=0.99, warmup=0.25, accum=2 → 1.3553 | 1.3562 | -0.0009  ← Best final
  - tail_pct=0.10, beta2=0.98, warmup=0.30, accum=2 → 1.3622 | 1.3503 | 0.0119
  - tail_pct=0.10, beta2=0.98, warmup=0.25, accum=4 → 1.4509 | 1.2236 | 0.2273
  - CONTROL (no tail squeeze, eta_min=0.02, accum=2) → 1.4053 | 1.3620 | 0.0433
- **Observations:**
  1. Beta2 damping helps: 0.99 outperformed 0.98 on final loss at matched settings.
  2. Tail squeeze 10% > 8% for final loss; 15% did not improve over 10% in this grid.
  3. Longer warmup (30%) slightly worsened final loss vs 25% under tail squeeze.
  4. Accum=4 re-introduced a large rebound; Accum=2 remained stable (skipped_updates=0).
  5. CONTROL without tail squeeze shows higher rebound (0.0433) and worse final (1.4053).
- **Result:**
  - Best final validation loss: **1.3553** with near-zero rebound (**-0.0009**) at `lr=4.5e-3, eta_min_factor=0.01, warmup_pct=0.25, tail_pct=0.10, beta2_tail=0.99, accum=2`.
  - Target rebound ≤ 0.01: **PASS** (best config). Target final ≤ 1.33: **NOT YET**.
- **Next Steps:**
  - Narrow sweep around the best: `beta2_tail ∈ {0.985, 0.995}`, `eta_min_factor ∈ {0.005, 0.01}`; confirm robustness.
  - Keep `accum=2`; test `tail_squeeze_pct ∈ {0.10, 0.12}`; avoid 8% and 15% for now.
  - If plateau: consider Residual Scaling (LayerScale) as low-risk architectural follow-up.


## lr_sched_06dA — Micro-tune A: eta_min=0.005, tail=0.12, β2=0.995
- **Commit:** `3cd2ee1`
- **Change:** Increase tail to 12% and stronger β2 damping
- **Rationale:** Test if longer tail and stronger second-moment damping reduce final loss
- **Key settings:** `lr=4.5e-3, warmup=0.25, tail=0.12, beta2=0.995, accum=2`
- **Result:** best val loss = **1.3544**
- **Figures:**  
  ![val loss](figures/lr_sched_06dA_loss_val.png)  
  ![train loss](figures/lr_sched_06dA_loss_train.png)


## lr_sched_06dB — Micro-tune B: eta_min=0.01, tail=0.10, β2=0.985
- **Commit:** `3cd2ee1`
- **Change:** Slightly lower β2 target at best tail=10%
- **Rationale:** Verify if reducing β2 improves final while keeping rebound≈0
- **Key settings:** `lr=4.5e-3, warmup=0.25, tail=0.10, beta2=0.985, accum=2`
- **Result:** best val loss = **1.3668**
- **Figures:**  
  ![val loss](figures/lr_sched_06dB_loss_val.png)  
  ![train loss](figures/lr_sched_06dB_loss_train.png)


## Quick_Win_Experiments — Quick Win Optimization Experiments
- **Commit:** `1512f6f`
- **Change:** LR micro-tuning, architecture combo, and tail squeeze refinement
- **Rationale:** Targeted optimization to minimize validation loss with low-risk changes
- **Key settings:** `LR=4.2e-3, eta_min=0.03, warmup=20%, grad_accum=2, tail_squeeze=10%, beta2_tail=0.99`
- **Result:** best val loss = **1.3325**
- **Figures:**  
  ![val loss](figures/Quick_Win_Experiments_loss_val.png)  
  ![train loss](figures/Quick_Win_Experiments_loss_train.png)


## exp — Experiment 8 Phase 1: LR Precision Tuning
- **Commit:** `97698d2`
- **Change:** Systematic LR micro-tuning around 4.2e-3
- **Rationale:** Fine-tune learning rate to eliminate rebound and optimize final performance
- **Key settings:** `LR=4.0e-3 to 4.9e-3, eta_min_factor=0.03, warmup_pct=0.20, grad_accum_steps=2, tail_squeeze enabled`
- **Result:** best val loss = **1.332499** (LR=4.2e-3) - **NEW BEST RESULT**
- **Key Achievement:** Eliminated late-epoch rebound (negative rebound = -0.005920)
- **Figures:**  
  ![val loss](figures/Experiment_8_Phase_1_LR_Precision_Tuning_20251017_024236/20251017_loss_val.png)  
  ![train loss](figures/Experiment_8_Phase_1_LR_Precision_Tuning_20251017_024236/20251017_loss_train.png)


## exp — Experiment 8 Phase 2: Eta Min Factor Tuning
- **Commit:** `cc21501`
- **Change:** Systematic eta_min_factor tuning around 0.03
- **Rationale:** Fine-tune learning rate floor to optimize final performance
- **Key settings:** `eta_min_factor=0.025,0.035,0.04, LR=4.2e-3, warmup_pct=0.20, grad_accum_steps=2, tail_squeeze enabled`
- **Result:** best val loss = **1.3773**
- **Figures:**  
  ![val loss](figures/exp_loss_val.png)  
  ![train loss](figures/exp_loss_train.png)

## Experiment 8 Phase 3: Warmup Percentage Tuning
- **Commit:** `ffadef8`
- **Change:** Systematic warmup percentage tuning around 20%
- **Rationale:** Fine-tune warmup duration to optimize early training stability
- **Key settings:** `warmup_pct=0.15,0.18,0.22,0.25, LR=4.2e-3, eta_min_factor=0.03, grad_accum_steps=2, tail_squeeze enabled`
- **Result:** best val loss = **1.3539** (no improvement over Phase 1)
- **Key Finding:** Warmup percentage parameter saturation - minimal impact on performance
- **Status:** Micro-tuning approach reached saturation point
- **Figures:**  
  ![val loss](../docs/figures/Experiment_8_Phase_3_Warmup_Percentage_Tuning_20251016_120000_20251017_033209/20251017_loss_val.png)  
  ![train loss](../docs/figures/Experiment_8_Phase_3_Warmup_Percentage_Tuning_20251016_120000_20251017_033209/20251017_loss_train.png)

## Experiment 9: Architecture Enhancement - Residual Scaling + SwiGLU
- **Commit:** `TBD`
- **Change:** Architecture enhancement with residual scaling + SwiGLU activation
- **Rationale:** Move beyond hyperparameter optimization to architectural improvements
- **Key settings:** `--residual_scale --mlp_activation swiglu, LR=4.2e-3, eta_min_factor=0.03, warmup_pct=0.20, grad_accum_steps=2, tail_squeeze enabled`
- **Result:** best val loss = **1.287965** (BREAKTHROUGH - 3.36% improvement)
- **Key Finding:** Architecture enhancement highly effective - 26.6% total improvement over baseline
- **Status:** New best result achieved - optimal configuration found
- **Figures:**  
  ![val loss](../docs/figures/Experiment_9_Architecture_Enhancement_Residual_Scaling_SwiGLU_20251017_033500_20251017_035046/20251017_loss_val.png)  
  ![train loss](../docs/figures/Experiment_9_Architecture_Enhancement_Residual_Scaling_SwiGLU_20251017_033500_20251017_035046/20251017_loss_train.png)


## exp — Pre-LN Architecture Implementation
- **Commit:** `da060ff`
- **Change:** Implemented Pre-LN transformer blocks with normalization before attention/MLP
- **Rationale:** Pre-LN often provides 3-7% improvement over Post-LN by improving gradient flow and training stability
- **Key settings:** `lr=4.2e-3, eta_min_factor=0.03, warmup_pct=0.20, grad_accum_steps=2, tail_squeeze=True, residual_scale=True, mlp_activation=swiglu, pre_ln=True`
- **Result:** best val loss = **1.2877**
- **Figures:**  
  ![val loss](figures/exp_loss_val.png)  
  ![train loss](figures/exp_loss_train.png)

### 2025-01-17 04:23:08 - Enhanced Experiment 10 Documentation
- **Change:** Added comprehensive visual analysis to Experiment 10 report
- **Rationale:** Provide detailed training curve analysis with visual evidence to support findings
- **Key additions:** 5 training curve visualizations, performance comparison table, quantitative analysis
- **Impact:** Improved report quality and assessment documentation with visual evidence
- **Figures:** Enhanced report with validation loss, training loss, learning rate, performance, and perplexity plots


## exp — iteration
- **Commit:** `67f5de5`
- **Change:** n/a
- **Rationale:** n/a
- **Key settings:** ``
- **Result:** best val loss = **0.0000**
- **Figures:**  
  ![val loss](figures/exp_loss_val.png)  
  ![train loss](figures/exp_loss_train.png)


## Experiment_11_RMSNorm_Implementation — RMSNorm Architecture Implementation
- **Commit:** `67f5de5`
- **Change:** Replaced LayerNorm with RMSNorm throughout the model
- **Rationale:** RMSNorm provides better training stability and is used in modern transformers like LLaMA and PaLM
- **Key settings:** `norm_type=rmsnorm, lr=4.2e-3, eta_min_factor=0.03, warmup_pct=0.20, grad_accum_steps=2, tail_squeeze, residual_scale, mlp_activation=swiglu, pre_ln`
- **Result:** best val loss = **1.3396**
- **Figures:**  
  ![val loss](figures/Experiment_11_RMSNorm_Implementation_loss_val.png)  
  ![train loss](figures/Experiment_11_RMSNorm_Implementation_loss_train.png)


## Experiment_12_MQA_Implementation — Multi-Query Attention Implementation
- **Commit:** `TBD`
- **Change:** Implemented Multi-Query Attention (MQA) with single key/value head shared across query heads
- **Rationale:** MQA reduces memory usage while maintaining or improving performance, used in LLaMA and PaLM
- **Key settings:** `attention_type=mqa, norm_type=rmsnorm, lr=4.2e-3, eta_min_factor=0.03, warmup_pct=0.20, grad_accum_steps=2, tail_squeeze, residual_scale, mlp_activation=swiglu, pre_ln`
- **Result:** best val loss = **1.270052** (NEW BEST - 5.1% improvement over MHA)
- **Key Achievement:** 27.5% total improvement over baseline with memory efficiency
- **Figures:**  
  ![val loss](figures/Experiment_12_MQA_Implementation_20251017_045250/20251017_loss_val.png)  
  ![train loss](figures/Experiment_12_MQA_Implementation_20251017_045250/20251017_loss_train.png)

