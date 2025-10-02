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
  ![val loss](../figures/adamw_warmup_01_loss_val.png)  
  ![train loss](../figures/adamw_warmup_01_loss_train.png)  
  ![lr](../figures/adamw_warmup_01_lr.png)  
  ![tokens/sec](../figures/adamw_warmup_01_perf_tokens_per_sec.png)  
  ![perplexity](../figures/adamw_warmup_01_metrics_perplexity.png)
