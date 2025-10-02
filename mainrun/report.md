MainRun Report

Baseline

- Best validation loss (7 epochs): 1.7533
- Source run: mainrun/runs/latest
- Figures (latest snapshot):

![Training Loss](../docs/figures/20251002_083200_loss_train.png)
![Validation Loss](../docs/figures/20251002_083200_loss_val.png)
![Learning Rate](../docs/figures/20251002_083200_lr.png)
![Tokens/sec](../docs/figures/20251002_083200_perf_tokens_per_sec.png)
![Perplexity](../docs/figures/20251002_083200_metrics_perplexity.png)

Notes

- Curves show steady training loss decrease with periodic validation spikes aligned with evals.
- Validation loss converges near 1.75 by epoch 7, meeting the baseline target.
- LR cosine schedule decays smoothly; throughput is stable.


## baseline_20251002 — Baseline run (7 epochs)
- Change: Initial logging + report pipeline
- Rationale: Establish reference metrics and artifacts
- Settings: `optimizer=SGD, lr=6e-3, batch=64`
- Best val loss: **1.7533**

Figures:
![Validation Loss](../docs/figures/baseline_20251002_loss_val.png)
![Training Loss](../docs/figures/baseline_20251002_loss_train.png)
![Learning Rate](../docs/figures/baseline_20251002_lr.png)
![Tokens/sec](../docs/figures/baseline_20251002_perf_tokens_per_sec.png)
![Perplexity](../docs/figures/baseline_20251002_metrics_perplexity.png)

