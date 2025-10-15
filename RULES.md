# MainRun Local Rules
Do NOT change: epochs=7, seed=<baseline>, dataset, val_fraction=<baseline>, or `evaluate()`.
Do NOT use: pretrained weights or data augmentation.
Allowed: model arch, tokenization, optimizer, scheduler, training loop (except `evaluate()`).

Local guardrails:
- Python guard verifies seed/epochs/val_fraction and `evaluate.py` SHA256.
- Git pre-commit hook blocks violating commits.