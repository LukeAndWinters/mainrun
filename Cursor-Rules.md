# MainRun Guardrails
**Do NOT change:** epochs (7), random seed, dataset/split, or `evaluate()` (definition or call sites).
**Allowed:** tokenizer, model internals, optimizer/schedule, training loop structure around (but not inside) `evaluate()`, logging, configs, AMP, compile, packing.

**House style:** small PRs; every change has PLAN → RISKS → DIFF PLAN → TESTS. If a request risks touching forbidden items, STOP and propose an alternative.