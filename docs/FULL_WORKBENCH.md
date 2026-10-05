# Full model-engineering workbench

The appliance now includes the architectural building blocks for every planned feature.

## Central workflow

The primary user workflow is **12 GB Release Search**:

1. select a tuned source checkpoint;
2. profile experts on coding/agent/long-context calibration data;
3. generate structured pruning candidates;
4. recovery-tune survivors;
5. generate standard GGUF, TurboQuant and CAT-Q candidates;
6. run cheap screening;
7. run memory/context qualification;
8. run coding and agent suites;
9. execute the OxCoder dominance gate;
10. maintain the Pareto frontier;
11. package and optionally publish the winner.

## Added subsystems

- multi-objective Pareto frontier engine;
- deterministic prune/quant candidate generation;
- sensitivity-driven mixed precision planner;
- composite expert importance scoring;
- recovery KL/distillation primitive;
- benchmark adapter registry;
- OxCoder gate;
- long-context evaluation matrix;
- agent failure trace storage and classification;
- dataset weighted-mixture builder;
- deduplication;
- provenance/license manifests;
- held-out contamination checker;
- checkpoint lineage graph;
- distributed/independent GPU scheduler;
- host resource policy;
- runtime/KV/Flash-Attention/CPU-MoE matrix generator;
- release manifest/model-card packaging;
- webhook notifications;
- encrypted-at-rest appliance secret store;
- role/permission model;
- API feature surface.

## Integration principle

Heavy external benchmark suites remain adapters, not vendored copies. The workbench pins the
command/config/version used for each suite and stores its outputs. This keeps benchmark
implementations independently upgradeable without making the appliance image impossible to maintain.

Likewise, the search engine delegates SFT, profiling, pruning, recovery, quantization and evaluation
to the existing registered jobs. It does not duplicate those implementations.
