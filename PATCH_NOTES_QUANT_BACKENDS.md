# Integration notes

The bundle adds pluggable quantization backends under `appliance/quant/`.

To register the generic quant job in the appliance's existing job registry, add a job
that runs:

```text
python -m appliance.quant.job --config <path-to-json>
```

The SFT/prune/quant bundle already has a quantization UI/config mechanism; point that
existing `quantize` stage at `appliance.quant.registry.get_backend(config["backend"])`
instead of directly invoking llama.cpp.

Three example configs are included:

- `configs/appliance/quant-llama.json`
- `configs/appliance/quant-turboquant.json`
- `configs/appliance/quant-bittern.json`

The Docker build should execute `scripts/build_quant_backends.sh` after installing build
dependencies. For RTX 5090 set `CUDA_ARCH=120`.
