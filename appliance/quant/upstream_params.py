"""Trainable adapters around pinned GSQ and CAT-Q components."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.overrides import TorchFunctionMode


class SoftenedRounding(TorchFunctionMode):
    """Adapt CAT-Q's published inference quantizer for STE training."""

    def __init__(self, rate: float):
        super().__init__()
        self.rate = rate

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func == torch.round:
            value = args[0]
            return value + self.rate * (torch.round(value) - value).detach()
        return func(*args, **(kwargs or {}))


class ProjectionQuantizer(torch.nn.Module):
    def __init__(self, weight, spec, components, hessian, cfg):
        super().__init__()
        self.shape = tuple(weight.shape)
        self.spec = spec
        self.temperature = 2.0
        self.logit_scale = 10.0
        self.rate = 1.0
        matrix = weight.detach().reshape(-1, weight.shape[-1])
        self.register_buffer("base", matrix.clone())
        prior, quant, ternary, _ = components
        group = spec["group_size"]
        if matrix.shape[-1] % group:
            raise ValueError("Upstream training requires projection width divisible by group_size")
        if spec["bits"] == 1.58:
            # CAT-Q applies its low-rank correction before ternary quantization.
            # Keep independent factors for each fused routed expert.
            rank = int(cfg.get("catq_rank", 4))
            if rank < 1:
                raise ValueError("catq_rank must be positive")
            self.correction_scale = float(cfg.get("catq_alpha", rank)) / rank
            self.lora_a = torch.nn.Parameter(
                weight.new_empty(*weight.shape[:-2], rank, weight.shape[-1])
            )
            self.lora_b = torch.nn.Parameter(
                weight.new_zeros(*weight.shape[:-2], weight.shape[-2], rank)
            )
            torch.nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
            self.quantizer = ternary(
                {
                    "group_size": group,
                    "shift_mu": False,
                    "drop_quant_mu": True,
                    "ter_scale_type": "absmean",
                    "init_scale_from_raw_weights": True,
                    "learnable_scale": True,
                    "learnable_mu": False,
                    "learnable_round": True,
                    "init_round_thd": 0.5,
                    "learnable_factor_act": "sigmoid",
                },
                shape=matrix.shape,
            ).to(matrix.device)
            for parameter in self.quantizer.parameters():
                parameter.requires_grad_(True)
        else:
            if hessian is None:
                raise ValueError("Missing activation Hessian for GSQ; RTN fallback is disabled")
            # Use the real GPTQ algorithm. Generic name avoids upstream's extra
            # q/k-only 2000-step warmup and its INT4-default mismatch.
            layer = torch.nn.Linear(
                matrix.shape[1],
                matrix.shape[0],
                bias=False,
                device=matrix.device,
                dtype=matrix.dtype,
            )
            layer.weight.data.copy_(matrix)
            settings = SimpleNamespace(quantization=SimpleNamespace(gsq_bits=spec["bits"]))
            initializer = prior.GPTQ(
                layer, "appliance_projection", settings, matrix.device, matrix.dtype
            )
            initializer.H = hessian.to(matrix.device).float().clone()
            initializer.quantizer = prior.Quantizer()
            initializer.quantizer.configure(spec["bits"], perchannel=True, sym=True, mse=True)
            q, scales = initializer.fasterquant(
                None,
                groupsize=group,
                blocksize=int(cfg.get("gptq_block_size", 128)),
                percdamp=float(cfg.get("gptq_damp", 0.1)),
            )
            common = (q, scales, group, 0.01, 6, matrix.device, matrix.dtype, matrix.dtype)
            self.quantizer = (
                quant.GumbelQuantizer2Bit(*common)
                if spec["bits"] == 2
                else quant.GumbelQuantizerInt(*common, bits=spec["bits"])
            )
            del layer, initializer

    def corrected_weight(self):
        return self.base + (self.lora_b @ self.lora_a).reshape_as(self.base) * self.correction_scale

    def forward(self):
        if self.spec["bits"] == 1.58:
            weight = self.corrected_weight()
            if self.training:
                with SoftenedRounding(self.rate):
                    result = self.quantizer(weight)
            else:
                result = self.quantizer(weight)
        else:
            result = (
                self.quantizer(self.temperature, self.logit_scale)
                if self.training
                else self.quantizer.get_hard_weights()[0]
            )
        return result.reshape(self.shape)

    def packed(self):
        from appliance.quant.lowbit import pack_codes

        self.eval()
        if self.spec["bits"] == 1.58:
            grouped = self.corrected_weight().reshape(-1, self.spec["group_size"])
            scales = grouped.abs().mean(-1, keepdim=True) + 1e-6
            scales = scales * self.quantizer.generate_scale_factor()
            hard = self.forward().reshape_as(grouped)
            codes = (hard / scales).round().long() + 1
            scales = scales.half()
        else:
            hard, scale = self.quantizer.get_hard_weights()
            per_column = scale[:, self.quantizer.idx]
            minimum = -(2 ** (int(self.spec["bits"]) - 1))
            codes = (hard / per_column).round().long() - minimum
            scales = scale.reshape(-1, 1).half()
        maximum = 2 if self.spec["bits"] == 1.58 else 2 ** int(self.spec["bits"]) - 1
        if codes.min() < 0 or codes.max() > maximum or not torch.isfinite(scales).all():
            raise ValueError("Quantizer cannot be represented by the declared scalar grid")
        return {"codes": pack_codes(codes.cpu().byte(), self.spec["bits"]), "scales": scales.cpu()}


def schedule(quantizers, step: int, steps: int):
    progress = step / max(1, steps - 1)
    for quantizer in quantizers.values():
        quantizer.temperature = 2.0 * (0.025**progress)
        quantizer.logit_scale = 10.0 * (50**progress)
        quantizer.rate = 0.5 + 0.5 * progress
