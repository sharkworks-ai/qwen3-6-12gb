from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
    kl_weight: float = 0.5,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    kl = kl * (temperature**2)

    if labels is None:
        return kl

    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.shape[-1]),
        labels.view(-1),
        ignore_index=-100,
    )
    return (1.0 - kl_weight) * ce + kl_weight * kl
