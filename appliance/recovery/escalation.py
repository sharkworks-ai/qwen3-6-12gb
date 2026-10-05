from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class RecoveryPolicy:
    trigger_coding_retention_below: float=0.98
    trigger_agent_retention_below: float=0.98
    trigger_long_context_retention_below: float=0.98
    qat_coding_retention_below: float=0.95
    qat_agent_retention_below: float=0.95
    max_recovery_rounds: int=2

def _ret(v,r): return None if v is None or r in (None,0) else v/r

def decide(candidate: dict[str,Any], reference: dict[str,Any], policy: RecoveryPolicy, recovery_round: int=0) -> dict[str,Any]:
    c=_ret(candidate.get('coding_score'),reference.get('coding_score')); a=_ret(candidate.get('agent_score'),reference.get('agent_score')); l=_ret(candidate.get('long_context_score'),reference.get('long_context_score'))
    if recovery_round>=policy.max_recovery_rounds: action='reject'
    elif (c is not None and c<policy.qat_coding_retention_below) or (a is not None and a<policy.qat_agent_retention_below): action='qat_recovery'
    elif (c is not None and c<policy.trigger_coding_retention_below) or (a is not None and a<policy.trigger_agent_retention_below) or (l is not None and l<policy.trigger_long_context_retention_below): action='post_quant_recovery'
    else: action='accept_without_recovery'
    return {'action':action,'retention':{'coding':c,'agent':a,'long_context':l},'recovery_round':recovery_round}
