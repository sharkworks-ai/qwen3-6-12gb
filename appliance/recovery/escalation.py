from __future__ import annotations

import math
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

def _ret(v,r):
    try: v,r=float(v),float(r)
    except (TypeError,ValueError): return None
    return v/r if math.isfinite(v) and math.isfinite(r) and r>0 and v>=0 else None

def decide(candidate: dict[str,Any], reference: dict[str,Any], policy: RecoveryPolicy, recovery_round: int=0) -> dict[str,Any]:
    c=_ret(candidate.get('coding_score'),reference.get('coding_score')); a=_ret(candidate.get('agent_score'),reference.get('agent_score')); l=_ret(candidate.get('long_context_score'),reference.get('long_context_score'))
    # Missing or invalid evidence must block the decision, never read as "no regression".
    missing=[k for k,v in (('coding',c),('agent',a),('long_context',l)) if v is None]
    if missing: action='insufficient_evidence'
    elif recovery_round>=policy.max_recovery_rounds: action='reject'
    elif (c is not None and c<policy.qat_coding_retention_below) or (a is not None and a<policy.qat_agent_retention_below): action='qat_recovery'
    elif (c is not None and c<policy.trigger_coding_retention_below) or (a is not None and a<policy.trigger_agent_retention_below) or (l is not None and l<policy.trigger_long_context_retention_below): action='post_quant_recovery'
    else: action='accept_without_recovery'
    return {'action':action,'retention':{'coding':c,'agent':a,'long_context':l},'missing':missing,'recovery_round':recovery_round}
