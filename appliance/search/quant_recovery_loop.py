from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Callable
from appliance.recovery.escalation import RecoveryPolicy,decide

@dataclass
class QuantRecoveryLoop:
    policy: RecoveryPolicy
    launch_stage: Callable[[str,dict[str,Any]],str]
    def next_action(self,*,candidate_metrics,reference_metrics,recovery_round,config):
        d=decide(candidate_metrics,reference_metrics,self.policy,recovery_round)
        if d['action'] in {'post_quant_recovery','qat_recovery'}: d['run_id']=self.launch_stage(d['action'],config)
        return d
