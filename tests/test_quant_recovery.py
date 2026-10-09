from pathlib import Path

import torch

from appliance.qat.fake_quant import FakeQuantSpec, fake_quant_weight
from appliance.recovery.escalation import RecoveryPolicy, decide
from appliance.recovery.failure_buffer import FailureBuffer


def test_fake_quant_shape_and_grad():
    x=torch.randn(17,requires_grad=True); y=fake_quant_weight(x,FakeQuantSpec(3,8)); assert y.shape==x.shape; y.sum().backward(); assert x.grad is not None

def test_ternary_fake_quant():
    x=torch.tensor([-2.,-.01,0.,.01,2.],requires_grad=True); y=fake_quant_weight(x,FakeQuantSpec(1.58,5,True)); assert torch.count_nonzero(y).item()<=2

def test_escalation():
    ref={'coding_score':100,'agent_score':100,'long_context_score':100}
    assert decide({'coding_score':99,'agent_score':99,'long_context_score':99},ref,RecoveryPolicy())['action']=='accept_without_recovery'
    assert decide({'coding_score':97,'agent_score':97,'long_context_score':97},ref,RecoveryPolicy())['action']=='post_quant_recovery'
    assert decide({'coding_score':94,'agent_score':94,'long_context_score':97},ref,RecoveryPolicy())['action']=='qat_recovery'

def test_failure_buffer(tmp_path:Path):
    b=FailureBuffer(tmp_path/'f.jsonl'); rows=[{'prompt':'x','teacher_success':True,'student_success':False,'teacher_output':'good','student_output':'bad'}]
    assert b.extend_from_comparison(rows)==1; assert b.extend_from_comparison(rows)==0; assert len(b.read())==1
