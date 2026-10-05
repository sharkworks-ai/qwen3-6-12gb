from __future__ import annotations
import argparse,json,random
from pathlib import Path
from appliance.recovery.failure_buffer import FailureBuffer

DEFAULT={'agentic_coding':.35,'repository_coding':.25,'reasoning':.15,'long_context':.10,'general':.10,'quant_failure':.05}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text())
    groups={}
    for name,path in cfg.get('sources',{}).items(): groups[name]=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    groups['quant_failure']=[{'messages':[{'role':'user','content':x.prompt},{'role':'assistant','content':x.teacher_output}],'metadata':x.metadata} for x in FailureBuffer(Path(cfg['failure_buffer'])).read()]
    names=[n for n,v in groups.items() if v]; weights=cfg.get('weights',DEFAULT); rng=random.Random(cfg.get('seed',42)); out=[]
    for _ in range(int(cfg.get('count',5000))):
        n=rng.choices(names,weights=[weights.get(x,0) for x in names],k=1)[0]; row=dict(rng.choice(groups[n])); row['_recovery_bucket']=n; out.append(row)
    dest=Path(cfg['output']); dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',encoding='utf-8') as h:
        for row in out: h.write(json.dumps(row,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
