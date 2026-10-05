from __future__ import annotations
import argparse,json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,TrainingArguments,Trainer,DataCollatorForLanguageModeling
from appliance.qat.parametrize import apply_fake_quant

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text())
    model=AutoModelForCausalLM.from_pretrained(cfg['student_model'],torch_dtype='bfloat16',trust_remote_code=True,low_cpu_mem_usage=True)
    matched=apply_fake_quant(model,cfg.get('mode','q3_moe'))
    tok=AutoTokenizer.from_pretrained(cfg['student_model'],trust_remote_code=True); ds=load_dataset('json',data_files=cfg['dataset'],split='train'); ml=int(cfg.get('max_length',8192))
    def enc(r):
        text=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=False) if 'messages' in r else r.get('text',''); z=tok(text,truncation=True,max_length=ml); z['labels']=z['input_ids'].copy(); return z
    ds=ds.map(enc,remove_columns=ds.column_names)
    out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True)
    args=TrainingArguments(output_dir=str(out/'trainer'),per_device_train_batch_size=int(cfg.get('batch_size',1)),gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps',8)),learning_rate=float(cfg.get('learning_rate',5e-6)),max_steps=int(cfg.get('max_steps',200)),bf16=True,logging_steps=1,save_steps=int(cfg.get('save_steps',100)),report_to=[],remove_unused_columns=False,fsdp='full_shard auto_wrap' if int(cfg.get('num_processes',2))>1 else '',fsdp_config={'use_orig_params':True,'cpu_ram_efficient_loading':True,'sync_module_states':True} if int(cfg.get('num_processes',2))>1 else None)
    tr=Trainer(model=model,args=args,train_dataset=ds,processing_class=tok,data_collator=DataCollatorForLanguageModeling(tok,mlm=False)); tr.train(); tr.save_model(str(out/'recovered')); tok.save_pretrained(out/'recovered')
    (out/'qat-manifest.json').write_text(json.dumps({'mode':cfg.get('mode','q3_moe'),'matched_parameters':matched,'note':'Re-run target quant backend after QAT; this checkpoint is not the deployment artifact.'},indent=2))
if __name__=='__main__': main()
