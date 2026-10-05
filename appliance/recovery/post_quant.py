from __future__ import annotations
import argparse,json
from pathlib import Path
from appliance.stages.sft import main as sft_main

# Post-quant recovery intentionally reuses the proven SFT/QLoRA stage. The recovery
# dataset contains teacher-success/student-failure replay plus general retention data.
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args()
    # stages.sft already consumes --config from sys.argv; invoke as module-compatible path.
    import sys,runpy
    sys.argv=['appliance.stages.sft','--config',a.config]
    runpy.run_module('appliance.stages.sft',run_name='__main__')
if __name__=='__main__': main()
