import subprocess,sys
import torch,peft,transformers
from runtime import ROOT,TRAIN,OLD,FORM,BASE,ASSETS,read,dump,sha
def main():
    mount=subprocess.check_output(['findmnt','-T',str(ROOT),'-n','-o','TARGET,FSTYPE'],text=True).strip()
    assert mount.split()[0]=='/private'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    cpu=subprocess.check_output([str(BASE/'env/bin/python'),'-c','import sys,torch;print(sys.version);print(torch.__version__)'],text=True).strip().splitlines()
    paths=[TRAIN/'code'/x for x in ('train.py','readouts.py','policy.py','shared.py')]+[BASE/'CUP/agent/rollout.py',BASE/'CUP/agent/grpo/reward.py',BASE/'CUP/agent/grpo/scorer.py',BASE/'CUP/vilt/modules/vilt_module.py',ASSETS/'ucm_main_domain_tool.ckpt',OLD/'inputs/continuous_lock.json',FORM/'inputs/ucm.json']
    dump(ROOT/'records/runtime.json',dict(mount=mount,gpu=gpu,python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,peft=peft.__version__,transformers=transformers.__version__,cpu=cpu,sources={str(p):sha(p) for p in paths},roster=sha(ROOT/'inputs/roster.json'),panel=sha(ROOT/'inputs/panel.json')))
if __name__=='__main__':main()
