import sys,subprocess,torch,importlib.metadata as md
from pathlib import Path
from env_pair import ROOT,OLD,HIST,PREVIOUS,sha,dump,weight,ASSETS
def main():
    deps=[OLD/'code'/n for n in ('sample_panel.py','read_panel.py','pack.py','base.py')]
    deps.extend(PREVIOUS/'code'/n for n in ('shared.py','readouts.py','policy.py','train.py'))
    deps.extend(ASSETS.parent/'CUP'/n for n in ('agent/rollout.py','vilt/modules/vilt_module.py'))
    dump(ROOT/'records/environment.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
        mount=subprocess.check_output(['df','-P',str(ROOT)],text=True).splitlines()[-1].split()[-1],python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,transformers=md.version('transformers'),peft=md.version('peft'),
        starting={s:weight(s) for s in 'ab'},policy_inputs=sha(ROOT/'inputs/candidates.json'),prior_panel=sha(HIST/'inputs/panel.json'),blind_panel=sha(ROOT/'confirm/inputs/panel.json'),frozen_reader=sha(ASSETS/'ucm_main_domain_tool.ckpt'),sources={str(p):sha(p) for p in deps}))
if __name__=='__main__':main()
