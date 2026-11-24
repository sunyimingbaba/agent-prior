import importlib.metadata as md,subprocess,sys,torch
from common import ROOT,OLD,PREVIOUS,REBUILD,ARMS,model_path,weight,sha,dump
def main():
    old=[OLD/'code'/f for f in ('base.py','pack.py','sample_panel.py','read_panel.py')]
    dependencies=[PREVIOUS/'code'/f for f in ('shared.py','policy.py','readouts.py','train.py')]
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    if (ROOT/'records/runtime.json').exists():return
    dump(ROOT/'records/runtime.json',dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,transformers=md.version('transformers'),peft=md.version('peft'),
        gpu=gpu,device_mount=subprocess.check_output(['df','-P',str(ROOT)],text=True).splitlines()[-1],init={s:weight(s) for s in 'ab'},
        dependencies={str(p):sha(p) for p in old+dependencies},new_training_code={p.name:sha(p) for p in sorted((ROOT/'code').glob('*.py')) if p.name in ('common.py','worker.py','run_train.py','build_rosters.py','contexts.py','evaluate.py','vectors.py','text_bank.py')}))
if __name__=='__main__':main()
