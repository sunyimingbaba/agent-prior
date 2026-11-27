"""Read out the archived hidden states through the live CPU worker and compare all winners."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]


def main():
    expected=json.loads((ROOT/"inputs/expected.json").read_text())
    checked=[]
    for seed in "ab":
        messages=[]; answers=[]
        for name,byseed in expected.items():
            p=ROOT.parent/"cold/records/real"/seed/"raw"/(name+".pt")
            t=torch.load(p,map_location="cpu",weights_only=True)
            for k in range(2):
                buf=base64.b64encode(t["hidden"][8*k:8*k+8].float().numpy().tobytes()).decode()
                messages.append(json.dumps(dict(image=name,data=buf)));answers.append(byseed[seed][k])
        proc=subprocess.run([sys.executable,"-Bu",str(ROOT/"code/target_worker.py"),seed],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OPENBLAS_NUM_THREADS="1"),input="\n".join(messages)+"\n",
            text=True,stdout=subprocess.PIPE,check=True)
        lines=proc.stdout.splitlines();assert lines[0]=="READY" and len(lines)==len(messages)+1
        for line,gold in zip(lines[1:],answers):assert json.loads(line)["winners"]==gold
        checked.append(len(messages))
    (ROOT/"results/在线评分核对.json").write_text(json.dumps(dict(images=len(expected),groups=checked,arms=3),indent=2)+"\n")

if __name__=="__main__":
    main()
