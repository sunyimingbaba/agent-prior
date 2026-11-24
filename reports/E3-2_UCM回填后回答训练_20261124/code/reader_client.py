import base64,json,os,subprocess
from pathlib import Path
import torch
from shared import BASE

class Reader:
    def __init__(self,mode):
        self.proc=subprocess.Popen([str(BASE/'env/bin/python'),'-u',str(Path(__file__).with_name('reader_worker.py')),mode],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        assert self.proc.stdout.readline().strip()=='READY','reader startup failed'
    def score(self,image,states):
        payload=base64.b64encode(torch.stack(states).float().cpu().numpy().tobytes()).decode()
        self.proc.stdin.write(json.dumps(dict(image=image,hidden=payload))+'\n')
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())
    def close(self):
        self.proc.stdin.close()
        assert self.proc.wait()==0
