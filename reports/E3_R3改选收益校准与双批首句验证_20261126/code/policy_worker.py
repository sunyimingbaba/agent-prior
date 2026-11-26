"""Score only the on-policy training trajectories; the policy process holds its own GPU."""
import os
import sys
import base64
import json
from pathlib import Path
pipe=os.fdopen(os.dup(1),'w',buffering=1)
os.dup2(2,1)
import numpy as np
import torch
from online_reward import Scorer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'multicap_rfpo_20261124/code'))
from runtime import table,rgb,read,sha,setup
from readouts import load_model

@torch.inference_mode()
def main(seed):
    setup();torch.set_num_threads(3)
    model=load_model(False);images=table()
    cases={r['image']:r for r in read(ROOT/'inputs/pilot/train.json')[seed]}
    reward=Scorer(seed)
    print('READY',file=pipe)
    for line in sys.stdin:
        msg=json.loads(line);name=msg['image']
        pix=model.preprocess(rgb(images,cases[name]))[None]
        states=np.frombuffer(base64.b64decode(msg['data']),dtype=np.float32).reshape(8,3584)
        cached={};vectors=[]
        for h in states:
            key=h.tobytes()
            if key not in cached:
                cached[key]=model.img_embeds(pix,agent_hidden=torch.from_numpy(h.copy())[None])['image_feats'][0].double().numpy()
            vectors.append(cached[key])
        winners=reward.run(name,np.stack(vectors))
        for slots in winners.values():
            assert all((i in slots)==(j in slots) for i in range(8) for j in range(i) if np.array_equal(vectors[i],vectors[j]))
        print(json.dumps(dict(winners=winners,unique=len(cached))),file=pipe)

if __name__=='__main__':main(sys.argv[1])
