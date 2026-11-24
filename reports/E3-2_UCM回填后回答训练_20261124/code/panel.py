"""Fresh post-tool actions and held-out caption ranks on the fixed training images."""
import sys
import numpy as np
import torch
from shared import ROOT,read,dump,sha,setup,table,rgb
from train import sample,finish,SAMPLES
from policy import policy
from reader_client import Reader
from metrics import csvwrite

@torch.inference_mode()
def main(arm):
    setup()
    model,proc=policy(arm)
    reader=Reader('panel')
    contexts=torch.load(ROOT/'contexts.pt',weights_only=True,map_location='cpu')
    cases=read(SAMPLES)
    t=table()
    rows=[]
    for i,c in enumerate(cases):
        ctx=contexts[c['image']]
        torch.manual_seed(921000+i)
        prefix,tr=sample(model,proc,rgb(t,c),c['scene_query'],ctx)
        hs,third=finish(model,proc,prefix,tr,ctx['answer'])
        scores=reader.score(c['image'],hs)
        for j,(result,record) in enumerate(zip(scores,tr,strict=True)):
            rows.append(dict(arm=arm,image=c['image'],pair=c['pair'],slot=j,margin=result['margin'],mrr=result['mrr'],
                second_tokens=' '.join(map(str,record[0])),second_eos_only=len(record[0])==1 and record[0][0] in model.generation_config.eos_token_id,
                third_round=bool(third[j])))
    reader.close()
    out=ROOT/'panel';out.mkdir(exist_ok=True)
    f=out/(arm+'.csv')
    csvwrite(f,rows)
    dump(out/(arm+'.json'),dict(n=len(rows),sha=sha(f),adapter=sha(ROOT/arm/'adapter/adapter_model.safetensors')))
if __name__=='__main__':main(sys.argv[1])
