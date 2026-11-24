"""Generate exactly one greedy deployment prior per validation image with the real scene tool."""
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import torch
from shared import ROOT, REBUILD, ASSETS, read, dump, sha, setup, table, rgb
from reader_client import Reader
from policy import policy
from rollout import rollout_one, SYSTEM_PROMPT
from tools import scene_classify_with_conf

@torch.inference_mode()
def run(arm):
    setup()
    inputs = REBUILD/'paired_views_20261120/inputs'
    init = read(ROOT/'initial.json')
    weight = sha(ROOT/arm/'adapter/adapter_model.safetensors')
    source = init['adapter'] if arm=='R0' else read(ROOT/arm/'complete.json')['final']
    assert source == weight and sha(inputs/'validation.json')==init['val']
    lock = read(inputs/'lock.json')
    assert sha(inputs/'retrieval_text.pt') == lock['retrieval_text.pt']
    assert sha(inputs/'e1_extension.json') == lock['e1_extension.json']
    for name, expected in read(inputs/'e1_extension.json')['files'].items():
        assert sha(inputs/'e1'/name)==expected
    model, proc = policy(arm)
    reader=Reader('eval')
    answers=read(ROOT/'tools.json')
    val,t=read(inputs/'validation.json'),table('val')
    assert len(val)==len(t)==210
    out = ROOT/arm/'eval'
    out.mkdir(exist_ok=True)
    generator = model.generate
    def record(**kw):
        response = generator(**kw)
        ids = response.sequences[0,kw['input_ids'].shape[1]:].tolist()
        rounds.append(dict(length=len(ids),eos=ids[-1] in model.generation_config.eos_token_id,
                           ids=ids,text=proc.decode(ids,skip_special_tokens=True)))
        return response
    model.generate = record
    def save_feature(entry,path):
        v=torch.tensor(reader.score(entry['image'],[entry['prior']])[0]['feature'])
        assert torch.isfinite(v).all() and abs(v.norm().item()-1)<1e-4
        entry['feature']=v
        tmp=path.with_suffix('.part')
        torch.save(entry,tmp)
        tmp.replace(path)
    executor=ThreadPoolExecutor(max_workers=1)
    pending=deque()
    for case in val:
        path = out/f"{case['row']:03d}.pt"
        if path.exists():
            entry = torch.load(path,map_location='cpu',weights_only=True)
            assert entry['adapter']==weight and entry['image']==case['image']
        else:
            im = rgb(t,case)
            rounds,events = [],[]
            def scene(image):
                if not events:
                    answer, conf = answers[case['rgb_sha256']]
                    events.append((answer,float(conf)))
                else:
                    events.append(events[0])
                return events[-1][0]
            h,used,filtered = rollout_one(model,proc,im,case['orig_query'],{'scene_classify':scene},3,64,
                                         'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
            assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all() and used==len(events) and filtered==0
            entry = dict(image=case['image'],row=case['row'],rgb=case['rgb_sha256'],adapter=weight,query=case['orig_query'],
                         rounds=rounds,tools=events,prior=h)
            pending.append(executor.submit(save_feature,entry,path))
            if len(pending)>=12:
                pending.popleft().result()
        if (case['row']+1)%10==0:
            print(arm,case['row']+1,flush=True)
    for job in pending:job.result()
    executor.shutdown()
    rows=[sha(out/f"{c['row']:03d}.pt") for c in val]
    reader.close()
    dump(out/'complete.json', dict(adapter=weight,rows=rows,gallery=lock['retrieval_text.pt'],
         reader=sha(ASSETS/'ucm_main_domain_tool.ckpt'),tool=sha(ASSETS/'ucm_scene_cls.pt')))

if __name__=='__main__':
    run(sys.argv[1])
