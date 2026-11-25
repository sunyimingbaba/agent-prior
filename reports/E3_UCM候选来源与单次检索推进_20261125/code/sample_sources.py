import importlib.util
import sys
import torch
from remote_common import BASE,ROOT,roster,directory,read,sha,table,rgb,setup,savept,dump
from eval_io import weights
from prefixes import greedy
from policy import policy,prompt

source=BASE/'attribute_questions_20261125/code/run.py'
spec=importlib.util.spec_from_file_location('frozen_question_sampler',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@torch.inference_mode()
def main(seed,tag):
    setup();torch.set_num_threads(1)
    rows=roster(tag);target=directory(tag,seed,'raw')
    path,adapter=weights(seed+'_deployed');model,proc=policy(path);data=table()
    cfg=model.generation_config
    assert cfg.top_k==50 and cfg.top_p==1 and cfg.repetition_penalty==1.05
    for j,row in enumerate(rows):
        name=row['image'];f=target/(name+'.pt')
        if f.exists():
            old=torch.load(f,map_location='cpu',weights_only=True)
            assert old['image']==name and old['rgb']==row['rgb_sha256'] and old['adapter']==adapter
            continue
        image=rgb(data,row);record=dict(image=name,rgb=row['rgb_sha256'],adapter=adapter,
                                        feedback=row['feedback'],modes={})
        for mode,query in [('scene',row['scene_query']),('question',row['scene_query']+' '+row['question'])]:
            anchor,actions,calls,_=greedy(model,proc,image,query,row['feedback'])
            inp=proc(text=[prompt(proc,query)],images=[image],return_tensors='pt').to('cuda')
            states=[];paths=[];rngs=[]
            for batch in range(2):
                rng=(10000000 if seed=='a' else 20000000)+row['row']*1000+batch*100
                h,ids=module.collect(model,proc,inp,list(range(8)),rng,row['feedback'])
                states.extend(h);paths.extend(ids);rngs.append(rng)
            vec=torch.stack(states+[anchor])
            assert vec.shape==(17,3584) and torch.isfinite(vec).all()
            record['modes'][mode]=dict(query=query,hidden=vec,paths=paths,greedy=actions,calls=calls,rngs=rngs)
        savept(f,record);print(tag,seed,j+1,len(rows),flush=True)
    dump(target/'complete.json',dict(adapter=adapter,sampler=sha(source),entry=sha(ROOT/'code/sample_sources.py'),lock=sha(ROOT/'inputs/lock.json'),
        files={r['image']:sha(target/(r['image']+'.pt')) for r in rows}))
if __name__=='__main__':main(*sys.argv[1:])
