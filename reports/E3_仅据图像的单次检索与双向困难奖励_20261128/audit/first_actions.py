"""Independently connect each eight-way sample to the true first image/text context."""
from collections import Counter
import re
import numpy as np
import torch
from image_inputs import MODEL,ROOT,QUERY,ASSETS,AutoProcessor,read,load,sha,dump

def main():
    st=ROOT/'first_round';rows=read(st/'inputs/roster.json')['rows']
    proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    matched=ROOT/'matched';sigs=read(st/'inputs/train_lock.json');allcases={}
    fbs=read(matched/'inputs/feedback_verified.json')['images']
    results={}
    for seed in 'ab':
        paired={}
        for arm in ('r2','strict'):
            name=seed+'_'+arm;d=st/'models'/name;steps=read(d/'steps.json');cp=read(d/'complete.json')
            assert cp['steps']==cp['scheduled_slots']==len(steps)==len(rows)==24
            assert cp['lock']==sha(st/'inputs/train_lock.json') and cp['scores']==sha(d/'steps.json')
            ref=ASSETS/'e3-image-only-pair-reward-20261128/first_round/models'/name/'adapter/adapter_model.safetensors'
            assert cp['final']==sha(ref) and cp['initial']==sigs['starts'][seed]
            stat=Counter();pair=None
            for i,(r,step) in enumerate(zip(rows,steps,strict=True)):
                n=r['image'];origin=matched/'contexts'/seed/(n+'.pt');first=load(origin)
                pf=d/'trajectories'/(n+'.pt');record=load(pf)
                assert record['image']==step['image']==n and record['reference']==sha(origin)==step['context']
                assert record['query']==step['query']==first['query']==QUERY and sha(pf)==step['trajectory']
                assert record['image_input_token_hash']==first['token_hash'] and record['pixel_values']==first['pixel_values_sha']
                assert record['context_batch_sizes'][0]==8 and torch.equal(record['context_ids'][0],first['first_input_ids'])
                assert record['feedback']==fbs[n]['feedback']
                assert len(record['actions'])==8 and record['terminal_hidden'].shape==(8,3584)
                assert record['rng']==step['seed']==({'a':231000000,'b':241000000}[seed]+1000*i)
                rewards=np.array(step['r2'] if arm=='r2' else step['r3']);adv=(rewards-rewards.mean())/(rewards.std(ddof=1)+1e-8)
                np.testing.assert_allclose(step['advantage'],adv,rtol=0,atol=1e-13)
                ntokens=0
                for j,(trajectory,log) in enumerate(zip(record['actions'],step['actions'],strict=True)):
                    assert 1<=len(trajectory)<=3 and trajectory[0]['context']==0 and len(log)==len(trajectory)
                    stat['sampled_trajectories']+=1
                    stat['first_eos_only']+=int(trajectory[0]['ids']==[151645])
                    stat['trajectories_continued']+=int(len(trajectory)>1)
                    for k,(act,item) in enumerate(zip(trajectory,log,strict=True)):
                        assert all(act[key]==item[key] for key in ('ids','text','context','sampled_logp'))
                        assert proc.decode(act['ids'],skip_special_tokens=True).strip()==act['text'].strip()
                        if abs(adv[j])>1e-10:ntokens+=len(act['ids'])
                        if k+1<len(trajectory):
                            assert re.search(r'<tool_call>\s*scene_classify\b',act['text'])
                            add=proc(text=[act['text']+'\n<result>'+fbs[n]['feedback']+'</result>\n'],return_tensors='pt').input_ids
                            assert torch.equal(record['context_ids'][trajectory[k+1]['context']],torch.cat((record['context_ids'][act['context']],add),dim=1))
                assert ntokens==step['replayed_tokens']
                if i==0:pair=(record,step)
                stat['nonzero_gradients']+=int(step['norm']>0)
                stat['distinct_strict_advantages']+=int(step['adv2']!=step['adv3'])
            paired[arm]=pair
            assert cp['nonzero']==stat['nonzero_gradients'] and stat['sampled_trajectories']==192
            results[name]=dict(**stat,steps=len(steps),adapter=cp['final'])
        a,ha=paired['r2'];b,hb=paired['strict']
        assert ha['actions']==hb['actions'] and torch.equal(a['terminal_hidden'],b['terminal_hidden'])
        assert all(torch.equal(u,v) for u,v in zip(a['context_ids'],b['context_ids'],strict=True))
    dump('首轮训练动作与末步权重独立核查.json',results)

if __name__=='__main__':
    torch.set_num_threads(2);main()
