"""Equal-input paired-picture intervention on saved greedy priors."""
from collections import defaultdict
import torch
from remote_common import ROOT,roster,read,sha,table,rgb,savept,setup
from readouts import load_model

@torch.inference_mode()
def main():
    setup();torch.set_num_threads(2);m=load_model(False);t=table()
    for tag in ('first','second'):
        rows=roster(tag);pairs=defaultdict(list)
        for c in rows:pairs[c['pair']].append(c)
        for seed in 'ab':
            dest=ROOT/'records'/tag/seed/'swapped.pt'
            src=dest.parent/'raw';index=read(src/'complete.json')
            if dest.is_file():
                assert torch.load(dest,weights_only=True)['proof']==sha(src/'complete.json')
                continue
            swapped={};max_self=0.
            for a,b in pairs.values():
                assert a['scene_query']==b['scene_query'] and a['question']==b['question'] and a['side']!=b['side']
                for c,other in ((a,b),(b,a)):
                    name=c['image'];peer=other['image'];f=src/(peer+'.pt')
                    assert sha(f)==index['files'][peer]
                    obj=torch.load(f,weights_only=True);pix=m.preprocess(rgb(t,c))[None]
                    vec={}
                    for mode,z in obj['modes'].items():
                        vec[mode]=m.img_embeds(pix,agent_hidden=z['hidden'][-1:].float())['image_feats'][0].clone()
                    swapped[name]=dict(peer=peer,visual=vec)
                    if c['pair']==rows[0]['pair']:
                        self_state=torch.load(src/(name+'.pt'),weights_only=True)
                        ref=torch.load(dest.parent/'features'/(name+'.pt'),weights_only=True)
                        for mode,z in self_state['modes'].items():
                            self_feature=m.img_embeds(pix,agent_hidden=z['hidden'][-1:].float())['image_feats'][0]
                            max_self=max(max_self,float((ref['modes'][mode]['visual'][-1]-self_feature).abs().max()))
            assert max_self<1e-4
            savept(dest,dict(conditions=swapped,adapter=index['adapter'],proof=sha(src/'complete.json'),max_self=max_self,
                protocol=sha(ROOT/'同图与配对图先验核查口径.md')))
            print(tag,seed,len(swapped),flush=True)
if __name__=='__main__':main()
