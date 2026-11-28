"""Replay all frozen image readouts on the original pixels with the saved terminal states."""
import sys
import time

import numpy as np
import torch
from image_inputs import IMAGE, ROOT, READOUT, CONVERT, QUERY, tables, image, load, read, sha, dump

@torch.inference_mode()
def main(variant):
    original=read(ROOT/'audit/通用问题与原始输入独立核验.json')
    dest=IMAGE if variant=='image_only' else ROOT/variant/'evaluation'
    source=tables(); rows=read(IMAGE/'inputs/roster.json')
    expected=original['expected_tokens_by_image_size']['256x256']['digest']
    model=torch.jit.load(str(READOUT/'cup_native_trace.pt'),map_location='cpu').eval()
    converted={r['rgb']:CONVERT(image(r,source)) for r in rows}
    for seed in 'ab':
        for arm in ('r2','strict'):
            for panel in ('official','extra'):
                tag=seed+'_'+arm; output=ROOT/'audit'/(variant+'_'+tag+'_'+panel+'_终态读出.json')
                d=dest/'records'/tag/panel
                while not (d/'complete.json').is_file():time.sleep(4)
                cp=read(d/'complete.json')
                if output.exists():
                    assert read(output)['receipt']==sha(d/'complete.json')
                    continue
                feature=load(d/'features.pt'); selected=[r for r in rows if r['panel']==panel]
                assert feature['images']==[r['image'] for r in selected] and sha(d/'features.pt')==cp['features']
                lock=read(dest/'inputs/manifest.json')
                assert feature['adapter']==cp['adapter']==lock['policies'][tag]['adapter']
                vexpected=[];approximations=[];feedback_counts=[]
                for i in range(0,len(selected),4):
                    current=selected[i:i+4];hs=[];xs=[]
                    for j,r in enumerate(current):
                        f=d/(r['image']+'.pt');o=load(f)
                        assert sha(f)==cp['images'][r['image']] and o['adapter']==cp['adapter']
                        assert o['actual_query']==o['query']==QUERY and o['rgb']==r['rgb'] and o['feedback']==r['feedback']
                        assert o['template']==original['expected_chat_template_sha256'] and o['first_input_ids_sha256']==expected
                        assert all(v==r['feedback'] for v in o['calls'])
                        assert o['actions'] and o['prior'].shape==(3584,) and torch.isfinite(o['prior']).all()
                        assert torch.equal(o['v'],feature['features'][i+j]);vexpected.append(o['v'])
                        xs.append(converted[r['rgb']]);hs.append(o['prior'].float());feedback_counts.append(len(o['calls']))
                    approximations.append(model(torch.stack(xs),torch.stack(hs)).cpu())
                em=torch.cat(approximations);vs=torch.stack(vexpected)
                errors=(em-vs).abs().max(1).values
                assert errors.max()<3e-4,(variant,tag,panel,float(errors.max()))
                metrics=dict(receipt=sha(d/'complete.json'),image_count=len(selected),max_abs_error=float(errors.max()),
                            median_max_abs_error=float(errors.median()),mean_cosine=float(torch.nn.functional.cosine_similarity(em,vs).mean()),
                            tool_calls=int(sum(feedback_counts)),images_requesting_tool=int(np.count_nonzero(feedback_counts)),
                            verified_prompt_tokens=expected)
                dump(output.name,metrics);print(variant,tag,panel,metrics['max_abs_error'],flush=True)
if __name__=='__main__':
    torch.set_num_threads(4);main(sys.argv[1])
