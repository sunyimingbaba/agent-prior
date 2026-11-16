from pathlib import Path
import itertools
import time
import torch
from milestone_common import ROOT,PLAN,setup,seed
from quantity_model import QuantityScorer
from path_runtime_v2 import intervene,clear,model_hash
from analysis_common import quantity_metrics
from stage_io import read,dump
while not (ROOT/'gpu_retry_result.json').exists():time.sleep(5)
assert read(ROOT/'gpu_retry_result.json')['status']=='complete';setup();seed(0);cases=[x for x in read(ROOT/'inputs/evaluation_cases.json') if x['image'] in PLAN['training_images']];by={x['image']:x for x in cases};scorer=QuantityScorer(cases);m=scorer.pl;before=model_hash(m);assert before==read(ROOT/'gpu_result.json')['original_model_parameter_hash'];ts=torch.load(ROOT/'path/tensors.pt',map_location='cpu',weights_only=True);ix={x:i for i,x in enumerate(ts['record_ids'])};selected=read(ROOT/'selection.json')['selected_variant'];rows=[];grads=[];directions=[];hiddens=[];qrows=[]
for family in ['initial_SFT30','balanced32']:
    for n in PLAN['training_images']:
        sid=f'greedy__{family}__{n}';c=by[n];h0=ts['hidden'][ix[sid]].float().cuda();cols=scorer.gallery['quantity_columns'][n]['A'];texts=scorer.gallery['quantity_text_order'];prefix=texts[cols[0]].rsplit(' ',1)[0];pool=[i for i,s in enumerate(texts) if s.rsplit(' ',1)[0]==prefix];wrong=[i for i in pool if i!=cols[0]]
        for role,variant in [('native','auto'),('selected',selected)]:
            h=h0.detach().clone().requires_grad_(True);hooks=intervene(m,variant,ts['offsets'][family].cuda())
            try:
                visual=m.img_embeds(scorer.images[n],agent_hidden=h[None])['image_feats'];q=(visual@scorer.qgpu.T).flatten();negative=wrong[int(q[wrong].argmax())];margin=q[cols[0]]-q[negative];grad=torch.autograd.grad(margin,h)[0];norm=float(grad.norm());seed(2026111602+len(rows))
                direction=grad.detach()/grad.norm() if norm>1e-12 else torch.nn.functional.normalize(torch.randn_like(h),dim=0)
                values=[];gates=[];negatives=[]
                with torch.no_grad():
                    for sign in [-1.,1.]:
                        hp=h0+sign*.1*direction;z=m.agent_proj(hp[None]);raw=m.cluster_mapping(z[:,None,:].repeat(1,16,1));gates.append(float(raw[0,0,0].clamp(0,1)));vp=m.img_embeds(scorer.images[n],agent_hidden=hp[None])['image_feats'];qp=(vp@scorer.qgpu.T).flatten();negatives.append(wrong[int(qp[wrong].argmax())]);values.append(float(qp[cols[0]]-qp[wrong].max()))
                rows.append({'family':family,'image':n,'role':role,'variant':variant,'source_state_id':sid,'A_hard_margin':float(margin.detach()),'negative_column':negative,'gradient_norm':norm,'directional_derivative':float((grad.detach()*direction).sum()),'fd_epsilon':.1,'fd_central_derivative':(values[1]-values[0])/.2,'margin_minus':values[0],'margin_plus':values[1],'gate_minus':gates[0],'gate_plus':gates[1],'negative_minus':negatives[0],'negative_plus':negatives[1]});grads.append(grad.detach().cpu());directions.append(direction.detach().cpu());hiddens.append(h0.cpu());qrows.append(q.detach().cpu())
            finally:clear(hooks)
            del h,visual,q,margin,grad
assert len(rows)==32 and model_hash(m)==before and all(not p.requires_grad and p.grad is None for p in m.parameters())
out=ROOT/'gradient';out.mkdir();dump(out/'records.json',rows);torch.save({'hidden':torch.stack(hiddens),'gradient':torch.stack(grads),'direction':torch.stack(directions),'quantity_similarities':torch.stack(qrows)},out/'tensors.pt');dump(ROOT/'gradient_result.json',{'ok':True,'input_gradient_witnesses':32,'model_hash_unchanged':before,'optimizer_updates':0,'new_Qwen_generations':0,'contract':'F32 local sensitivity around dequantized BF16 cached states; epsilon0.1; no parameter update'});print('ALL_32_INPUT_GRADIENT_WITNESSES_COMPLETE',flush=True)
