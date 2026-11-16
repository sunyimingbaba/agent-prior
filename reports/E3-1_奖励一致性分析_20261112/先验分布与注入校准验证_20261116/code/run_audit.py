from pathlib import Path
import json
import time
import torch
from milestone_common import ROOT,PLAN,setup,seed,case_images
from quantity_model import QuantityScorer
from analysis_common import quantity_metrics
from path_runtime import tensor_sha,model_hash,intervene,clear
from stage_io import read,dump,sha
setup();seed(0);start=time.monotonic();cases=read(ROOT/'inputs/evaluation_cases.json');by={x['image']:x for x in cases};train=PLAN['training_images'];families=['original_cached','initial_SFT30','balanced32'];records=read(ROOT/'states/records.json');byid={x['id']:x for x in records};scorer=QuantityScorer(cases);m=scorer.pl;assert all(not p.requires_grad for p in m.parameters());before=model_hash(m)
H=torch.stack([torch.load(ROOT/x['state_file'],map_location='cpu',weights_only=True) for x in records]);assert H.shape==(511,3584)
linear=[];projected=[];raw_gate=[];ratio=[];first_prompt=[];prompt_norm=[]
with torch.no_grad():
    learned=m.visual_prompt_dropout(m.visual_prompt_proj(m.visual_prompt_embeddings));assert learned.shape==(1,16,1024)
    for h in H:
        a=m.agent_proj[0](h[None].float().cuda());z=m.agent_proj[1](a);expanded=z[:,None,:].repeat(1,16,1);g=m.cluster_mapping(expanded);assert float(g.max()-g.min())==0.;rr=g.clamp(0,1);prompt=m.visual_prompt_gather_dropout(m.visual_prompt_proj_gather((1-rr)*expanded+rr*learned))
        linear.append(a.cpu()[0]);projected.append(z.cpu()[0]);raw_gate.append(float(g[0,0,0]));ratio.append(float(rr[0,0,0]));first_prompt.append(prompt.cpu()[0,0]);prompt_norm.append(float(prompt.norm()))
L=torch.stack(linear);Z=torch.stack(projected);offsets={};ix={x['id']:i for i,x in enumerate(records)}
orig_ids=[f'original__{n}' for n in train];origmean=L[[ix[n] for n in orig_ids]].mean(0)
for family in families:
    offsets[family]=torch.zeros(1,1024) if family=='original_cached' else (origmean-L[[ix[f'greedy__{family}__{n}'] for n in train]].mean(0))[None]
(ROOT/'path').mkdir();weights={}
for name in ['agent_proj','cluster_mapping','visual_prompt_proj','visual_prompt_embeddings','visual_prompt_proj_gather']:
    obj=getattr(m,name)
    if isinstance(obj,torch.Tensor):weights[name]=obj.detach().cpu()
    else:
        for k,v in obj.state_dict().items():weights[name+'.'+k]=v.detach().cpu()
torch.save(weights,ROOT/'path/weights.pt');torch.save({'record_ids':[x['id'] for x in records],'hidden':H,'linear':L,'projected':Z,'gate_raw':torch.tensor(raw_gate),'gate_clipped':torch.tensor(ratio),'prompt_first':torch.stack(first_prompt),'prompt_norm':torch.tensor(prompt_norm),'learned_prompt':learned.cpu()[0],'offsets':{k:v.cpu() for k,v in offsets.items()}},ROOT/'path/tensors.pt')
dump(ROOT/'path/architecture.json',{'agent_proj':str(m.agent_proj),'layernorm_eps':m.agent_proj[1].eps,'prompt_length':m.prompt_length,'agent_hidden_dim':3584,'visual_prompt_dim':1024,'module_types':{n:type(getattr(m,n)).__name__ for n in ['agent_proj','cluster_mapping','visual_prompt_proj','visual_prompt_proj_gather']},'source_lines':{'vilt/modules/vilt_module.py':[584,605],'prompt_clip/model.py':[240,275]},'original_model_parameter_hash':before})
# Four actual native-forward witnesses record the insertion into the vision transformer.
for sid in ['original__102.tif','greedy__initial_SFT30__102.tif','greedy__balanced32__102.tif','original__1937.tif']:
    x=byid[sid];captured={};hooks=[]
    for name,module in [('agent_linear',m.agent_proj[0]),('agent_projected',m.agent_proj[1]),('gate_raw',m.cluster_mapping),('prompt',m.visual_prompt_proj_gather),('vision_preln',m.model.visual.ln_pre)]:
        hooks.append(module.register_forward_hook(lambda module,args,out,key=name:captured.__setitem__(key,out.detach().cpu())))
    hooks.append(m.model.visual.transformer.register_forward_pre_hook(lambda module,args:captured.__setitem__('transformer_input',args[0].detach().cpu())))
    try:scorer.score(H[ix[sid]],by[x['image']])
    finally:clear(hooks)
    seq=captured['vision_preln'].permute(1,0,2);expected=torch.cat([seq[:1],captured['prompt'].permute(1,0,2),seq[1:]],dim=0);assert torch.equal(expected,captured['transformer_input']);torch.save(captured,ROOT/'path'/('witness_'+sid+'.pt'))
# Full grid under five predeclared frozen-forward interventions.
out=ROOT/'scoring';out.mkdir();allrows=[];features=[];qs=[];vs=[];vids=[];gallery=scorer.gallery
base_rows=read(ROOT/'baseline/records.json');base_map={(('initial_SFT30' if x['arm']=='before' else 'balanced32'),x['image']):x for x in base_rows if x['arm'] in ['before','after'] and x['role']=='greedy'};max_replay=0.
def score_one(sid,target,variant,phase,role):
    global max_replay
    x=byid[sid];h=H[ix[sid]];hooks=intervene(m,variant,offsets[x['family']].cuda())
    try:value=scorer.score(h,by[target])
    finally:clear(hooks)
    q=scorer.last_qsim;met=quantity_metrics(q,gallery,by[target]);rid=f'{phase}__{variant}__{sid}__target_{target}'
    row={'id':rid,'phase':phase,'role':role,'variant':variant,'family':x['family'],'donor_id':sid,'donor_image':x['image'],'target_image':target,'target_split':by[target]['split'],'actual_count':by[target]['actual_count'],'same_image':x['image']==target,'same_object':by[x['image']]['object']==by[target]['object'],'state_numeric_sha256':x['numeric_state_sha256'],'scorer_state_sha256':tensor_sha(h),'state_file':x['state_file'],**value,**met}
    if variant=='auto' and x['role']=='greedy' and x['image']==target:
        old=base_map[x['family'],target]
        for k in ['A_positive','A_negative','B_positive','B_negative','margin_A','margin_B']:err=abs(value[k]-old[k]);max_replay=max(max_replay,err);assert err<1e-6
    if variant=='auto' and x['family']=='original_cached' and x['image']==target:
        for w in ['A','B']:
            for sign in ['positive','negative']:assert abs(value[w+'_'+sign]-by[target]['pairs'][w]['historical_'+sign+'_score'])<1e-6
    allrows.append(row);features.append(scorer.last_feature);qs.append(q)
    if scorer.last_vsim is not None:vids.append(rid);vs.append(scorer.last_vsim)
    with (out/'records.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return row
for family in families:
    for variant in PLAN['variants']:
        for donor in train:
            sid=f'original__{donor}' if family=='original_cached' else f'greedy__{family}__{donor}'
            for target in train:score_one(sid,target,variant,'diagnostic_grid','grid')
        dump(ROOT/'progress.json',{'phase':'diagnostic_grid','completed':len(allrows),'total':1470});print('AUDIT_SCORED',len(allrows),'/1470',flush=True)
assert len(allrows)==960
choices=[]
for rank,v in enumerate(PLAN['variants']):
    xs=[x for x in allrows if x['phase']=='diagnostic_grid' and x['family']=='balanced32' and x['variant']==v and x['same_image']];assert len(xs)==8
    choices.append({'variant':v,'tie_rank':rank,'A_hard_macro':sum(x['A_hard_correct'] for x in xs)/8,'A_margin_macro':sum(x['A_hard_margin'] for x in xs)/8})
best=max(choices,key=lambda x:(x['A_hard_macro'],x['A_margin_macro'],-x['tie_rank']));selected=best['variant'];dump(ROOT/'selection.json',{'selected_variant':selected,'primary_policy':'balanced32','training_images':train,'A_only_candidates':choices,'B_used_for_selection':False,'freeze_before_confirmation':True,'records_at_selection':960});print('CANDIDATE_LOCKED',selected,flush=True)
for family in ['initial_SFT30','balanced32']:
    for c in cases:
        target=c['image'];score_one(f'greedy__{family}__{target}',target,selected,'confirmation','selected_own');score_one(f'greedy__{family}__102.tif',target,selected,'confirmation','selected_shared')
for c in cases:
    score_one(f'greedy__balanced32__{c["image"]}',c['image'],'gate_1','confirmation','learned_only');score_one('original__'+c['image'],c['image'],'auto','confirmation','original_cached')
assert len(allrows)==1470 and len({x['id']+'__'+x['role'] for x in allrows})==1470
# Multiple roles may share the same numerical forward; IDs include role for an unambiguous tensor index.
for i,x in enumerate(allrows):x['id']=x['id']+'__'+x['role']
# Re-map val IDs to the amended role-specific IDs in their original scoring order.
vids=[x['id'] for x in allrows if x['target_split']=='val'];dump(out/'all_scores.json',allrows);torch.save({'record_ids':[x['id'] for x in allrows],'image_features':torch.cat(features),'quantity_similarities':torch.stack(qs),'val_record_ids':vids,'val_similarities':torch.stack(vs)},out/'tensors.pt')
after=model_hash(m);assert before==after and all(not p.requires_grad for p in m.parameters())
dump(ROOT/'gpu_result.json',{'ok':True,'scores':1470,'grid_scores':960,'confirmation_scores':510,'projection_states':511,'selected_variant':selected,'original_model_parameter_hash':before,'final_model_parameter_hash':after,'native_replay_max_error':max_replay,'optimizer_updates':0,'new_Qwen_generations':0,'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()});print('ALL_AUDIT_AND_CONFIRMATION_SCORES_COMPLETE',flush=True)
