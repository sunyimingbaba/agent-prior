from pathlib import Path
import json
import time
from milestone_common import ROOT,PLAN,setup,seed,case_images,real_tools
from quantity_model import QuantityScorer
from live_rollout_eos import tensor_sha
from stage_io import read,dump,sha
torch=setup();seed(0);start=time.monotonic();cases=read(ROOT/'inputs/cases.json');by={x['image']:x for x in cases};scorer=QuantityScorer(cases);tools=real_tools();actual={n:tools['scene_classify'](x['image']) for n,x in case_images(cases).items()};records=[];features=[];qs=[];replay=0.;new=[];old=read(ROOT/'historical/records.json');out=ROOT/'scoring';out.mkdir()
for policy in PLAN['policies']:
    name=policy['name'];rows=[json.loads(s) for s in (ROOT/'generation'/name/'records.jsonl').read_text().splitlines()];assert len(rows)==128
    greedy=[x for x in old if x['policy']==name and x['role']=='greedy'];assert len(greedy)==8
    for x in rows+greedy:
        sf=ROOT/x['state_file'];assert sha(sf)==x['state_file_sha256'];h=torch.load(sf,map_location='cpu',weights_only=True);assert tensor_sha(h)==x['generation_state_sha256'];trace=read(ROOT/x['trace_file']);assert sha(ROOT/x['trace_file'])==x['trace_sha256']
        for e in trace['tool_events']:assert e['origin']=='model_requested_actual_tool' and e['text']==actual[x['image']]['text'] and abs(e['confidence']-actual[x['image']]['confidence'])<1e-6
        for target in cases:
            value=scorer.score(h,target);role='new_sample' if x['role']=='candidate' else 'reused_greedy';i=f"{role}__{x['id']}__target_{target['image']}"
            row={'id':i,'policy':name,'donor_id':x['id'],'donor_image':x['image'],'target_image':target['image'],'actual_count':target['actual_count'],'object':target['object'],'donor_actual_count':by[x['image']]['actual_count'],'same_image':x['image']==target['image'],'same_object':by[x['image']]['object']==target['object'],'grid_role':role,'sample':x.get('sample'), 'seed':x['seed'],'donor_quality':x['quality_usable'],'donor_strict_quality':x['strict_protocol_usable'],'donor_tools_used':x['tools_used'],'state_file':x['state_file'],'state_file_sha256':x['state_file_sha256'],'generation_state_sha256':x['generation_state_sha256'],'scorer_state_sha256':tensor_sha(h),'trace_file':x['trace_file'],'trace_sha256':x['trace_sha256'],**value}
            if role=='reused_greedy' and row['same_image']:
                for metric in ['A_positive','A_negative','B_positive','B_negative','margin_A','margin_B']:replay=max(replay,abs(value[metric]-x[metric]));assert abs(value[metric]-x[metric])<1e-6
            records.append(row);features.append(scorer.last_feature);qs.append(scorer.last_qsim)
            with (out/'records.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        dump(ROOT/'progress.json',{'phase':'grid_scoring','completed':len(records),'total':2176})
        if len(records)%128==0:print('PERMUTATION_GRID_SCORED',len(records),'/2176',flush=True)
assert len(records)==2176 and len({x['id'] for x in records})==2176
dump(out/'all_scores.json',records);torch.save({'record_ids':[x['id'] for x in records],'image_features':torch.cat(features),'quantity_similarities':torch.stack(qs)},out/'tensors.pt');dump(out/'actual_tools.json',actual)
dump(out/'result.json',{'ok':True,'records':2176,'new_sample_grid_records':2048,'reused_greedy_grid_records':128,'greedy_diagonal_replay_max_error':replay,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-start});print('ALL_2176_GRID_SCORES_COMPLETE',flush=True)
