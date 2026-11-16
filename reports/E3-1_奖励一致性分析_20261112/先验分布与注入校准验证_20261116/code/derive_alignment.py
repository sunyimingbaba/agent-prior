import argparse
import csv
from pathlib import Path
import torch
from stage_io import read,dump
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;raw=r/'原始运行记录';torch.set_num_threads(2);states=read(raw/'states/records.json');t=torch.load(raw/'path/tensors.pt',map_location='cpu',weights_only=True);w=torch.load(raw/'path/weights.pt',map_location='cpu',weights_only=True);eps=read(raw/'path/architecture.json')['layernorm_eps'];byid={x['id']:i for i,x in enumerate(states)};rows=[]
for fam in ['initial_SFT30','balanced32']:
    xs=[x for x in states if x['family']==fam and x['role']=='greedy']
    for x in xs:
        i=byid[x['id']];j=byid['original__'+x['image']];z=t['linear'][i].double()+t['offsets'][fam][0].double();aligned=(z-z.mean())/torch.sqrt(z.var(unbiased=False)+eps)*w['agent_proj.1.weight'].double()+w['agent_proj.1.bias'].double();g=float((aligned@w['cluster_mapping.weight'][0].double()+w['cluster_mapping.bias'][0].double()).clamp(0,1));co='training8' if x['image'] in read(r/'plan.json')['training_images'] else 'confirmation55' if x['split']=='train' else 'existing_val22'
        rows.append({'family':fam,'image':x['image'],'cohort':co,'native_to_original_projected_cosine':float(torch.nn.functional.cosine_similarity(t['projected'][i].double(),t['projected'][j].double(),dim=0)),'aligned_to_original_projected_cosine':float(torch.nn.functional.cosine_similarity(aligned,t['projected'][j].double(),dim=0)),'native_gate':float(t['gate_clipped'][i]),'aligned_gate_float64_reconstruction':g})
with (r/'alignment_by_image.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
summary={}
for fam in ['initial_SFT30','balanced32']:
    for co in ['training8','confirmation55','existing_val22']:
        xs=[x for x in rows if x['family']==fam and x['cohort']==co];summary[fam+'/'+co]={k:sum(x[k] for x in xs)/len(xs) for k in ['native_to_original_projected_cosine','aligned_to_original_projected_cosine','native_gate','aligned_gate_float64_reconstruction']}
dump(r/'alignment_summary.json',summary);print('ALIGNMENT_RECONSTRUCTION_EXPORTED')
