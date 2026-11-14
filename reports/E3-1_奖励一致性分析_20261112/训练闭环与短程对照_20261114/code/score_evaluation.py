import io
import json
import shutil
import time

from milestone_common import ROOT,BASE,ASSETS,SOURCE,PLAN,dump,sha,setup,seed,private_clip,load_proxy,table_images
torch=setup()
private_clip()
from run_frozen import MyBackboneFinetuning
from vilt.modules.vilt_module import ViLTransformerSS

start=time.monotonic();out=ROOT/'scoring';out.mkdir(exist_ok=False)
verify=ROOT/'verification_inputs';verify.mkdir()
for source,name in [(SOURCE/'scoring/scoring_tensors.pt','original_scoring_tensors.pt'),
    (SOURCE/'scoring/text_gallery.json','text_gallery.json'),
    (SOURCE.parent/'proxy_gallery_20261113/grpo_val_cache_rebuilt_last50.pt','proxy_gallery.pt')]:
    shutil.copy2(source,verify/name)
gallery=json.loads((verify/'text_gallery.json').read_text())
old=torch.load(verify/'original_scoring_tensors.pt',map_location='cpu',weights_only=True)
text_features=old['text_features'];text_gpu=text_features.cuda()
seed(0);scorer=load_proxy()
path=ASSETS/'ucm_main_domain_tool.ckpt';assert sha(path)==PLAN['utility_model_sha256']
saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
config=dict(saved['hyper_parameters']['config']);config.update(load_path='',test_only=False)
utility=ViLTransformerSS(config);utility.load_state_dict(saved['state_dict'],strict=True)
utility.eval().requires_grad_(False).cuda()
images={row:utility.preprocess(item['image']).unsqueeze(0).cuda() for row,item in table_images(PLAN['validation_rows']).items()}
records=[]
for policy in PLAN['evaluation']['policy_points']:
    assert json.loads((ROOT/'evaluation'/policy/'result.json').read_text())['ok']
    records += [json.loads(x) for x in (ROOT/'evaluation'/policy/'records.jsonl').read_text().splitlines()]
assert len(records)==330
scores=[];visuals=[];proxies=[];actual=[];proxy_sims=[]
for n,record in enumerate(records,1):
    assert sha(ROOT/record['state_file'])==record['state_file_sha256']
    h=torch.load(ROOT/record['state_file'],map_location='cpu',weights_only=True)
    captured=[];encode=scorer.pl.model.encode_image
    def capture(*args,**kwargs):
        feat=encode(*args,**kwargs);captured.append(feat.detach().clone());return feat
    scorer.pl.model.encode_image=capture
    try:rank,reward=scorer.score(h.float().tolist(),record['row'])
    finally:scorer.pl.model.encode_image=encode
    assert len(captured)==1
    proxy=captured[0]/captured[0].norm(dim=1,keepdim=True)
    with torch.no_grad():
        ps=(proxy@scorer.image_features.cuda().T).cpu().flatten()
        visual=utility.img_embeds(images[record['row']],agent_hidden=h.unsqueeze(0).cuda())['image_feats']
        sims=(visual@text_gpu.T).cpu().flatten()
    assert all(torch.isfinite(x).all() for x in [h,proxy,visual,ps,sims])
    positive=gallery['positive_gallery_columns'][str(record['row'])]
    negatives=[j for j in range(gallery['unique_val_caption_count']) if j not in positive]
    best=float(sims[positive].max());negative=float(sims[negatives].max())
    actual_rank=int((sims[:gallery['unique_val_caption_count']]>best).sum())
    entry={**record,'proxy_rank0':rank,'r_retrieval':reward,'r_format':.2 if record['quality_usable'] else 0.,
        'r_total':reward+(.2 if record['quality_usable'] else 0.),
        'utility_rank0':actual_rank,'utility_rr':1/(1+actual_rank),'utility_margin':best-negative}
    for label,cols in gallery['quantity_text_columns'][str(record['row'])].items():
        entry['E1_margin_'+label]=float(sims[cols[0]]-sims[cols[1]])
    scores.append(entry);visuals.append(visual.cpu());proxies.append(proxy.cpu());actual.append(sims);proxy_sims.append(ps)
    dump(ROOT/'progress.json',{'phase':'scoring','completed':n,'total':330})
    print('SCORED',n,'/330',entry['id'],flush=True)
dump(out/'scores.json',scores)
torch.save({'record_ids':[x['id'] for x in scores],'text_features':text_features,'image_features':torch.cat(visuals),
    'proxy_features':torch.cat(proxies),'actual_similarities':torch.stack(actual),'proxy_similarities':torch.stack(proxy_sims)},
    out/'scoring_tensors.pt')
dump(out/'verification.json',{'ok':True,'records':330,'text_features_reused_bitwise':True,
    'strict_checkpoint_load':True,'frozen_eval_no_grad':True,'verification_input_sha256':{p.name:sha(p) for p in verify.iterdir()},
    'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
print('SCORING_COMPLETE',flush=True)
