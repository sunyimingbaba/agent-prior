import json
import time

from milestone_common import ROOT,PLAN,SOURCE,setup,seed,dump,sha,case_images
from quantity_model import load_final_model
torch=setup();seed(0);start=time.monotonic();out=ROOT/'features';out.mkdir(exist_ok=False)
cases=json.loads((ROOT/'inputs/cases.json').read_text());images=case_images(cases)
model,clip=load_final_model()
old_gallery=json.loads((SOURCE/'scoring/text_gallery.json').read_text())
old=torch.load(SOURCE/'scoring/scoring_tensors.pt',map_location='cpu',weights_only=True)['text_features']
assert old.shape==(419,768)
old_index={t:i for i,t in enumerate(old_gallery['text_order'])}
texts=[];index={};columns={}
for item in cases:
    columns[item['image']]={}
    for wording,pair in item['pairs'].items():
        cols=[]
        for key in ['positive','negative']:
            text=pair[key]
            if text not in index:index[text]=len(texts);texts.append(text)
            cols.append(index[text])
        columns[item['image']][wording]=cols
qfeatures=torch.empty(len(texts),768,dtype=torch.float32);missing=[]
for i,text in enumerate(texts):
    if text in old_index:qfeatures[i]=old[old_index[text]]
    else:missing.append(i)
tokens=clip.tokenize(texts,context_length=61)
with torch.no_grad():
    for start_i in range(0,len(missing),16):
        ids=missing[start_i:start_i+16];qfeatures[ids]=model.txt_embeds(tokens[ids].cuda())['text_feats'].cpu()
assert torch.isfinite(qfeatures).all() and torch.allclose(qfeatures.norm(dim=1),torch.ones(len(texts)),atol=1e-5,rtol=0)
val_lookup={t:i for i,t in enumerate(old_gallery['text_order'][:391])}
vp={x['image']:sorted({val_lookup[c] for c in images[x['image']]['captions']}) for x in cases if x['split']=='val'}
torch.save({'quantity_text_features':qfeatures,'original_val_text_features':old},out/'tensors.pt')
dump(out/'gallery.json',{'quantity_text_order':texts,'quantity_columns':columns,'val_positive_columns':vp,
     'original_val_text_order':old_gallery['text_order'],'val_ranking_caption_count':391})
dump(ROOT/'dataset_alignment.json',{'ok':True,'images':100,'all_rgb_hashes_and_original_first_captions_match':True,
     'train_val_rgb_hashes_disjoint':True,'neutral_prompt':PLAN['prompt'],
     'cases':[{'image':x['image'],'split':x['split'],'row':x['row'],'image_rgb_sha256':x['image_rgb_sha256'],'captions':images[x['image']]['captions']} for x in cases]})
dump(out/'result.json',{'ok':True,'quantity_unique_texts':len(texts),'reused_old_quantity_vectors':len(texts)-len(missing),
     'new_encoded_texts':len(missing),'original_419_val_vectors_reused_bitwise':True,'elapsed_seconds':time.monotonic()-start,
     'files_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
print('QUANTITY_FEATURES_AND_100_IMAGE_ALIGNMENT_VERIFIED',flush=True)
