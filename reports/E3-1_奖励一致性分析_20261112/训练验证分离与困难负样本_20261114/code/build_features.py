import json
import shutil
import time

from milestone_common import ROOT,PLAN,SOURCE,setup,seed,dump,sha
from split_reward import load_final_model
torch=setup();seed(0);start=time.monotonic()
out=ROOT/'features';out.mkdir(exist_ok=False)
model,clip=load_final_model()
gallery=json.loads((ROOT/'inputs/training_text_gallery.json').read_text())
tokens=clip.tokenize(gallery['text_order'],context_length=61)
features=[]
with torch.no_grad():
    for i in range(0,len(tokens),16):features.append(model.txt_embeds(tokens[i:i+16].cuda())['text_feats'].cpu())
training=torch.cat(features);assert training.shape==(1437,768) and torch.isfinite(training).all()
assert torch.allclose(training.norm(dim=1),torch.ones(len(training)),atol=1e-5,rtol=0)
torch.save({'text_features':training,'text_order':gallery['text_order']},out/'train.pt')
original=torch.load(SOURCE/'scoring/scoring_tensors.pt',map_location='cpu',weights_only=True)
validation_gallery=json.loads((SOURCE/'scoring/text_gallery.json').read_text())
torch.save({'text_features':original['text_features']},out/'val.pt');dump(out/'val_gallery.json',validation_gallery)
lookup={t:i for i,t in enumerate(validation_gallery['text_order'][:391])}
metadata=json.loads((ROOT/'inputs/val_metadata.json').read_text())
assert set(lookup)=={c for x in metadata['rows'] for c in x['captions']}
dump(out/'val_positive_columns.json',{x['row']:sorted({lookup[c] for c in x['captions']}) for x in metadata['rows']})
with torch.no_grad():
    anchors=model.txt_embeds(clip.tokenize(validation_gallery['text_order'][:16],context_length=61).cuda())['text_feats'].cpu()
error=float((anchors-original['text_features'][:16]).abs().max());assert error<1e-6
dump(out/'result.json',{'ok':True,'training_captions':len(training),'validation_captions_for_ranking':391,
     'validation_text_feature_reused_bitwise':True,'validation_16_feature_anchor_max_error':error,
     'files_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'elapsed_seconds':time.monotonic()-start})
print('SEPARATED_TEXT_FEATURES_BUILT',flush=True)
