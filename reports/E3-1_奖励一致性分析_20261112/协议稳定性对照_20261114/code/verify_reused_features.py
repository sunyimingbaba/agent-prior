import argparse
import hashlib
import json
from pathlib import Path
import tarfile

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
names={'features/tensors.pt':'features/tensors.pt','features/gallery.json':'features/gallery.json',
       'scoring/tensors.pt':'baseline_evidence/scoring_tensors.pt','scoring/scores.json':'baseline_evidence/scores.json',
       'analysis.json':'baseline_evidence/analysis.json','plan.json':'baseline_evidence/plan.json'}
verified={}
with tarfile.open(r/'baseline_evidence/quantity_counterfactual_evidence_20261114.tar.gz') as t:
    for name,local in names.items():
        original=t.extractfile(name).read();copied=(r/local).read_bytes();assert copied==original,local
        verified[local]={'bytes':len(copied),'sha256':hashlib.sha256(copied).hexdigest()}
a.out.write_text(json.dumps({'ok':True,'all_feature_gallery_and_baseline_copies_bitwise_equal':True,'files':verified},indent=2)+'\n',encoding='utf-8')
print('REUSED_FEATURES_AND_BASELINE_FILES_BITWISE_VERIFIED',len(verified))
