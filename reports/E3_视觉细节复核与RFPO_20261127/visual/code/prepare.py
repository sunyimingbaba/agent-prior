from collections import defaultdict
from pathlib import Path
import json,hashlib
import torch

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT.parents[1]
def r(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cal=REPORTS/'E3_困难奖励跨图跨措辞校准_20261127'
    pool=REPORTS/'E3_回填后双收益训练图筛选_20261127'
    test=cal/'inputs/test_panel.json'
    extra=cal/'followup_pool/inputs/rows.json'
    train=pool/'scoring/locks/training.json'
    libraries=[REPORTS/'E3_UCM同图偏好对与独立验收_20261124/guard/records/text.pt',
               REPORTS/'E3_UCM多数配文新图复核_20261125/records/text.pt']
    libraries.append(REPORTS/'E3_UCM同图偏好对与独立验收_20261124/reward_confirm/records/text.pt')
    banks={k:torch.load(p,map_location='cpu',weights_only=True) for k,p in zip(('v122','v148','external'),libraries)}
    rows=[];features={};groups=defaultdict(list)
    for panel,rs in [('train',r(train)['rows']),('main',r(test)),('external',r(extra))]:
        for row in rs:
            case={k:row[k] for k in ('image','row','rgb_sha256','side','pair','attribute','C','split','cohort','orig_query')}
            case['panel']=panel
            name=case['image']
            assert name not in features and not(set(row['C'])&set(row['D']+row['E']))
            bank=banks['external' if panel=='external' else row['cohort']]
            idx=[bank['texts'].index(t) for t in row['C']]
            if panel!='external':assert idx==bank['maps'][name]['C']
            features[name]=bank['hard'][idx].clone()
            assert features[name].shape==(2,768)
            case['options_side0']=case['C'] if case['side']==0 else case['C'][::-1]
            groups[row['cohort'],row['pair']].append(case['options_side0'])
            rows.append(case)
    assert all(all(x==opts[0] for x in opts) for opts in groups.values())
    assert len(rows)==336==len({x['rgb_sha256'] for x in rows})
    out=ROOT/'inputs'
    (out/'rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    torch.save(features,out/'c_vectors.pt')
    (out/'lock.json').write_text(json.dumps(dict(rows=sha(out/'rows.json'),vectors=sha(out/'c_vectors.pt'),
        historical={str(p.relative_to(REPORTS)):sha(p) for p in [train,test,extra,*libraries]}),indent=2)+'\n')
if __name__=='__main__':main()
