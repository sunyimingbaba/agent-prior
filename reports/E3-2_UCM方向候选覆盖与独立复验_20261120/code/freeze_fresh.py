import csv
from shared import ROOT,read,dump,sha
def main():
    path=ROOT/'inputs';review=ROOT/'review'
    votes={r['image']:r for r in read(review/'pixel_votes.json')}
    with (review/'画面重叠候选.csv').open(encoding='utf-8-sig') as f:
        hits={r[k] for r in csv.DictReader(f) for k in ('image_a','image_b')}
    rows=read(path/'proposals.json')
    assert set(votes)=={r['image'] for r in rows}
    kept=[r for r in rows if votes[r['image']]['accept'] and votes[r['image']]['direction']==r['direction'] and r['image'] not in hits]
    assert len(kept)==len(set(r['rgb_sha256'] for r in kept))
    dump(path/'fresh.json',kept)
    dump(review/'adjudication.json',{r['image']:{'kept':r in kept,'coordinate_direction':r['direction'],'blind_direction':votes[r['image']]['direction'],'overlap_hit':r['image'] in hits} for r in rows})
if __name__=='__main__':main()
