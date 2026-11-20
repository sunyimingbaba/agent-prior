"""Freeze the physically reversible training questions and the matched protocol."""
from collections import Counter
from shared import ROOT,read,dump,sha
folder=ROOT/'inputs'
reasons={'house_surround':'Enclosure and land cover survive a 180-degree rotation.',
         'diagonal_bridge':'Over/under describes vertical stacking, unchanged by image rotation.',
         'overpass_order':'Bridge layer order is independent of screen coordinates.',
         'river_road':'The original caption for 1668.tif describes roads on both banks. Exclude the entire reciprocal pair to avoid ambiguous road reference.'}
rows=[];audit=[]
for r in read(folder/'candidates.json'):
    reason=reasons.get(r['pair'])
    audit.append(dict(image=r['image'],pair=r['pair'],decision='exclude' if reason else 'retain',reason=reason or 'Screen-coordinate order reverses on the rotated real photograph. Original first caption is orientation invariant.'))
    if not reason:rows.append(r)
assert len(rows)==16 and set(Counter(r['pair'] for r in rows).values())=={2}
for k in set(z['pair'] for z in rows):
    a,b=[z for z in rows if z['pair']==k]
    for w in ('A','B'):assert a[w]==b[w][::-1]
val=read(folder/'validation.json')
assert not ({r['image'] for r in val}&{r['image'] for r in rows} or {r['rgb_sha256'] for r in val}&{r['rgb_sha256'] for r in rows})
dump(folder/'train.json',rows)
dump(ROOT/'review/rotation_screen.json',audit)
arms=[s+'_'+arm for s in ('a','b') for arm in ('single','separate','joint')]
dump(folder/'protocol.json',dict(arms=arms,schedules={'a':284000,'b':784000},steps=[8,16],n_source=16,group_size=8,
  common='Start each arm from the same previously deployed UCM LoRA. Same 16 train photos and order, 8 sampled trajectories per photo, historical scene-only cue and native tool loop. Deterministic PyTorch, optimizer AdamW lr 5e-6, trajectory log-probability sum including emitted EOS, grad clip 1. No format reward.',
  single='8 independent upright trajectories. Half caption midrank and half same-scene direction midrank, standardized within eight. Replicates the established single-view objective on this training subset.',
  separate='Four upright and four 180-degree rotated candidates. Both use the same orientation-invariant original train caption and negative pool; the correct direction caption reverses on rotation. For both terms compute per-view midranks, then per-view sample standardized advantages.',
  joint='Same exposures, random seeds, rank scales, gradient weight and normalization as separate. In each of four independently sampled candidate pairs replace the directional input on both views by min(correct signed upright margin, correct signed rotated margin). Pair by fixed sample slot before scoring. Both views must have a positive margin for their minimum to be positive.',
  seed_pairing='slot 0-3 upright, slot 4-7 rotated in paired arms, all eight upright in single. Paired arms must have identical pre-update paths and scores up to the first effective reward difference.',
  primary='Step 16 joint vs separate full native six-recall mR and number of both-correct orientation questions among 32 (16 val photos x two wordings; sources paired), and comparison to the historical starting weights. No checkpoint selection.',
  promotion='For each seed final joint must reach historical native mR and both-view count, surpass separate on both-view count and not lose mR versus separate. Report each endpoint even on failure.',
  secondary='Whole validation generic query, E1 quantity A/B, attribute and real relation pair transfer, upright/rotated pair accuracy and weakest signed margin, greedy EOS/tool use. Middle checkpoint = step 8. Training-set deployment on the exact A wording and unused B wording, and training candidate diagnostics, are descriptive. No formal test or downstream retraining.',
  inference='Fixed native greedy decoding. Same frozen readout. Fixed-gallery source-paired percentile bootstrap with 20000 resamples, source clustering over wordings and views; equal weight across two seeds.'))
files=['train.json','candidates.json','validation.json','detail.json','retrieval_text.pt','e1_extension.json','protocol.json']
dump(folder/'lock.json',{name:sha(folder/name) for name in files})
