from shared import ROOT, read, dump
a=read(ROOT/'policy_a_off/steps.json')[:3]
b=read(ROOT/'policy_a_on/steps.json')[:3]
summary=[]
for x,y in zip(a,b,strict=True):
    same={k:x[k]==y[k] for k in ('active_params','caption_and_hard_margins','traces','format_score','advantages')}
    summary.append(dict(step=x['step'],evidence=same,off_formats=x['format_score'],on_formats=y['format_score']))
dump(ROOT/'results/prefix_replay.json',summary)
# Only steps with no differential reward require equality after the update.
for item in summary:
    if len(set(item['off_formats']))>1:break
    assert all(item['evidence'].values()),item
print('passed paired initial updates')
