"""Check old and new true greedy priors with exactly the same frozen CPU image reader."""
from pathlib import Path
import sys,torch,json
ROOT=Path(__file__).resolve().parents[1];HIST=ROOT.parent/'multicap_rfpo_20261124';OLD=ROOT.parent/'soft_guard_20261125'
sys.path.insert(0,str(HIST/'code'))
from runtime import table,rgb,setup,read,dump
from readouts import load_model
@torch.inference_mode()
def main():
    setup();torch.set_num_threads(2);m=load_model(False);t={x:table(x) for x in ('val','test')}
    rows={r['image']:r for r in read(HIST/'inputs/panel.json')};bank=torch.load(HIST/'records/text.pt',weights_only=True);out=[]
    for r in read(ROOT/'results/diagnosis.json')['five']:
        s,n=r['seed'],r['image'];row=rows[n];pix=m.preprocess(rgb(t[row['split']],row))[None]
        ix=bank['maps']['primary'][n][r['wording']];d=bank['hard'][ix[0]].double()-bank['hard'][ix[1]].double()
        raw=[HIST/'greedy'/(s+'_deployed')/'primary'/(n+'.pt'),OLD/'mined_policy'/(s+'_selected_joint')/'evaluate/primary'/(n+'.pt')]
        vals=[]
        for p in raw:
            rec=torch.load(p,weights_only=True);assert rec['rgb']==row['rgb_sha256']
            v=m.img_embeds(pix,agent_hidden=rec['prior'].float()[None])['image_feats'][0]
            vals.append(float(v.double()@d))
        out.append(dict(image=n,seed=s,wording=r['wording'],paired_cpu=vals,archived=r['margin'],reproduced=vals[0]>0>=vals[1]))
    dump(ROOT/'results/five_cpu.json',out)
if __name__=='__main__':main()
