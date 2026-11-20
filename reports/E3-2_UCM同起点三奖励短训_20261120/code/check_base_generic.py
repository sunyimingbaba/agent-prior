"""Repeat all baseline generic queries with a new process, and compare original tokens/states to a fingerprinted reload."""
import torch
from shared import ROOT,read,dump,sha
import deploy
def main():
    mirror=ROOT/'scratch/replay_base_generic';mirror.mkdir(parents=True,exist_ok=True)
    for name in ('inputs','review'):
        if not (mirror/name).exists():(mirror/name).symlink_to(ROOT/name,target_is_directory=True)
    deploy.ROOT=mirror
    deploy.run('base','generic')
    old=ROOT/'eval/base/generic';new=mirror/'eval/base/generic';compared=[]
    fingerprint=read(new/'loaded_weights.json')
    assert fingerprint['loaded_params']==read(ROOT/'policy_old/update.json')['initial_params']
    for r in read(ROOT/'inputs/validation.json'):
        p=__import__('pathlib').Path(r['image']).stem+'_upright.json'
        a,b=read(old/p),read(new/p)
        for key in ['query','traces','image','rendered_sha256','model']:assert a[key]==b[key]
        av=torch.load((old/p).with_suffix('.pt'),weights_only=True,map_location='cpu')
        bv=torch.load((new/p).with_suffix('.pt'),weights_only=True,map_location='cpu');assert torch.equal(av,bv)
        compared.append(dict(image=r['image'],hidden=a['traces'][0]['hidden']))
    assert len(compared)==210
    dump(ROOT/'review/base_generic_replay.json',dict(reloaded=fingerprint,identical_trajectories=len(compared),identical_hidden=len(compared),cases=compared))
if __name__=='__main__':main()
