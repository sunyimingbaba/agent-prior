"""Compare final actual greedy tool/action sequences and final hidden priors against the common starting model."""
import io
import sys
import tarfile
import numpy as np
import torch
from compare import OUT,read,dump,sha
from single_scores import RFPO

def records(path,legacy):
    keep={}
    with tarfile.open(path,'r|gz') as tf:
        for m in tf:
            bits=m.name.split('/')
            if len(bits)<2 or not m.isfile() or not m.name.endswith('.tif.pt') or bits[-2] not in ('primary','official'):continue
            z=torch.load(io.BytesIO(tf.extractfile(m).read()),weights_only=True)
            key=(bits[-2],z['image'])
            keep[key]=(z['rounds' if legacy else 'actions'],z['tools' if legacy else 'calls'],z['prior'],z['query'],z['rgb'])
    assert len(keep)==336
    return keep

def main(stage):
    start='R0' if stage=='exact_policy' else 'deployed'
    index=read(OUT/'archives'/(stage+'_index.json'));report={}
    for s in 'ab':
        f=RFPO/'archives'/(s+'_'+start+'.tar.gz')
        anchors=records(f,True)
        for arm,z in index.items():
            if not arm.startswith(s+'_'):continue
            arc=OUT/'archives'/z['archive'];assert sha(arc)==z['archive_sha'];new=records(arc,False)
            assert set(new)==set(anchors);report[arm]={}
            for sub in ('primary','official'):
                values=[]
                for key,item in new.items():
                    if key[0]!=sub:continue
                    actions,tools,h,q,rgb=item;a,t,h0,q0,rgb0=anchors[key];assert q==q0 and rgb==rgb0
                    rel=float(torch.linalg.vector_norm(h.float()-h0.float())/torch.linalg.vector_norm(h0.float()))
                    values.append((a!=actions,a[0]!=actions[0],t!=tools,rel))
                y=np.array(values)
                report[arm][sub]=dict(images=len(y),changed_full_action=int(y[:,0].sum()),changed_first_action=int(y[:,1].sum()),
                    changed_tool_feedback=int(y[:,2].sum()),median_relative_hidden_displacement=float(np.median(y[:,3])),
                    median_displacement_same_action=float(np.median(y[y[:,0]==0,3])) if np.any(y[:,0]==0) else None)
    dump(OUT/'results'/(stage+'_behavior.json'),report)
if __name__=='__main__':main(sys.argv[1])
