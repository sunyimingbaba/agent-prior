"""Verify each evaluated input against images decoded from Arrow and the exact historical baseline."""
import argparse
from pathlib import Path
import torch
from shared import ROOT, read, sha, dump
from analyze import study_states

def check(previous):
    pixels = read(ROOT/'results/source_pixels.json')
    cases = {part:read(ROOT/'inputs'/(name+'.json')) for part,name in [
        ('native','validation'),('generic','validation'),('detail','detail'),('trainpanel','train'),('quarter','quarter_questions')]}
    historical = {}
    for part in ('native','generic','detail'):
        old, now = previous/'eval/base_00', ROOT/'eval/base_00'
        before, after = (read(p/(part+'.json')) for p in (old,now))
        assert before['active']==after['active'] and before['weights']==after['weights']
        assert before['records']==after['records']
        old_values, new_values = [torch.load(p/(part+'.pt'), weights_only=True,map_location='cpu') for p in (old,now)]
        assert old_values.keys()==new_values.keys()
        for key in old_values:
            assert torch.equal(old_values[key],new_values[key]),(part,key)
        historical[part] = dict(identical_records=len(after['records']), prior_metadata=sha(old/(part+'.json')),
                                current_metadata=sha(now/(part+'.json')), exact_tensors=list(old_values))
    sizes={}
    for folder in ['eval','quarter_eval']:
        counts={}
        for state in study_states():
            if folder=='quarter_eval' and state.endswith('_08'):continue
            path=ROOT/folder/state
            proof=read(path/'complete.json')
            expected=read(previous/'eval/base_00/complete.json')['active'] if state=='base_00' else read(ROOT/('policy_'+state.rsplit('_',1)[0])/('step_'+state.rsplit('_',1)[1])/'update.json')['final_params']
            assert proof['active']==expected
            nstate=0
            for part,digest in proof['parts'].items():
                meta=read(path/(part+'.json'))
                assert sha(path/(part+'.json'))==digest and meta['active']==expected
                bank=pixels['train' if part=='trainpanel' else 'val']
                views=('cw90','ccw90') if part=='quarter' else ('upright','rotated') if part in ('detail','trainpanel') else ('upright',)
                ordered=[(item,v) for item in cases[part] for v in views]
                for rec,(item,view) in zip(meta['records'],ordered,strict=True):
                    prompt=item['orig_query'] if part=='native' else 'An aerial photograph.' if part=='generic' else item['scene_query']
                    assert (rec['image'],rec['row'],rec['view'],rec['query'],rec['seed'],rec['rendered_sha'])==(
                            item['image'],item['row'],view,prompt,394000+10*item['row'],bank[item['image']][view])
                nstate+=len(ordered)
            counts[state]=nstate
        sizes[folder]=counts
    assert sum(sizes['eval'].values())==6292 and sum(sizes['quarter_eval'].values())==224
    dump(ROOT/'results/deployment_checks.json',dict(previous_start_proof=sha(previous/'eval/base_00/complete.json'),identical_start=historical,verified_records=sizes,
                                                     pixel_source_sha=sha(ROOT/'results/source_pixels.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('previous',type=Path);check(p.parse_args().previous)
