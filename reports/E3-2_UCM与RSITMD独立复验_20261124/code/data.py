from shared import ROOT,ASSETS,read,sha
import pyarrow as pa

DOMAINS=('ucm','rsitmd')
PREVIOUS=ROOT.parent/'second_answer_5090_20261124'
STATES={'R0':('','R0'),'a_R2':('','R2'),'b_R2':('repeat','R2'),
        'a_R1':('','R1'),'b_R1':('repeat','R1'),
        'a_first':('','first_R2'),'b_first':('repeat','first_R2')}

def dataset(name):
    p=(ASSETS/'ucm' if name=='ucm' else ROOT/'inputs')/f'{name}_caption_karpathy_label_test.arrow'
    proof=read(ROOT/'inputs/audit.json')
    assert sha(p)==proof['files'][name+'_test']
    manifest=ROOT/'inputs'/f'{name}.json'
    assert sha(manifest)==proof['regions'][name]['manifest']
    return pa.ipc.open_file(pa.memory_map(str(p))).read_all(),read(manifest)
