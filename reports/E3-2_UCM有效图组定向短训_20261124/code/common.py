import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/private/AGSE_E3_1_rebuild_20261112/r3_after_tool_trainpanel_20261124')
os.environ['E3_PANEL_ROOT']=str(ROOT)
sys.path.insert(0,str(OLD/'code'))
from base import ASSETS,REBUILD,PREVIOUS,model_path,weight,sha,dump,read,rgb,setup,table,savept
def key(s):
    return hashlib.sha256(('e3-active-20261124|'+s).encode()).hexdigest()
def pool():
    rows={}
    for sub in ('','confirm'):
        folder=OLD/sub
        assert sha(folder/'inputs/panel.json')==read(folder/'inputs/lock.json')['panel']
        for c in read(folder/'inputs/panel.json'):
            assert c['image'] not in rows
            rows[c['image']]=dict(c,source=sub)
    return rows
ARMS=('regular_R3','selected_R3','focused_R3','focused_R2')

