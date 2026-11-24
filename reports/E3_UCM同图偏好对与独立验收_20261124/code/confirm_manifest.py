import json,io,hashlib,sys,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REP=ROOT.parent;HIST=REP/'E3-2_UCM有效图组定向短训_20261124'
sys.path.insert(0,str(HIST/'code'))
from fresh_panel import OLD,ARROW,read,blocked,samples,picture,BANK,D,SCENE,E as original_E
E={**original_E,
'freeway_axis':('The main highway extends across the image from side to side.','The main highway extends through the image from top to bottom.'),
'freeway_color':('The highway has pale paving.','The highway has dark paving.'),
'berths':('Boats occupy nearly every docking space.','Quite a few docking spaces are unoccupied.'),
'harbor_rows':('The long lines of boats go from left to right in this harbor.','The long lines of boats go from top to bottom in this harbor.'),
'crosswalk':('Striped pedestrian crossings mark this intersection.','This intersection has no striped pedestrian crossings.'),
'bridge_axis':('The upper deck extends sideways through the frame.','The upper deck extends vertically through the frame.'),
'bridge_slope':('The overpass points from bottom left to top right.','The overpass points from top left to bottom right.'),
'runway_number':('Large digits have been painted on the runway.','There are no large painted digits on the runway.'),
'runway_bars':('The runway has broad rectangular white markings.','The runway lacks broad rectangular white markings.')}
PAIRS=[
('field_axis',27,66),('field_axis',79,22),('field_color',33,74),('field_color',51,68),
('infield',250,259),('infield',277,248),('infield',242,227),('infield',216,225),('infield',223,241),('infield',231,264),('infield',217,270),
('surf',363,376),('surf',364,377),('surf',370,375),('surf',357,350),('water_side',311,320),
('rooftops',453,466),('rooftops',446,408),('rooftops',469,423),
('street_axis',673,677),('street_axis',652,643),('street_axis',644,666),
('freeway_axis',850,852),('freeway_axis',854,857),('freeway_axis',875,847),('freeway_axis',861,844),('freeway_color',834,870),('freeway_color',873,864),
('berths',1058,1041),('berths',1062,1040),('berths',1068,1035),('berths',1049,1071),('harbor_rows',1059,1010),
('crosswalk',1180,1145),
('street_axis',1243,1279),('street_axis',1253,1236),('rooftops',1246,1260),('rooftops',1244,1265),
('mobile_axis',1364,1317),('mobile_axis',1362,1310),('mobile_axis',1359,1361),
('bridge_axis',1434,1412),('bridge_axis',1450,1415),('bridge_axis',1403,1447),('bridge_axis',1404,1411),('bridge_axis',1421,1453),('bridge_axis',1444,1478),
('bridge_slope',1413,1465),('bridge_slope',1460,1430),
('parking_fill',1522,1536),('parking_fill',1539,1504),('parking_ground',1570,1542),('parking_ground',1509,1556),('parking_axis',1512,1524),
('river_axis',1657,1610),('river_axis',1665,1676),('river_axis',1643,1640),('river_bank',1625,1667),('river_bank',1627,1678),('river_bank',1619,1674),
('runway_number',1727,1779),('runway_number',1775,1776),('runway_number',1751,1732),('runway_number',1761,1764),('runway_bars',1724,1716),('runway_bars',1749,1758),('runway_bars',1756,1770),
('house_light',1837,1874),('house_light',1816,1864),('house_pool',1866,1849),('house_pool',1820,1846),
('court_surround',2017,2045),('court_surround',2061,2011),('court_surround',2046,2024),('court_surround',2068,2038),('court_surround',2019,2037)]
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    t,idx=samples();ban=blocked()|{r["image"] for r in read(HIST/"inputs/panel.json")}|{n for p in read(HIST/"inputs/lock.json")["removed"] for n in p["images"]};seen=sorted(ban & idx.keys());nprior=len(seen)
    reserved=set();rows=[];skipped=[];cache={};cv2.setNumThreads(1);cv2.setRNGSeed(20261124)
    sift=cv2.SIFT_create(nfeatures=200);bf=cv2.BFMatcher()
    for dataset in ('ucm','rsitmd'):
        reserved.update(v['rgb_sha256'] for v in read(REP/'E3-2_UCM与RSITMD独立复验_20261124/inputs'/(dataset+'.json')))
    def get(name):
        if name not in cache:
            im=picture(t,idx[name]);g=np.asarray(im.convert('L').resize((192,192),3));kp,ds=sift.detectAndCompute(g,None)
            cache[name]=hashlib.sha256(im.tobytes()).hexdigest(),np.float32([p.pt for p in kp]),ds
        return cache[name]
    def overlap(a,b):
        h,x,v=get(a);g,y,w=get(b)
        if h==g:return True
        if v is None or w is None or min(len(v),len(w))<12:return False
        matches=[m for m,n in bf.knnMatch(v,w,k=2) if m.distance<.72*n.distance]
        u={tuple(y[z.trainIdx].round().astype(int)):z for z in sorted(matches,key=lambda m:-m.distance)}
        if len(u)<12:return False
        i=np.float32([x[z.queryIdx] for z in u.values()]);j=np.float32([y[z.trainIdx] for z in u.values()])
        tf,mask=cv2.estimateAffinePartial2D(i,j,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=3000)
        if tf is None:return False
        hit=mask.ravel().astype(bool)
        return hit.sum()>=12 and hit.mean()>=.6 and min(cv2.contourArea(cv2.convexHull(z[hit])) for z in (i,j))>=.1*192**2
    old=read(OLD/'inputs/panel.json');proposed=[]
    for attr,a,b in PAIRS:
        ids=[f'{a}.tif',f'{b}.tif'];scene=(a-1)//100;assert scene==(b-1)//100 and not (set(ids)&(set(proposed)|ban))
        proposed+=ids;over=next(((n,s) for n in ids for s in seen if overlap(n,s)),None)
        if over or overlap(*ids) or any(get(n)[0] in reserved for n in ids):
            skipped.append(dict(images=ids,overlap=over or ids));continue
        sample=next(r for r in old if r['scene_id']==scene);seen+=ids
        for side,n in enumerate(ids):
            j=idx[n];caps=t['caption'][j].as_py();assert t['split'][j].as_py()=='train'
            rows.append(dict(image=n,row=j,rgb_sha256=get(n)[0],split='train',pair=f'{a}-{b}',side=side,scene_id=scene,scene=SCENE[scene],attribute=attr,scene_query=sample['scene_query'],orig_query=caps[0],captions=caps,
                **{k:[p[side],p[1-side]] for k,p in zip('CDE',[BANK[attr][2],D[attr],E[attr]],strict=True)}))
    dest=ROOT/'confirm/inputs';dest.mkdir(exist_ok=True,parents=True);p=dest/'panel.json';p.write_text(json.dumps(rows,indent=2))
    (dest/'lock.json').write_text(json.dumps(dict(panel=digest(p),train_arrow=digest(ARROW),protocol=digest(ROOT/'验收与比较约定.md'),excluded_prior=nprior,proposed_pairs=len(PAIRS),removed=skipped,pairs=len(rows)//2),indent=2))
if __name__=='__main__':main()

