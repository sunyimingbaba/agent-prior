from confirm_manifest import ROOT,REP,HIST,OLD,ARROW,read,blocked,samples,picture,BANK,D,SCENE,E
import numpy as np,cv2,json,hashlib
# Extra attributes describe visible civilian details in this independent panel.
extra={
'community_pool':[
('A swimming pool is visible in the residential area.','No swimming pool is visible in the residential area.'),
('There is a pool among the homes.','There is no visible pool among the homes.'),
('The neighborhood contains a visible swimming pool.','The neighborhood shows no swimming pool.'),
('A swimming pool appears between the residential buildings.','There is no swimming pool in sight between the residential buildings.'),
('One can see a swimming pool on this residential block.','No swimming pool can be seen on this residential block.')],
'bridge_curve':[
('The upper roadway curves.','The upper roadway is straight.'),('The overpass deck bends across the image.','The overpass deck runs straight across the image.'),
('The elevated road follows a curved path.','The elevated road follows a straight path.'),('A bend is visible in the roadway on top.','The roadway on top has a linear alignment.'),
('The direction of the upper road changes across the scene.','The upper road keeps a straight course across the scene.')],
'court_pool':[
('A swimming pool is visible beside the tennis courts.','There is no visible swimming pool beside the tennis courts.'),
('A pool appears next to the tennis facility.','There is no pool in sight next to the tennis facility.'),
('The grounds by the tennis courts include a swimming pool.','The grounds by the tennis courts show no swimming pool.'),
('A swimming pool can be seen near the marked courts.','No swimming pool can be seen near the marked courts.'),
('There is a pool near the tennis playing area.','The tennis playing area has no visible pool nearby.')],
'main_hue':[
('The main house has an orange or red roof.','The main house has a gray roof.'),
('The largest residential roof has a reddish tone.','The largest residential roof has a gray tone.'),
('Warm red or orange roofing covers the principal home.','Gray roofing covers the principal home.'),
('The roof of the main home looks red or orange.','The roof of the main home looks gray.'),
('The primary house has a red-toned rooftop.','The primary house has a gray-toned rooftop.')]}
BANK=dict(BANK);D=dict(D);E=dict(E)
for key,fs in extra.items():BANK[key]=fs[:3];D[key]=fs[3];E[key]=fs[4]
PAIRS=[
('field_axis',31,24),('field_axis',49,19),('field_color',32,3),('field_color',18,45),
('infield',222,234),
('surf',371,374),('surf',369,359),('surf',372,352),('surf',368,353),('water_side',309,316),
('rooftops',460,417),
('community_pool',622,674),('community_pool',623,679),('community_pool',624,660),('community_pool',617,664),('community_pool',618,649),('community_pool',625,675),
('street_axis',654,606),('street_axis',663,619),('street_axis',642,608),
('freeway_axis',855,858),('freeway_axis',879,846),('freeway_axis',849,843),('freeway_axis',874,833),
('freeway_color',877,869),('freeway_color',835,871),('freeway_color',825,842),('freeway_color',826,838),
('berths',1069,1038),('berths',1053,1011),('harbor_rows',1065,1046),
('crosswalk',1137,1146),
('community_pool',1252,1275),('community_pool',1242,1206),('community_pool',1255,1245),('community_pool',1202,1274),('community_pool',1263,1238),
('rooftops',1239,1262),('rooftops',1234,1261),
('mobile_axis',1365,1316),('mobile_axis',1328,1308),
('bridge_axis',1436,1427),('bridge_axis',1424,1473),('bridge_axis',1462,1474),
('bridge_curve',1433,1439),('bridge_curve',1466,1459),('bridge_curve',1408,1414),('bridge_curve',1468,1410),('bridge_curve',1418,1477),
('parking_fill',1529,1526),('parking_fill',1543,1566),('parking_fill',1521,1519),('parking_ground',1511,1548),('parking_ground',1507,1561),('parking_ground',1505,1555),
('river_axis',1606,1647),('river_axis',1641,1648),('river_bank',1623,1666),('river_bank',1620,1679),
('runway_number',1750,1774),('runway_bars',1721,1772),('runway_bars',1769,1771),('runway_bars',1742,1704),('runway_bars',1702,1752),
('main_hue',1807,1855),('main_hue',1803,1847),
('house_light',1842,1838),('house_light',1825,1862),
('court_pool',2079,2042),('court_pool',2080,2048),('court_pool',2018,2077),('court_pool',2069,2047),
('court_surround',2020,2058)]
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    t,idx=samples();ban=blocked()|{r["image"] for r in read(HIST/"inputs/panel.json")}|{n for p in read(HIST/"inputs/lock.json")["removed"] for n in p["images"]};ban.update(r['image'] for r in read(ROOT/'confirm/inputs/panel.json'));ban.update(n for drop in read(ROOT/'confirm/inputs/lock.json')['removed'] for n in drop['images']);seen=sorted(ban & idx.keys());nprior=len(seen)
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
                **{k:[p[side],p[1-side]] for k,p in zip('ABCDE',list(BANK[attr])+[D[attr],E[attr]],strict=True)}))
    dest=ROOT/'reward_confirm/inputs';dest.mkdir(exist_ok=True,parents=True);p=dest/'panel.json';p.write_text(json.dumps(rows,indent=2))
    (dest/'lock.json').write_text(json.dumps(dict(panel=digest(p),train_arrow=digest(ARROW),protocol=digest(ROOT/'reward_confirm/实验口径.md'),excluded_prior=nprior,proposed_pairs=len(PAIRS),removed=skipped,pairs=len(rows)//2),indent=2))
if __name__=='__main__':main()


