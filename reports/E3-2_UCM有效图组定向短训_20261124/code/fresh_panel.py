"""Lock natural opposite-side pairs before opening deployment scores. See panel_assets.py for exclusion sources."""
import sys
from panel_assets import ROOT,REP,OLD,ARROW,read,blocked
sys.path.insert(0,str(OLD/'code'))
from confirm_panel import D
from pairs import BANK,SCENE
import numpy as np,cv2,hashlib,json
from panel_assets import picture,samples
PAIRS=[
('field_axis',17,80),('field_axis',20,77),('field_curve',54,75),('field_curve',60,23),('field_color',34,41),
('infield',267,254),('infield',245,235),('infield',214,229),('infield',247,237),('infield',215,239),('infield',251,224),('infield',233,257),('infield',243,240),
('surf',366,379),('surf',302,378),('surf',361,314),('surf',367,327),('water_side',305,323),('water_side',303,322),
('rooftops',470,435),('rooftops',459,419),('roof_hue',638,665),('roof_hue',639,678),
('street_axis',650,659),('street_axis',658,680),('street_axis',653,676),
('street_axis',1207,1277),('street_axis',1240,1250),('street_axis',1241,1248),('street_axis',1251,1256),('rooftops',1233,1266),('rooftops',1232,1219),
('mobile_axis',1349,1311),('mobile_axis',1350,1315),('mobile_axis',1344,1312),('mobile_axis',1307,1358),
('parking_fill',1546,1520),('parking_fill',1547,1535),('parking_fill',1530,1559),('parking_ground',1577,1550),('parking_ground',1569,1551),('parking_axis',1525,1578),('parking_axis',1563,1575),('parking_axis',1538,1573),
('river_axis',1673,1608),('river_axis',1631,1616),('river_axis',1607,1622),('river_bank',1615,1671),('river_bank',1624,1632),('river_bank',1611,1672),
('house_light',1871,1875),('house_light',1839,1843),('house_pool',1802,1856),('house_pool',1817,1848),('house_pool',1801,1878),('house_pool',1824,1859),
('court_surround',2026,2044),('court_surround',2025,2043),('court_surround',2001,2034),('court_surround',2016,2062),('court_surround',2008,2041),('court_surround',2052,2029),('court_surround',2028,2060),('court_surround',2002,2055)]
E={
'field_axis':('Furrows are oriented across the width of this field.','Furrows are oriented along the height of this field.'),
'field_curve':('A bend is visible in the planted rows.','The planted rows form straight lines.'),
'field_color':('Most of this cultivated plot appears green.','Most of this cultivated plot appears brown.'),
'infield':('The area inside the baseball diamond has a grassy center.','The area inside the baseball diamond has a dirt center.'),
'surf':('A broad swath of white foam breaks along the beach.','Very little white foam breaks along the beach.'),
'water_side':('The sea occupies the left side of the shoreline view.','The sea occupies the right side of the shoreline view.'),
'rooftops':('Most of the rooftops are pale in this scene.','Most of the rooftops are dark in this scene.'),
'roof_hue':('A reddish color characterizes the residential roofs.','A gray color characterizes the residential roofs.'),
'street_axis':('The main street between the homes extends sideways in the image.','The main street between the homes extends vertically in the image.'),
'mobile_axis':('The road between the trailers crosses the picture from side to side.','The road between the trailers crosses the picture from top to bottom.'),
'parking_axis':('The lines of cars extend sideways across the lot.','The lines of cars extend vertically across the lot.'),
'parking_fill':('Almost every stall in these parking rows contains a vehicle.','Many stalls in these parking rows contain no vehicle.'),
'parking_ground':('Vehicles stand on a grassy parking area.','Vehicles stand on a paved parking area.'),
'river_axis':('The channel stretches sideways across the scene.','The channel stretches vertically through the scene.'),
'river_bank':('Green trees grow thickly beside this watercourse.','There are few green trees beside this watercourse.'),
'house_light':('The largest house has pale-colored roofing.','The largest house has dark-colored roofing.'),
'house_pool':('There is a swimming pool on the grounds around this home.','The grounds around this home have no visible swimming pool.'),
'court_surround':('A red surface borders the lined part of the tennis court.','A green surface borders the lined part of the tennis court.')}
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    t,idx=samples();ban=blocked();seen=sorted(ban & idx.keys());nprior=len(seen)
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
    dest=ROOT/'inputs';dest.mkdir(exist_ok=True);p=dest/'panel.json';p.write_text(json.dumps(rows,indent=2))
    (dest/'lock.json').write_text(json.dumps(dict(panel=digest(p),train_arrow=digest(ARROW),protocol=digest(ROOT/'实验口径.md'),excluded_prior=nprior,proposed_pairs=len(PAIRS),removed=skipped,pairs=len(rows)//2),indent=2))
if __name__=='__main__':main()
