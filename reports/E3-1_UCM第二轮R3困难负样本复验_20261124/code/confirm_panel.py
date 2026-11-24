"""Freeze the independent 50-percent comparison before reading scores on these training images."""
import hashlib,io,json
from pathlib import Path
import cv2
import numpy as np
import pyarrow as pa
from PIL import Image
from pairs import BANK,SCENE
from manifest import RUN
NEW=[
('field_axis',11,21),('field_color',78,76),('field_curve',61,7),('infield',219,208),('infield',246,212),('infield',256,260),('infield',249,205),
('surf',346,315),('surf',362,319),('water_side',310,324),('rooftops',427,428),('roof_hue',631,669),
('street_axis',637,634),('freeway_axis',878,814),('freeway_color',853,862),('harbor_rows',1054,1012),('berths',1044,1042),('berths',1015,1033),
('crosswalk',1139,1143),('crosswalk',1138,1116),('rooftops',1259,1264),('mobile_axis',1306,1336),
('bridge_axis',1451,1455),('bridge_axis',1443,1420),('bridge_slope',1457,1445),('bridge_slope',1441,1470),
('parking_fill',1514,1549),('parking_fill',1506,1508),('parking_ground',1574,1545),('parking_axis',1579,1503),
('river_axis',1660,1633),('river_bank',1613,1669),('river_bank',1604,1670),
('runway_number',1723,1739),('runway_number',1719,1733),('runway_bars',1729,1706),
('house_light',1814,1861),('house_pool',1869,1821),('house_pool',1812,1832),
('court_surround',2015,2036),('court_surround',2053,2006),('court_surround',2003,2040)]
D={
 'field_axis':('The planting furrows have a left-right orientation.','The planting furrows have a top-bottom orientation.'),
 'field_color':('The farmland looks green.','The farmland looks brown.'),
 'field_curve':('The furrows change direction as they cross the plot.','The furrows maintain a straight course across the plot.'),
 'infield':('Within the diamond there is a grass-covered patch around the mound.','Within the diamond the ground around the mound is dirt.'),
 'surf':('There is extensive breaking whitewater at the shore.','There is little breaking whitewater at the shore.'),
 'water_side':('Looking across the scene from left to right, water comes before sand.','Looking across the scene from left to right, sand comes before water.'),
 'rooftops':('Light shades predominate on the roofs.','Dark shades predominate on the roofs.'),
 'roof_hue':('The homes have red-toned roofing.','The homes have gray-toned roofing.'),
 'street_axis':('The residential road follows a left-right course.','The residential road follows a top-bottom course.'),
 'freeway_axis':('The highway lanes traverse the scene sideways.','The highway lanes traverse the scene from its top to its bottom.'),
 'freeway_color':('A light-colored highway surface is visible.','A dark-colored highway surface is visible.'),
 'harbor_rows':('Moored boats form rows across the width of the frame.','Moored boats form rows down the height of the frame.'),
 'berths':('Very few spaces for boats are vacant at the docks.','Many spaces for boats are vacant at the docks.'),
 'crosswalk':('There are painted stripes for pedestrians at the road crossing.','There are no painted stripes for pedestrians at the road crossing.'),
 'mobile_axis':('The access street between the trailers follows a left-right course.','The access street between the trailers follows a top-bottom course.'),
 'bridge_axis':('The deck on top has a left-right alignment.','The deck on top has a top-bottom alignment.'),
 'bridge_slope':('The left end of the upper road is lower in the frame than its right end.','The left end of the upper road is higher in the frame than its right end.'),
 'parking_fill':('The parking spaces are almost all occupied.','A large number of parking spaces are unoccupied.'),
 'parking_ground':('Grass is visible under the parked vehicles.','A paved lot is visible under the parked vehicles.'),
 'parking_axis':('The parking rows follow a left-right alignment.','The parking rows follow a top-bottom alignment.'),
 'river_axis':('The main course of the river goes across the frame sideways.','The main course of the river travels from the upper edge toward the lower edge.'),
 'river_bank':('A thick cover of green trees flanks the channel.','Green trees are sparse along the edges of the channel.'),
 'runway_number':('The runway carries a prominent number marking.','The runway has no prominent number marking.'),
 'runway_bars':('The runway includes conspicuous wide white bars.','There are no conspicuous wide white bars on the runway.'),
 'house_light':('Light-colored roofing covers the principal house.','Dark-colored roofing covers the principal house.'),
 'house_pool':('A swimming pool is visible on the residential property.','No swimming pool is visible on the residential property.'),
 'court_surround':('The outer apron of the tennis court is red.','The outer apron of the tennis court is green.')}
def main(assets):
    root=RUN/'confirm';out=root/'inputs';out.mkdir(parents=True,exist_ok=True)
    path=assets/'ucm/ucm_caption_karpathy_label_train.arrow';t=pa.ipc.open_file(pa.memory_map(str(path))).read_all();names=t['path'].to_pylist()
    previous=json.loads((RUN/'inputs/panel.json').read_text())
    seen=[r['image'] for r in previous]; blocked={r['rgb_sha256'] for r in previous}
    for dn in ['ucm','rsitmd']:
        ref=RUN.parent/'E3-2_UCM与RSITMD独立复验_20261124/inputs'/f'{dn}.json';blocked.update(r['rgb_sha256'] for r in json.loads(ref.read_text()))
    cv2.setNumThreads(1);cv2.setRNGSeed(20261124);sift=cv2.SIFT_create(nfeatures=200);bf=cv2.BFMatcher();cache={}
    def get(name):
        if name not in cache:
            im=Image.open(io.BytesIO(t['image'][names.index(name)].as_py())).convert('RGB')
            kp,des=sift.detectAndCompute(np.asarray(im.convert('L').resize((192,192),Image.Resampling.LANCZOS)),None)
            cache[name]=hashlib.sha256(im.tobytes()).hexdigest(),np.float32([k.pt for k in kp]),des
        return cache[name]
    def overlaps(a,b):
        h,x,v=get(a);g,y,w=get(b)
        if h==g:return True
        if v is None or w is None or min(len(v),len(w))<12:return False
        match=[m for m,n in bf.knnMatch(v,w,k=2) if m.distance<.72*n.distance]
        m={tuple(y[q.trainIdx].round().astype(int)):q for q in sorted(match,key=lambda m:-m.distance)}
        if len(m)<12:return False
        i=np.float32([x[q.queryIdx] for q in m.values()]);j=np.float32([y[q.trainIdx] for q in m.values()])
        tf,mask=cv2.estimateAffinePartial2D(i,j,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=3000)
        if tf is None:return False
        use=mask.ravel().astype(bool)
        return use.sum()>=12 and use.mean()>=.6 and min(cv2.contourArea(cv2.convexHull(z[use])) for z in (i,j))>=.1*192**2
    rows=[];skip=[]
    for key,a,b in NEW:
        ns=[f'{a}.tif',f'{b}.tif'];scene=(a-1)//100;assert scene==(b-1)//100 and len(set(ns+seen))==len(ns+seen)
        match=next(((n,r) for n in ns for r in seen if overlaps(n,r)),None)
        if match or overlaps(*ns) or any(get(n)[0] in blocked for n in ns):
            skip.append(dict(pair=f'{a}-{b}',overlap=match or ns));continue
        seen.extend(ns)
        sample=next(r for r in previous if r['scene_id']==scene)
        for side,n in enumerate(ns):
            j=names.index(n);caps=t['caption'][j].as_py();assert t['split'][j].as_py()=='train'
            rows.append(dict(image=n,row=j,split='train',rgb_sha256=get(n)[0],pair=f'{a}-{b}',attribute=key,side=side,scene_id=scene,
                         scene=SCENE[scene],scene_query=sample['scene_query'],orig_query=caps[0],captions=caps,
                         **{k:[p[side],p[1-side]] for k,p in zip('ABCD',[*BANK[key],D[key]],strict=True)}))
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'panel.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (out/'lock.json').write_text(json.dumps(dict(panel=sha(out/'panel.json'),train_arrow=sha(path),protocol=sha(root/'实验口径.md'),removed=skip),indent=2))
if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
