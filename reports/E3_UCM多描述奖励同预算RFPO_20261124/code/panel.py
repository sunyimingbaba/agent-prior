"""Lock visually paired examples, remove overlapping views, and preserve an exclusion log."""
import sys,json,hashlib
from collections import Counter
from pathlib import Path
import cv2,numpy as np
from photo_pool import ROOT,REP,ASSET,read,sha,gather,prior_detail_names
LAST=REP/'E3_UCM同图偏好对与独立验收_20261124'
sys.path.insert(0,str(LAST/'code'))
from reward_manifest import BANK,D,E,SCENE
from confirm_manifest import samples,picture
EXTRA={
'golf_sand':[('A pale sand bunker is visible on the golf course.','No pale sand bunker is visible on the golf course.'),('The golf grounds contain a light-colored sand trap.','The golf grounds show no light-colored sand trap.'),('A patch of pale bunker sand appears beside the grass.','The grass is shown without a visible patch of bunker sand.')],
'golf_multi':[('Several sand bunkers are visible.','Just one sand bunker is visible.'),('At least two sand traps can be seen on the course.','A single sand trap can be seen on the course.'),('The view includes multiple separate sand hazards.','The view includes only one sand hazard.')],
'plane_scale':[('Most visible airplanes are small light aircraft.','Most visible airplanes are large passenger jets.'),('The aircraft are predominantly small planes.','The aircraft are predominantly large airliners.'),('Small light aircraft dominate the view.','Large passenger aircraft dominate the view.')],
'plane_count':[('Several large jets are visible.','Just one large jet is visible.'),('The scene shows multiple large airliners.','The scene shows a single large airliner.'),('More than one large passenger plane appears.','Only one large passenger plane appears.')],
'water_boat':[('A boat is visible on the river.','There is no visible boat on the river.'),('A vessel appears along the river channel.','The river channel shows no vessel.'),('The river scene includes a boat on the water.','The water is shown without a visible boat.')],
'basin_cover':[('The circular structures have light solid covers.','The interiors of the circular structures contain exposed dark water.'),('Solid pale tops cover the large round structures.','Dark water is visible inside the open round structures.'),('The circular tanks have a bright covered surface.','The circular tanks show an open dark liquid surface.')],
'basin_count':[('One large circular structure is visible.','At least two large circular structures are visible.'),('The view shows a single large round tank.','The view shows multiple large round tanks.'),('There is one large circular tank in this scene.','Several large circular tanks are present in this scene.')],
'street_curve':[('The main residential street curves.','The main residential street runs straight.'),('The street between the houses follows a bend.','The street between the houses follows a straight line.'),('The principal road through the homes is curved.','The principal road through the homes is straight.')],
'court_red':[('The area immediately outside the playing lines is red.','The area immediately outside the playing lines is not red.'),('Red paving borders the court markings.','There is no red paving bordering the court markings.'),('A reddish border surrounds the painted court.','The border around the painted court lacks a reddish color.')],
'paired_bars':[('Adjacent parallel white bars form a pair beside the centerline.','There is no pair of adjacent white bars beside the centerline.'),('Two neighboring white strips sit side by side to one side of the middle line.','No side-by-side pair of white strips appears next to the middle line.'),('Paired parallel white marks flank the runway center.','The surface lacks paired parallel white marks flanking its center.')],
'roof_red':[('Several roofs are rusty red or reddish brown.','The roofs are predominantly white or gray.'),('A group of red-brown rooftops appears.','Pale and gray rooftops predominate.'),('Red-brown roof surfaces are visible together.','Most rooftop surfaces have a neutral white or gray color.')]}
SCENES=dict(SCENE);SCENES.update({1:'airport apron',7:'forest',9:'golf course',19:'storage tank facility'})
PROPOSALS={
'field_axis':[(91,99),(92,96),(95,98)],

'infield':[(293,291),(294,292),(297,298)],
'surf':[(394,391),(396,392),(397,393)],
'roof_red':[(489,488),(492,497)],

'roof_hue':[(681,688)],'community_pool':[(614,689),(1287,1285),(1292,1295)],
'street_axis':[(691,694),(692,697)],
'street_curve':[(1296,1293)],
'freeway_color':[(891,895),(892,897),(893,896)],
'berths':[(1092,1100),(1094,1098)],'harbor_rows':[(1095,1091)],
'crosswalk':[(1194,1195)],
'mobile_axis':[(1394,1392)],
'bridge_axis':[(1498,1491),(1499,1492),(1500,1494)],
'bridge_curve':[(1496,1493)],
'parking_axis':[(1591,1597),(1594,1599)],
'river_axis':[(1693,1699),(1692,1700)],'water_boat':[(1697,1691),(1698,1694)],
'runway_bars':[(1782,1786),(1793,1797)],'paired_bars':[(1789,1788),(1794,1792),(1796,1799),(1800,1791)],
'house_pool':[(1891,1892),(1893,1898),(1894,1900)],
'court_surround':[(2091,2094)],'court_red':[(2092,2096),(2093,2099),(2095,2098),(2097,2100)],
'plane_scale':[(193,197),(194,198),(195,199)],'plane_count':[(191,196)],
'golf_sand':[(986,981),(991,993),(992,996),(995,998)],
'golf_multi':[(994,999)],'basin_cover':[(1991,1996),(1992,1998),(1993,1997)],
'basin_count':[(1994,1995)]}
def main():
    rows=gather();blocked,srcs=prior_detail_names();train,idx=samples()
    prior=set(blocked)
    for root in (LAST/'confirm',LAST/'reward_confirm',REP/'E3-2_UCM有效图组定向短训_20261124'):
        prior.update(r['image'] for r in read(root/'inputs/panel.json'))
        prior.update(n for r in read(root/'inputs/lock.json')['removed'] for n in r['images'])
    past=sorted(prior&idx.keys());original=len(past);cache={};sift=cv2.SIFT_create(nfeatures=200);bf=cv2.BFMatcher();cv2.setNumThreads(1);cv2.setRNGSeed(20261124)
    # Include earlier labeled validation/test images in the geometric comparison as well.
    import pyarrow as pa,io
    from PIL import Image
    other={}
    for sp in ('val','test'):
        t=pa.ipc.open_file(str(ASSET/f'ucm_caption_karpathy_label_{sp}.arrow')).read_all()
        other.update({n:(t,j) for j,n in enumerate(t['path'].to_pylist()) if n in prior})
    past+=sorted(set(other)-set(past))
    def get(n):
        if n not in cache:
            if n in rows:im=rows[n]['im']
            elif n in idx:im=picture(train,idx[n])
            else:
                t,j=other[n];im=Image.open(io.BytesIO(t['image'][j].as_py())).convert('RGB')
            k,ds=sift.detectAndCompute(np.asarray(im.convert('L').resize((192,192),3)),None)
            cache[n]=(hashlib.sha256(im.tobytes()).hexdigest(),np.float32([v.pt for v in k]),ds)
        return cache[n]
    def same(a,b):
        h,x,u=get(a);g,y,v=get(b)
        if h==g:return True
        if u is None or v is None or min(len(u),len(v))<12:return False
        ms=[m for m,n in bf.knnMatch(u,v,k=2) if m.distance<.72*n.distance]
        ms=list({tuple(y[m.trainIdx].round().astype(int)):m for m in sorted(ms,key=lambda z:-z.distance)}.values())
        if len(ms)<12:return False
        i=np.float32([x[m.queryIdx] for m in ms]);j=np.float32([y[m.trainIdx] for m in ms])
        tr,mask=cv2.estimateAffinePartial2D(i,j,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=3000)
        if tr is None:return False
        hit=mask.ravel().astype(bool)
        return hit.sum()>=12 and hit.mean()>=.6 and min(cv2.contourArea(cv2.convexHull(v[hit])) for v in (i,j))>=.1*192**2
    taken=set();out=[];removed=[];nprev=len(past)
    old=read(REP/'E3-1_UCM第二轮R3困难负样本复验_20261124/inputs/panel.json')
    for attr,ps in PROPOSALS.items():
        for a,b in ps:
            names=[f'{a}.tif',f'{b}.tif'];assert not set(names)&taken and set(names)<=set(rows) and not set(names)&prior,(attr,names)
            taken.update(names);sc=rows[names[0]]['scene_id'];assert rows[names[1]]['scene_id']==sc
            assert rows[names[0]]['split']==rows[names[1]]['split']
            overlap=next(((n,m) for n in names for m in past if same(n,m)),None)
            if overlap or same(*names):removed.append(dict(images=names,overlap=overlap or names));continue
            past+=names
            forms=EXTRA[attr] if attr in EXTRA else [BANK[attr][2],D[attr],E[attr]]
            query=next((c['scene_query'] for c in old if c['scene_id']==sc),'Scene: '+('an ' if SCENES[sc][0] in 'aeiou' else 'a ')+SCENES[sc]+'.')
            for side,n in enumerate(names):
                c=rows[n];out.append(dict(**{k:c[k] for k in ('scene_id','split','row','rgb_sha256','orig_query')},image=n,pair=f'{a}-{b}',side=side,attribute=attr,scene_query=query,**{k:[v[side],v[1-side]] for k,v in zip('CDE',forms)}))
    p=ROOT/'inputs/panel.json';p.write_text(json.dumps(out,indent=2))
    (ROOT/'inputs/panel_lock.json').write_text(json.dumps(dict(panel=sha(p),protocol=sha(ROOT/'实验口径.md'),selection_source=sha(Path(__file__)),arrows={sp:sha(ASSET/f'ucm_caption_karpathy_label_{sp}.arrow') for sp in ('val','test')},prior_geometric=nprev,original_training_names=original,excluded_name_sources=srcs,proposed=len(taken)//2,removed=removed,pairs=len(out)//2,images_by_split=Counter(c['split'] for c in out)),indent=2))
if __name__=='__main__':main()
