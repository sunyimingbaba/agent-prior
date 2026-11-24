"""Paired visual annotations on UCM test, frozen before any scores were inspected."""
from shared import ROOT,read,dump
from pathlib import Path
import sys
def main(source):
    old=read(source)
    def original(template):
        row=next(x for x in old if x['template']==template and x['cohort']=='extension' and x['dimension']=='attribute')
        return (row['positive'],row['negative'])
    items=[]
    def pair(a,b,name,pos,neg,origin):
        assert a!=b and pos!=neg
        for i,x,y in [(a,pos,neg),(b,neg,pos)]:
            items.append(dict(image=f'{i}.tif',pair=f'{a}-{b}',scene=(a-1)//100,template=name,
                              vocabulary=origin,positive=x,negative=y))
        assert (a-1)//100==(b-1)//100
    def matched(a,b,name):
        pair(a,b,name,*original(name),'old_extension')
    matched(697,700,'building_roofs')
    matched(1591,1596,'parking_occupancy')
    matched(1593,1595,'parking_occupancy')
    matched(1699,1696,'river_shape')
    matched(1891,1899,'roof_brightness')
    matched(1099,1091,'boat_rows')
    matched(1194,1191,'crosswalk_style')
    matched(1493,1497,'overpass_slant')
    def previous(a,b,image):
        row=next(r for r in old if r['image']==image and r['cohort']=='previous' and r['dimension']=='attribute')
        pair(a,b,row['template'],row['positive'],row['negative'],'old_other')
    previous(293,298,'288.tif')
    previous(297,299,'288.tif')
    previous(1495,1491,'1486.tif')
    previous(2091,2094,'2086.tif')
    def extra(a,b,name,positive,negative):
        pair(a,b,name,positive,negative,'new_wording')
    extra(87,85,'field_color','The field is golden yellow in color.','The field is muted brown in color.')
    extra(394,392,'surf','Broad bands of white breaking surf cover the water.','The water is relatively smooth with little white surf.')
    extra(497,496,'roof_color','The large central roof is bright white.','The large central roof is tan.')
    extra(691,694,'residential_street','The street runs horizontally across the image.','The street runs vertically through the image.')
    extra(891,895,'freeway_color','The freeway pavement is very light in color.','The freeway pavement is dark in color.')
    extra(898,894,'median','The freeway median contains a row of trees.','The freeway median is bare ground without a row of trees.')
    extra(1394,1392,'mobile_home_street','The straight street runs horizontally between the rows of mobile homes.','The straight street runs vertically between the rows of mobile homes.')
    extra(1594,1597,'parking_axis','The long rows of cars extend horizontally.','The long rows of cars extend vertically.')
    extra(1693,1700,'river_orientation','The river runs mainly horizontally through the image.','The river runs mainly vertically through the image.')
    extra(1794,1796,'runway_slant','From left to right the runway slopes downward.','From left to right the runway slopes upward.')
    extra(1895,1893,'roof_hue','The main house has a reddish roof.','The main house has a pale cream roof.')
    extra(1991,1996,'tank_surface','The visible circular tank surface is pale.','The visible circular tank surface is dark.')
    extra(2095,2096,'court_surround','The area surrounding the marked tennis playing surface is red.','The area surrounding the marked tennis playing surface is dark gray.')
    valid={r['image'] for r in read(ROOT/'inputs/ucm.json')}
    assert len(items)==len({r['image'] for r in items}) and all(r['image'] in valid for r in items)
    dump(ROOT/'inputs/attributes.json',items)
if __name__=='__main__':main(Path(sys.argv[1]))
