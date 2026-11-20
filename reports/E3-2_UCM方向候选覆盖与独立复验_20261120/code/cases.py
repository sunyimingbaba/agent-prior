from shared import ROOT,read
NAMES={
    'driveway_house':('the driveway','the main house'),
    'pool_house':('the rectangular swimming pool','the main house'),
    'tank_vertical':('the round tank','the adjacent road'),
    'tank_horizontal':('the roadway','the tank site'),
    'diamond_plate':('home plate',"the pitcher's mound"),
    'diamond_field':('the dirt infield','the grassy outfield'),
    'beach_side':('the sea','the sandy shore'),
    'court_parking':('the parking area','the tennis courts'),
}
def query(row,condition):
    if condition=='scene':
        return row['scene_query']
    if condition!='objects':raise ValueError(condition)
    pair=row['landmarks'] if row['split']=='val' else NAMES[row['pair']]
    return row['scene_query']+' Locate '+pair[0]+' and '+pair[1]+'. Describe their relative positions in the image frame.'
def cases(group):
    return read(ROOT/'inputs'/('train.json' if group=='train' else 'fresh.json'))
