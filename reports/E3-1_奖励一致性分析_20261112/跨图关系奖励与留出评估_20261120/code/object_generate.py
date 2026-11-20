from shared import ROOT,read
from generate import main
main(read(ROOT/'object_inputs/cases.json'),object_views=True,output=ROOT/'object')
