"""Draw numbered original pixels for label review. The models always use unmarked source images."""
from pathlib import Path
import io,json,sys
from PIL import Image,ImageDraw
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[1]
def render(data,ids,out):
    t=pa.ipc.open_file(str(data/'ucm_caption_karpathy_label_val.arrow')).read_all()
    names=t['path'].to_pylist()
    for j in range(0,len(ids),6):
        sheet=Image.new('RGB',(3*300,2*324),'white');draw=ImageDraw.Draw(sheet)
        for pos,n in enumerate(ids[j:j+6]):
            image=Image.open(io.BytesIO(t['image'][names.index(n)].as_py())).convert('RGB')
            w,h=image.size
            x,y=pos%3*300,pos//3*324
            sheet.paste(image.resize((300,300)),(x,y));draw.text((x+2,y+301),n,fill='black')
        sheet.save(out/f'panel_{j//6:02}.jpg',quality=97)
if __name__=='__main__':
    render(Path(sys.argv[1]),json.loads(Path(sys.argv[2]).read_bytes()),Path(sys.argv[3]))
