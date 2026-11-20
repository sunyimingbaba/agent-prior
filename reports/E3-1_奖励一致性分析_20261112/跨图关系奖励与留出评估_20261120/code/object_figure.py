"""Illustrate two observed, post-hoc cases with contrasting outcomes."""
import csv,hashlib,io,json,sys
from pathlib import Path
from PIL import Image,ImageFont,ImageDraw
import pyarrow as pa
root=Path(__file__).resolve().parents[1]
t=pa.ipc.open_file(str(Path(sys.argv[1])/'ucm_caption_karpathy_label_val.arrow')).read_all()
font_path=sys.argv[2]
def font(px):return ImageFont.truetype(font_path,px)
with (root/'object/images.csv').open(encoding='utf-8-sig') as fp:stats={r['image']:r for r in csv.DictReader(fp)}
cases={r['image']:r for r in json.loads((root/'object_inputs/cases.json').read_bytes())}
out=Image.new('RGB',(1000,730),'white');draw=ImageDraw.Draw(out)
draw.text((30,16),'同一幅图旋转后，方向奖励选出的先验还能判对吗？',font=font(32),fill='#1a2630')
draw.text((30,66),'原验证图    ·    场景名输入    ·    每张图从 8 条候选中选择',font=font(19),fill='#455665')
for j,(name,label,relations) in enumerate([
        ('1890.tif','住宅与泳池',['泳池在房屋上方','泳池在房屋下方']),
        ('1988.tif','储罐与阴影',['阴影在储罐右侧','阴影在储罐左侧'])]):
    y=112+j*296
    case=cases[name];r=stats[name]
    im=Image.open(io.BytesIO(t['image'][case['row']].as_py())).convert('RGB')
    assert hashlib.sha256(im.tobytes()).hexdigest()==case['rgb_sha256']
    draw.text((30,y+32),label,font=font(27),fill='#0b6160')
    draw.text((30,y+74),name,font=font(19),fill='#455665')
    for k,photo in enumerate([im,im.transpose(Image.Transpose.ROTATE_180)]):
        x=236+380*k
        out.paste(photo.resize((202,202)),(x,y))
        draw.text((x,y+207),('原图' if not k else '旋转 180°')+'  ·  '+relations[k],font=font(19),fill='#1a2630')
        val=float(r[('rotated' if k else 'scene')+'_AB'])
        draw.text((x,y+238),f'两套留出措辞平均判对率  {val:.0%}',font=font(18),fill='#264960')
draw.text((30,690),'两例选自评分后的 11 张原图，用于展示不同结果。汇总和逐图分数见随附报告。',font=font(18),fill='#455665')
out.save(root/'双物体旋转案例.png')
