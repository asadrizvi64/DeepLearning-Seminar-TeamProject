"""Figures and recordings of the browser viewer, from real screenshots.

The screenshots in figs/viewer_shots/ were taken in a browser of viewer/index.html:
  rotate_*.jpg  Drosophila fit (demo/drosophila_crop.splat, 500 Gaussians), orbited and zoomed
                  ?files=demo/drosophila_crop.splat&zscale=5&thr=0.3&size=1
  frame_*.jpg   synthetic time series (demo/timeseries/t000..t005.splat), frame slider 0..5
                  ?files=demo/timeseries/t000.splat,...,t005.splat&size=1.5

    python presentation/make_viewer_figures.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SHOTS = HERE / 'figs' / 'viewer_shots'
OUT = HERE / 'figs'
REPORT = HERE.parent / 'team_report' / 'figs'

CANVAS = (190, 240, 690, 640)      # region of the 800x771 screenshot that holds the splats
UI = (0, 0, 310, 290)              # the control panel


def _font(size):
    for f in ('arialbd.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label(im, text):
    d = ImageDraw.Draw(im)
    d.text((12, 10), text, fill=(235, 235, 240), font=_font(22))
    return im


def _row(images, gap=8):
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    h = max(i.height for i in images)
    row = Image.new('RGB', (w, h), (255, 255, 255))
    x = 0
    for i in images:
        row.paste(i, (x, 0)); x += i.width + gap
    return row


def _save(im, name):
    for d in (OUT, REPORT):
        im.save(d / name)
    print('wrote', name)


rot = [Image.open(SHOTS / f'rotate_{i}.jpg').convert('RGB') for i in range(6)]
frm = [Image.open(SHOTS / f'frame_{i}.jpg').convert('RGB') for i in range(6)]

# static viewer: control panel + three orbit angles of the same fit
panel = rot[0].crop(UI)
views = [_label(rot[i].crop(CANVAS), f'view {k}') for k, i in enumerate((0, 1, 3), 1)]
scale = views[0].height / panel.height
panel = panel.resize((int(panel.width * scale), views[0].height))
_save(_row([panel] + views), 'viewer_static.png')

# time series: control panel at frame 4 + frames 0, 2, 4, 5
panel = frm[4].crop((0, 0, 310, 290)).resize((int(310 * 400 / 290), 400))
views = [_label(frm[i].crop(CANVAS), f'frame {i}') for i in (0, 2, 4, 5)]
_save(_row([panel] + views), 'viewer_timeseries.png')

# short recordings for the slides (GIF plays in PowerPoint's slide show), cropped to the
# drawing area so the blobs are large enough on a slide
REC = (180, 230, 700, 650)
rec = [r.crop(REC) for r in rot] + [r.crop(REC) for r in rot[::-1]]
rec[0].save(OUT / 'viewer_rotate.gif', save_all=True, append_images=rec[1:], duration=450, loop=0)
rec = [_label(f.crop(REC), f'frame {i}') for i, f in enumerate(frm)]
rec[0].save(OUT / 'viewer_timeseries.gif', save_all=True, append_images=rec[1:], duration=700, loop=0)
print('wrote viewer_rotate.gif, viewer_timeseries.gif')
