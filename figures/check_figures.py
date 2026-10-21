"""check_figures.py - check that the deposited figure files can be reproduced.

figures/r3/Fig1.tif ... Fig5.tif are the figure files submitted with the article. This
script runs figures/figures_r3.py into a temporary folder (the deposited files are not
touched) and compares every regenerated figure, pixel by pixel, with the deposited one.

Usage (from the repository root):  python figures/check_figures.py

The deposited files were rendered with matplotlib 3.10.8, pillow 12.1.1 and Liberation
Sans 2.1.5, a metric-compatible substitute for Arial (see requirements.txt). In that
environment all five figures are reproduced pixel-identically. figures_r3.py uses Arial
when it is installed, and other library versions render text slightly differently; small
pixel differences are then expected.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import PIL
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEPOSITED = ROOT / 'figures' / 'r3'
FONT_LIST = ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans']   # as in figures_r3.py


def font_in_use():
    with matplotlib.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_LIST}):
        path = font_manager.findfont(FontProperties(family=['sans-serif']))
    return font_manager.get_font(path).family_name, path


def main():
    family, path = font_in_use()
    print(f'matplotlib {matplotlib.__version__}, pillow {PIL.__version__}, font: {family} ({path})')
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(ROOT / 'figures' / 'figures_r3.py'),
                        str(ROOT / 'r3_tables'), tmp], check=True, cwd=ROOT)
        print('\nComparison with the deposited files in figures/r3/:')
        identical = 0
        for i in range(1, 6):
            dep = Image.open(DEPOSITED / f'Fig{i}.tif')
            new = Image.open(Path(tmp) / f'Fig{i}.tif')
            spec = (f'{dep.mode}, {dep.info.get("compression")}, '
                    f'{round(dep.info.get("dpi", (0, 0))[0])} dpi, {dep.width} x {dep.height} px')
            a = np.asarray(dep.convert('RGB'), dtype=np.int16)
            b = np.asarray(new.convert('RGB'), dtype=np.int16)
            if a.shape != b.shape:
                print(f'  Fig{i}.tif  size differs: deposited {dep.width} x {dep.height}, '
                      f'regenerated {new.width} x {new.height}  [{spec}]')
                continue
            n = int((np.abs(a - b).max(axis=2) > 0).sum())
            identical += n == 0
            print(f'  Fig{i}.tif  {"identical" if n == 0 else f"{n} pixels differ"}  [{spec}]')
    if identical == 5:
        print('\nAll five figures were reproduced pixel-identically.')
        sys.exit(0)
    print('\nNot all figures are pixel-identical. Small differences are expected with Arial or '
          'with other library versions (glyph shapes and a few pixels of layout); for an exact '
          'comparison use matplotlib 3.10.8, pillow 12.1.1 and Liberation Sans 2.1.5, without '
          'Arial installed.')
    sys.exit(1)


if __name__ == '__main__':
    main()
