import colorsys
from typing import Optional

import matplotlib.patheffects
from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1


def add_colorbar(im, title: Optional[str] = None, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    cb = im.axes.figure.colorbar(im, cax=cax, **kwargs)
    if isinstance(title, str):
        cb.set_label(title)
    return cb


def adapt_hls_color(r: float, g: float, b: float, h_by: float = 0, l_by: float = 0, s_by: float = 0) -> (float, float, float):
    h,l,s = colorsys.rgb_to_hls(r,g,b)
    h = max([0, min([1,h+h_by])])
    l = max([0, min([1,l+l_by])])
    s = max([0, min([1,s+s_by])])

    return colorsys.hls_to_rgb(h,l,s)

