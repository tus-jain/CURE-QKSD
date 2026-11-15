"""Matplotlib style of the paper figures. Call apply() before plotting."""

import matplotlib as mpl

from .paths import FIGURES


def figure_path(filename):
    """Return the repository output path for a figure."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / filename


NAVY = "#1b4f72"
BRICK = "#922b21"
SLATE = "#5d6d7e"
TEAL = "#117864"
OCHRE = "#b9770e"
INK = "#222222"
PALETTE = [NAVY, BRICK, TEAL, OCHRE, SLATE]


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.6,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.3,
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "grid.linestyle": ":",
        "axes.axisbelow": True,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
