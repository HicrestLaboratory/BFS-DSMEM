"""Shared matplotlib style for every figure in this folder.

Import it once, before any figure is created, for its side effect:

    import plotstyle                       # noqa: F401
    fig = plotstyle.figure(figsize=(9, 5.5))
    ...
    plotstyle.save(fig, "out.png")

The point of the large type is that these figures go straight onto beamer
slides at roughly half the slide width. A figure whose labels are comfortable
on a laptop screen is illegible from the back of a room, so the text is sized
relative to the figure rather than to the page.

Spines follow the Tufte convention: the top and right frame lines carry no
information, so they are off.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

RC = {
    # "text.usetex": True,
    # "text.latex.preamble": r"\usepackage{siunitx} \usepackage{sansmath} \sansmath",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 9,
    "legend.title_fontsize": 12,
    "figure.titlesize": 20,
    "axes.spines.right": False,   # Tufte: drop the frame lines that carry no data
    "axes.spines.top": False,
}

# A legend at 9 pt next to 20 pt tick labels is hard to read on a projector.
# Uncomment to scale it with the rest; everything above is the house style.
# RC["legend.fontsize"] = 16

mpl.rcParams.update(RC)

# A few settings that are not about type but keep these plots consistent.
mpl.rcParams.update({
    "figure.constrained_layout.use": True,  # no clipped labels at this type size
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def figure(**kwargs):
    """plt.figure with the house defaults already applied."""
    return plt.figure(**kwargs)


def subplots(*args, **kwargs):
    """plt.subplots with the house defaults already applied."""
    return plt.subplots(*args, **kwargs)


def save(fig, path, also_pdf=True):
    """Write `path`, and the same figure as .pdf beside it for LaTeX."""
    fig.savefig(path)
    out = [path]
    if also_pdf and not path.endswith(".pdf"):
        pdf = path.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf)
        out.append(pdf)
    return out
