"""Recompose Figure 9(b) with legible lettering.

The original composite was assembled at a canvas size that left the row and column
labels at roughly 3.5 pt once placed in the manuscript, below Elsevier's 7 pt floor.
The nine density panels themselves are fine, so this script crops them out of the old
composite and redraws the labels and the colour bar at a readable size.

Crop boxes were read off the source with a whitespace profile; they are hard-coded
because the source composite is a fixed artefact.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

SRC = "202604_Neurocomputing_FlowChain-CNF/figures_e6/fig9b_update_vs_kde_baseline_eth_2_11_t34.png"
DST = "202604_Neurocomputing_FlowChain-CNF/maeda11.png"

COLS = [(170, 1110), (1640, 2690), (3110, 4210)]
ROWS = [(200, 610), (1050, 1420), (1780, 2140)]
COL_TITLES = ["$t{+}6$", "$t{+}7$", "$t{+}8$"]
ROW_TITLES = ["FlowChain-CNF\nwithout update",
              "FlowChain-CNF\nwith update",
              "MID\n(KDE baseline)"]

_CMAP = LinearSegmentedColormap.from_list(
    "greens_from_white",
    [(1.0, 1.0, 1.0)] + [plt.get_cmap("Greens")(v) for v in np.linspace(0.15, 1.0, 8)])


def main():
    src = np.array(Image.open(SRC).convert("RGB"))
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 4.6))
    for r, (r0, r1) in enumerate(ROWS):
        for c, (c0, c1) in enumerate(COLS):
            ax = axes[r][c]
            ax.imshow(src[r0:r1, c0:c1])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(COL_TITLES[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(ROW_TITLES[r], fontsize=9, labelpad=6)
    fig.subplots_adjust(left=0.11, right=0.88, top=0.92, bottom=0.02,
                        wspace=0.04, hspace=0.06)
    cax = fig.add_axes([0.905, 0.12, 0.018, 0.72])
    cb = ColorbarBase(cax, cmap=_CMAP, norm=Normalize(vmin=0, vmax=1))
    cb.set_label("density (normalized)", fontsize=9)
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=8)
    fig.savefig(DST, dpi=600)
    print("wrote", DST)


if __name__ == "__main__":
    main()
