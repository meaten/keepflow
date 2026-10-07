import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

 
# Elsevier asks for >=300 dpi (500 for combination line/halftone art). matplotlib's
# default of 100 put the density maps at roughly 210-280 effective dpi once placed in a
# column, so the figures were rejected on resolution before anyone read them. Override
# with KEEPFLOW_FIG_DPI when re-rendering artwork for submission.
_DPI = int(os.environ.get("KEEPFLOW_FIG_DPI", "100"))
_COLORBAR = os.environ.get("KEEPFLOW_FIG_COLORBAR", "0") == "1"

# matplotlib's "Greens" starts at #f7fcf5, a pale green rather than white, so a cell of
# density zero still takes ink and the whole panel reads as tinted. Anchor the ramp at
# pure white and leave absent cells (NaN) unpainted.
_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "greens_from_white",
    [(1.0, 1.0, 1.0)] + [plt.get_cmap("Greens")(v) for v in np.linspace(0.15, 1.0, 8)])
_CMAP.set_bad(color="white", alpha=0.0)


def plot_density(x: np.array, y: np.array, p: np.array, path: Path, traj = None,
                 dpi: int = None, gt_point = None, colorbar: bool = None,
                 cbar_label: str = "density (normalized)",
                 vmin: float = None, vmax: float = None) -> None:
    """Render one density map.

    gt_point marks the single ground-truth position being scored at this step. Without it
    the panel shows the whole trajectory but not the point the log-probability is
    evaluated at, so a confidently-misplaced density is indistinguishable from a good one.
    """
    mesh = plt.pcolormesh(x, y, np.ma.masked_invalid(p.reshape(x.shape)),
                          shading='auto',
                          cmap=_CMAP,
                          norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    if traj is not None:
        obs, gt = traj
        sns.lineplot(x=gt[:, 0], y=gt[:, 1],
                     color='black', marker='o')
        sns.lineplot(x=obs[:, 0], y=obs[:, 1],
                     color='green', marker='o')
    if gt_point is not None:
        plt.scatter([gt_point[0]], [gt_point[1]], marker='x', s=190,
                    linewidths=3.0, color='crimson', zorder=10)

    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    if _COLORBAR if colorbar is None else colorbar:
        cb = plt.colorbar(mesh, ax=plt.gca(), fraction=0.035, pad=0.015)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    plt.savefig(path, bbox_inches='tight', dpi=dpi or _DPI)
    plt.close()


if __name__ == "__main__":
    from scipy.stats import gaussian_kde

    offsets = [[-2.5, -0.5], [0.0, 1.0]]
    nbins=300
    xi, yi = np.mgrid[-10:10:nbins*1j, -10:10:nbins*1j]
    zi_list = []
    for offset in offsets:
        # create data
        x = np.random.normal(size=500, scale=0.5) + offset[0]
        y = np.random.normal(size=500, scale=0.5) + offset[1]
        
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        k = gaussian_kde([x,y])
        zi_list.append(k(np.vstack([xi.flatten(), yi.flatten()])))

    z = np.zeros_like(zi_list[0])
    for i, zi in enumerate(zi_list):
        z += zi
        plot_density(xi, yi, z/(i+1), path=f"{i}.png")
    

