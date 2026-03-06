"""
Publication-quality QA plotting functions for HSC S23b training catalogs.

Modeled on s19a_plot.py. All 2D plots use hist2d + LogNorm + scatter background.
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

__all__ = [
    "plot_scale_dz",
    "plot_mag_dz",
    "plot_mag_z",
    "plot_color_z",
    "plot_color_color",
    "plot_z_hist_by_source",
    "plot_z_compare",
    "plot_completeness_heatmap",
    "plot_qa_metrics",
]

# ---------------------------------------------------------------------------
# Style defaults — shared across all hist2d-based functions
# ---------------------------------------------------------------------------
HIST2D_DEFAULTS = dict(
    bins=[120, 100],
    cmin=2,
    cmap="viridis",
    scatter_color="steelblue",
    scatter_s=2,
    scatter_alpha=0.2,
    grid_ls="--",
    grid_lw=2,
    grid_alpha=0.6,
    label_fontsize=25,
    text_fontsize=30,
)

SOURCE_STYLE = {
    "DESI_DR1": dict(cmap="viridis", color="steelblue", label=r"$\rm DESI$"),
    "COSMOSWeb2025_v1": dict(cmap="inferno", color="orangered", label=r"$\rm COSMOSWeb$"),
    "DESI_DR1/COSMOSWeb2025_v1": dict(cmap="cividis", color="forestgreen", label=r"$\rm Dual$"),
}


def _apply_grid(ax):
    """Apply standard grid styling."""
    ax.grid(linestyle="--", linewidth=2, alpha=0.6)


def _hist2d_with_scatter(ax, x, y, bins, xlim, ylim, cmap, cmin,
                         scatter_color, highlight=None, add_colorbar=True):
    """Core 2D histogram with scatter underlay and optional highlight overlay.

    Returns the hist2d return tuple (counts, xedges, yedges, image).
    """
    # Scatter background
    ax.scatter(x, y, s=2, alpha=0.2, color=scatter_color)

    # 2D histogram
    h = ax.hist2d(
        x, y, bins=bins, cmin=cmin, cmap=cmap,
        range=[list(xlim), list(ylim)],
        norm=LogNorm(),
    )

    # Highlight overlay
    if highlight is not None:
        ax.scatter(x[highlight], y[highlight], s=3, alpha=0.5, color="r")

    # Colorbar
    if add_colorbar and h[3] is not None:
        plt.colorbar(h[3], ax=ax, pad=0.02, fraction=0.046)

    _apply_grid(ax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return h


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_scale_dz(scale, logdz, ax=None, color="steelblue", cmap="viridis",
                  bins=None, xlim=(0.09, 1.03), ylim=(-5.9, 0.9),
                  cmin=2, label=None, highlight=None, add_colorbar=True):
    """Scale factor a=1/(1+z) vs log10(dz/(1+z)).

    Includes secondary x-axis showing redshift.
    """
    if bins is None:
        bins = [120, 100]
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)

    _hist2d_with_scatter(
        ax, scale, logdz, bins=bins, xlim=xlim, ylim=ylim,
        cmap=cmap, cmin=cmin, scatter_color=color,
        highlight=highlight, add_colorbar=add_colorbar,
    )

    # Secondary axis: scale factor -> redshift
    def forward(a):
        return (1.0 - a) / a

    def inverse(z):
        return 1.0 / (1.0 + z)

    secax = ax.secondary_xaxis("top", functions=(forward, inverse))
    secax.set_xticks([0, 1, 2, 3, 4, 6])
    secax.set_xlabel(r"$\rm Redshift$", fontsize=25)

    ax.set_xlabel(r"$\rm Scale\ Factor$", fontsize=25)
    ax.set_ylabel(r"$\log_{10}[\delta z / (1 + z)]$", fontsize=25)

    if label is not None:
        ax.text(0.10, 0.9, label, fontsize=30, transform=ax.transAxes)

    return ax


def plot_mag_dz(mag, logdz, ax=None, color="steelblue", cmap="viridis",
                bins=None, xlim=(16.5, 27.9), ylim=(-5.9, 0.9),
                cmin=2, label=None, phot="CModel", highlight=None,
                add_colorbar=True):
    """Magnitude vs log10(dz/(1+z))."""
    if bins is None:
        bins = [120, 100]
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)

    _hist2d_with_scatter(
        ax, mag, logdz, bins=bins, xlim=xlim, ylim=ylim,
        cmap=cmap, cmin=cmin, scatter_color=color,
        highlight=highlight, add_colorbar=add_colorbar,
    )

    ax.set_xlabel(
        r"${{\rm {:s}}}\ i\ [\rm mag]$".format(phot), fontsize=25)
    ax.set_ylabel(r"$\log_{10}[\delta z / (1 + z)]$", fontsize=25)

    if label is not None:
        ax.text(0.70, 0.9, label, fontsize=30, transform=ax.transAxes)

    return ax


def plot_mag_z(mag, z, ax=None, color="steelblue", cmap="viridis",
               bins=None, xlim=(16.5, 27.9), ylim=(-0.09, 5.9),
               cmin=2, label=None, phot="CModel", highlight=None,
               add_colorbar=True):
    """Magnitude vs redshift."""
    if bins is None:
        bins = [120, 100]
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)

    _hist2d_with_scatter(
        ax, mag, z, bins=bins, xlim=xlim, ylim=ylim,
        cmap=cmap, cmin=cmin, scatter_color=color,
        highlight=highlight, add_colorbar=add_colorbar,
    )

    ax.set_xlabel(
        r"${{\rm {:s}}}\ i\ [\rm mag]$".format(phot), fontsize=25)
    ax.set_ylabel(r"$\rm Redshift$", fontsize=25)

    if label is not None:
        ax.text(0.1, 0.9, label, fontsize=30, transform=ax.transAxes)

    return ax


def plot_color_z(color_val, z, ax=None, scatter_color="steelblue", cmap="viridis",
                 band_1="g", band_2="r", bins=None,
                 xlim=(-1.9, 4.9), ylim=(-0.09, 5.9),
                 cmin=2, label=None, phot="CModel", highlight=None,
                 add_colorbar=True):
    """Color vs redshift. Takes pre-computed color array."""
    if bins is None:
        bins = [120, 100]
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)

    _hist2d_with_scatter(
        ax, color_val, z, bins=bins, xlim=xlim, ylim=ylim,
        cmap=cmap, cmin=cmin, scatter_color=scatter_color,
        highlight=highlight, add_colorbar=add_colorbar,
    )

    ax.set_xlabel(
        r"${{\rm {:s}}}\ {:s}-{:s}\ [\rm mag]$".format(phot, band_1, band_2),
        fontsize=25)
    ax.set_ylabel(r"$\rm Redshift$", fontsize=25)

    if label is not None:
        ax.text(0.1, 0.9, label, fontsize=30, transform=ax.transAxes)

    return ax


def plot_color_color(color_x, color_y, ax=None, scatter_color="steelblue",
                     cmap="viridis", bins=None,
                     xlabel="g - r", ylabel="r - i",
                     xlim=(-1.0, 3.0), ylim=(-1.0, 3.0),
                     cmin=2, label=None, highlight=None,
                     add_colorbar=True):
    """Color-color diagram with 2D histogram."""
    if bins is None:
        bins = [120, 100]
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)

    _hist2d_with_scatter(
        ax, color_x, color_y, bins=bins, xlim=xlim, ylim=ylim,
        cmap=cmap, cmin=cmin, scatter_color=scatter_color,
        highlight=highlight, add_colorbar=add_colorbar,
    )

    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)

    if label is not None:
        ax.text(0.1, 0.9, label, fontsize=30, transform=ax.transAxes)

    return ax


def plot_z_hist_by_source(redshift, sources, ax=None, z_bins=None,
                          source_styles=None, xlim=(0, 5)):
    """Redshift histogram split by spectroscopic source."""
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.subplot(111)
    if z_bins is None:
        z_bins = np.arange(0, 7.5, 0.05)
    if source_styles is None:
        source_styles = SOURCE_STYLE

    sources_unique = sorted(set(sources))
    for src in sources_unique:
        mask = sources == src
        style = source_styles.get(src, {})
        color = style.get("color", "C3")
        label_str = style.get("label", src)
        ax.hist(
            redshift[mask], bins=z_bins, alpha=0.6,
            color=color, label=f"{label_str} (N={mask.sum():,})",
            histtype="stepfilled",
        )

    ax.set_xlabel(r"$\rm Redshift$", fontsize=20)
    ax.set_ylabel(r"$\rm Count$", fontsize=20)
    ax.set_xlim(xlim)
    ax.legend(fontsize=12)
    _apply_grid(ax)

    return ax


def plot_z_compare(z_1, z_2, ax=None, bins=None, z_min=0.0, z_max=5.8,
                   color="steelblue", cmap="viridis",
                   xlabel=None, ylabel=None, label=None,
                   add_colorbar=True):
    """Compare two redshift measurements (e.g., DESI vs COSMOSWeb)."""
    if bins is None:
        bins = [120, 120]
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)

    xlim = (z_min - 0.1, z_max + 0.1)
    ylim = xlim

    _hist2d_with_scatter(
        ax, z_1, z_2, bins=bins, xlim=xlim, ylim=ylim,
        cmap=cmap, cmin=2, scatter_color=color,
        add_colorbar=add_colorbar,
    )

    # 1:1 line
    ax.plot([z_min - 0.1, z_max + 0.1], [z_min - 0.1, z_max + 0.1],
            c="r", alpha=0.8, linestyle="--", linewidth=3)

    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([0, 1, 2, 3, 4, 5])

    if xlabel is None:
        xlabel = r"$z_1$"
    if ylabel is None:
        ylabel = r"$z_2$"
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)

    if label is not None:
        ax.text(0.25, 0.9, label, fontsize=30, transform=ax.transAxes)

    return ax


def plot_completeness_heatmap(completeness_matrix, flux_type_labels, band_labels,
                              ax=None, vmin=80, vmax=100, cmap="RdYlGn"):
    """Annotated heatmap of per-band completeness by flux type."""
    if ax is None:
        fig = plt.figure(figsize=(8, max(3, 0.6 * len(flux_type_labels))))
        ax = plt.subplot(111)

    im = ax.imshow(completeness_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(band_labels)))
    ax.set_xticklabels(band_labels)
    ax.set_yticks(range(len(flux_type_labels)))
    ax.set_yticklabels(flux_type_labels)
    ax.set_title("Photometric Completeness (% finite & positive)")

    for i in range(len(flux_type_labels)):
        for j in range(len(band_labels)):
            val = completeness_matrix[i, j]
            text_color = "white" if val < 90 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=10, color=text_color)

    plt.colorbar(im, ax=ax, label="%", pad=0.02, fraction=0.046)

    return ax


# ---------------------------------------------------------------------------
# Photo-z QA metrics panel (replaces fig_08 / fig_09)
# ---------------------------------------------------------------------------

# Marker and color conventions (s19a style):
#   solid marker  = biweighted   |   empty marker = conventional
#   circle = bias, square = scatter, triangle-down = outlier, x = loss
_QA_METRIC_STYLE = {
    "bias_bw":        dict(marker="o", color="skyblue",  ms=7, label=r"$\rm bias$"),
    "bias_conv":      dict(marker="o", color="skyblue",  ms=7, facecolor="none", lw=1.8),
    "sigma_bw":       dict(marker="s", color="orange",   ms=7, label=r"$\sigma$"),
    "sigma_conv":     dict(marker="s", color="orange",   ms=7, facecolor="none", lw=1.8),
    "f_outlier_bw":   dict(marker="v", color="green",    ms=7, label=r"$f_{\rm out}$"),
    "f_outlier_conv": dict(marker="v", color="green",    ms=7, facecolor="none", lw=1.8),
    "avg_loss":       dict(marker="x", color="red",      ms=7, label=r"$\rm loss$", lw=1.8),
}


def plot_qa_metrics(qa_mag, qa_z, output_path=None,
                    figsize=(18, 10), dpi=150, title=None):
    """QA metrics scatter + delta_z violin, vs i-mag and vs z_phot.

    Follows s19a ``plot_qa_mag_z`` style: scatter plots with solid (biweighted)
    and empty (conventional) markers for each metric, plus violin panels
    showing per-bin delta_z distributions.

    Parameters
    ----------
    qa_mag : dict
        Output of ``qa_mag_bins()``.  Keys include metric arrays and
        ``delta_z_per_bin`` (list of arrays).
    qa_z : dict
        Output of ``qa_z_bins()``.
    output_path : str or Path, optional
        If given, save and close.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    title : str, optional
        Suptitle text.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(
        2, 2, figsize=figsize,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )

    # --- Helper: scatter one QA panel ---
    def _scatter_qa(ax, data, xlabel):
        ax.axhline(0.0, color="k", ls="--", lw=1.5, zorder=1)
        ax.grid(ls="--", lw=1.5, alpha=0.3)
        centers = data["bin_centers"]

        for key, style in _QA_METRIC_STYLE.items():
            if key not in data:
                continue
            vals = data[key]
            kw = dict(s=style["ms"] ** 2, alpha=0.85, zorder=5)
            if "facecolor" in style:
                kw["facecolors"] = style["facecolor"]
                kw["edgecolors"] = style["color"]
                kw["linewidths"] = style.get("lw", 1.5)
            else:
                kw["color"] = style["color"]
            if style["marker"] == "x":
                kw.pop("facecolors", None)
                kw.pop("edgecolors", None)
                kw["color"] = style["color"]
                kw["linewidths"] = style.get("lw", 1.5)
            kw["marker"] = style["marker"]
            kw["label"] = style.get("label")
            ax.scatter(centers, vals, **kw)

        ax.set_xlabel(xlabel, fontsize=20)
        ax.legend(loc="best", fontsize=14, ncol=2, handletextpad=0.3)

    # --- Helper: violin panel ---
    def _violin_dz(ax, data, xlabel):
        ax.axhline(0.0, color="k", ls="--", lw=1.5, zorder=1)
        ax.grid(ls="--", lw=1.5, alpha=0.3)
        centers = data["bin_centers"]
        dz_list = data["delta_z_per_bin"]
        step = centers[1] - centers[0] if len(centers) > 1 else 0.5

        # Collect valid bins for violin
        positions = []
        violin_data = []
        for i, dz in enumerate(dz_list):
            if len(dz) >= 10:
                positions.append(centers[i])
                # Clip extreme outliers for display
                clipped = np.clip(dz, -1.0, 1.0)
                violin_data.append(clipped)

        if violin_data:
            parts = ax.violinplot(
                violin_data, positions=positions, widths=0.8 * step,
                showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("steelblue")
                pc.set_alpha(0.4)
            parts["cmedians"].set_color("k")
            parts["cmedians"].set_linewidth(1.5)

        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(r"$\Delta z / (1 + z)$", fontsize=18)
        ax.set_ylim(-0.5, 0.5)

    # Top-left: metrics scatter vs i-mag
    _scatter_qa(axes[0, 0], qa_mag, r"$i\ [\rm mag]$")
    # Top-right: metrics scatter vs z_spec
    _scatter_qa(axes[0, 1], qa_z, r"$z_{\rm spec}$")
    # Bottom-left: violin vs i-mag
    _violin_dz(axes[1, 0], qa_mag, r"$i\ [\rm mag]$")
    # Bottom-right: violin vs z_spec
    _violin_dz(axes[1, 1], qa_z, r"$z_{\rm spec}$")

    if title is not None:
        fig.suptitle(title, fontsize=18, y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.97] if title else None)

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig
