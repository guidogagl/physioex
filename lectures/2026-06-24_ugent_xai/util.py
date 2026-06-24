"""Plotting utilities for the PhysioEx lecture notebook."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import welch

# ── Global style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
STAGE_COLORS = ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71", "#9B59B6"]

# EEG frequency bands for PSD annotation (non-overlapping)
DISPLAY_BANDS = [
    {"name": "δ",  "low": 0.5, "high": 4.0,  "color": "#2ECC71"},
    {"name": "θ",  "low": 4.0, "high": 8.0,  "color": "#F39C12"},
    {"name": "α",  "low": 8.0, "high": 11.0, "color": "#E74C3C"},
    {"name": "σ",  "low": 11.0, "high": 16.0, "color": "#3498DB"},
    {"name": "β",  "low": 16.0, "high": 30.0, "color": "#9B59B6"},
]

# Band boundary frequencies for annotation lines on spectrograms
_BAND_BOUNDARIES = sorted({b["low"] for b in DISPLAY_BANDS} | {b["high"] for b in DISPLAY_BANDS})


def _add_band_labels_top(ax, bands=None, fontsize=12):
    """Add band name labels along the top inside edge of an axis.

    Uses axes-fraction Y coordinate so position is independent of data scale.
    """
    if bands is None:
        bands = DISPLAY_BANDS
    for band in bands:
        mid = (band["low"] + band["high"]) / 2
        ax.annotate(
            band["name"], xy=(mid, 0.93), xycoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=fontsize, fontweight="bold",
            color=band["color"],
        )


def _add_band_shading_v(ax, bands=None, alpha=0.12):
    """Add vertical band shading (for frequency on X axis)."""
    if bands is None:
        bands = DISPLAY_BANDS
    for band in bands:
        ax.axvspan(band["low"], band["high"], alpha=alpha, color=band["color"],
                   zorder=0)


def _add_band_shading_h(ax, bands=None, alpha=0.12):
    """Add horizontal band shading (for frequency on Y axis)."""
    if bands is None:
        bands = DISPLAY_BANDS
    for band in bands:
        ax.axhspan(band["low"], band["high"], alpha=alpha, color=band["color"],
                   zorder=0)


def _add_band_hlines(ax, color="0.5", alpha=0.5, linestyle="--", linewidth=0.7):
    """Add horizontal dashed lines at EEG band boundaries."""
    for freq in _BAND_BOUNDARIES:
        ax.axhline(freq, color=color, alpha=alpha, linestyle=linestyle,
                   linewidth=linewidth, zorder=1)


def _clean_spines(ax, top=False, right=False):
    """Remove specified spines."""
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)


# ── Plot functions ────────────────────────────────────────────────────────

def plot_two_epochs(sample_raw, sample_spec, epoch_idx=10):
    """Side-by-side: raw waveform vs spectrogram for the same epoch index."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    ch_raw = sample_raw["channel_order"][0]
    sig = sample_raw["signals"][ch_raw][epoch_idx].numpy()
    label_raw = sample_raw["labels"][epoch_idx].item()
    t = np.linspace(0, 30, len(sig))
    ax1.plot(t, sig, linewidth=0.5, color="#2563EB")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(
        f"Raw EEG — {STAGE_NAMES[label_raw] if label_raw >= 0 else '?'}",
    )
    ax1.set_xlim(0, 30)

    ch_spec = sample_spec["channel_order"][0]
    spec = sample_spec["signals"][ch_spec][epoch_idx].numpy()
    label_spec = sample_spec["labels"][epoch_idx].item()
    ax2.imshow(
        spec.T, aspect="auto", origin="lower", cmap="viridis",
        extent=[0, 30, 0, 50],
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title(
        f"Spectrogram — {STAGE_NAMES[label_spec] if label_spec >= 0 else '?'}",
    )

    plt.tight_layout()
    return fig


def plot_hypnogram(labels, title="Hypnogram", pred_labels=None):
    """Plot a hypnogram. If pred_labels given, show true vs predicted."""
    n_rows = 1 if pred_labels is None else 2
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(14, 2.0 * n_rows + 0.3), sharex=True,
        gridspec_kw={"hspace": 0.25},
    )
    if n_rows == 1:
        axes = [axes]

    items = (
        [(labels, "True")]
        if pred_labels is None
        else [(labels, "True"), (pred_labels, "Predicted")]
    )

    n_epochs = len(labels)

    for ax, (labs, name) in zip(axes, items):
        for i in range(len(labs)):
            l = labs[i].item() if hasattr(labs[i], "item") else int(labs[i])
            if l < 0:
                continue
            ax.fill_between(
                [i, i + 1], l - 0.35, l + 0.35,
                color=STAGE_COLORS[l], alpha=0.85, linewidth=0,
            )
        ax.set_yticks(range(5))
        ax.set_yticklabels(STAGE_NAMES)
        ax.set_ylabel(name, fontweight="bold")
        ax.set_ylim(-0.5, 4.5)
        ax.invert_yaxis()
        _clean_spines(ax)

        # Subtle vertical gridlines every 5 epochs
        ax.set_xticks(range(0, n_epochs + 1, 5), minor=False)
        ax.set_xticks(range(0, n_epochs + 1), minor=True)
        ax.grid(axis="x", which="major", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.tick_params(axis="x", which="minor", length=0)

    axes[-1].set_xlabel("Epoch (30 s each)")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_ig_raw(signal, attribution, pred_label, epoch_idx, model_name="Model"):
    """Plot raw EEG signal with IG attribution overlay."""
    sig = signal.squeeze().detach().cpu().numpy()
    att = attribution.squeeze().detach().cpu().numpy()
    t = np.linspace(0, 30, len(sig))

    # Normalize attribution for visibility
    att_max = np.abs(att).max()
    if att_max > 0:
        att_norm = att / att_max
    else:
        att_norm = att

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 5.5), sharex=True,
        gridspec_kw={"hspace": 0.08, "height_ratios": [1, 1]},
    )

    # Top: raw EEG
    ax1.plot(t, sig, linewidth=0.5, color="#2563EB")
    ax1.set_ylabel("EEG (µV)")
    ax1.set_title(
        f"{model_name} — IG for {STAGE_NAMES[pred_label]} (epoch {epoch_idx})",
    )
    ax1.grid(axis="x", alpha=0.2, linestyle="--", linewidth=0.5)
    _clean_spines(ax1)

    # Bottom: attribution
    ax2.fill_between(t, 0, att_norm, where=att_norm > 0,
                     color="#E74C3C", alpha=0.6, label="+ attribution", linewidth=0)
    ax2.fill_between(t, 0, att_norm, where=att_norm < 0,
                     color="#3498DB", alpha=0.6, label="− attribution", linewidth=0)
    ax2.axhline(0, color="0.5", linewidth=0.5)
    ax2.set_ylabel("Norm. Attribution")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(0, 30)
    ax2.grid(axis="x", alpha=0.2, linestyle="--", linewidth=0.5)
    ax2.legend(loc="upper left", framealpha=0.8, edgecolor="none")
    _clean_spines(ax2)

    plt.tight_layout()
    return fig


def plot_ig_spectrogram(spectrogram, attribution, true_label, pred_label, model_name="Model"):
    """Plot spectrogram with IG attribution side-by-side + band boundary lines."""
    spec_np = spectrogram.squeeze().detach().cpu().numpy()
    attr_np = attribution.squeeze().detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5),
                                    gridspec_kw={"wspace": 0.25})

    # Left: spectrogram
    ax1.imshow(
        spec_np.T, aspect="auto", origin="lower", cmap="viridis",
        extent=[0, 30, 0, 50],
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(f"Spectrogram — {STAGE_NAMES[true_label]}")
    ax1.set_ylim(0, 45)
    _add_band_hlines(ax1, color="white", alpha=0.4, linewidth=0.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: IG attribution
    vmax = np.abs(attr_np).max()
    im = ax2.imshow(
        attr_np.T, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, extent=[0, 30, 0, 50],
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title(f"IG Attribution — {STAGE_NAMES[pred_label]}")
    ax2.set_ylim(0, 45)
    _add_band_hlines(ax2, color="0.4", alpha=0.6, linewidth=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Band name labels on the right edge of IG panel
    for band in DISPLAY_BANDS:
        mid_y = (band["low"] + band["high"]) / 2
        if mid_y <= 45:
            ax2.annotate(
                band["name"], xy=(1.02, mid_y),
                xycoords=("axes fraction", "data"),
                ha="left", va="center", fontsize=10, fontweight="bold",
                color=band["color"],
            )

    cb = plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.08)
    cb.set_label("Attribution", fontsize=10)

    plt.tight_layout()
    return fig


def plot_epoch_with_psd(signal, fs, stage_name):
    """Raw EEG waveform (left) + Welch PSD with EEG band annotations (right)."""
    sig = signal.squeeze().detach().cpu().numpy() if hasattr(signal, "detach") else np.asarray(signal).squeeze()
    t = np.linspace(0, 30, len(sig))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 4.5),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.30},
    )

    # Left: raw waveform
    ax1.plot(t, sig, linewidth=0.5, color="#2563EB")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (µV)")
    ax1.set_title(f"Raw EEG — {stage_name}")
    ax1.set_xlim(0, 30)
    _clean_spines(ax1)

    # Right: PSD with band shading
    freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    mask = freqs <= 40
    ax2.semilogy(freqs[mask], psd[mask], color="#2563EB", linewidth=1.2)
    _add_band_shading_v(ax2)
    _add_band_labels_top(ax2, fontsize=12)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD (µV²/Hz)")
    ax2.set_title(f"Power Spectrum — {stage_name}")
    ax2.set_xlim(0, 40)
    # Tighten Y range to the data
    psd_in_range = psd[mask]
    psd_min = psd_in_range[psd_in_range > 0].min() * 0.3
    psd_max = psd_in_range.max() * 3
    ax2.set_ylim(psd_min, psd_max)
    ax2.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    _clean_spines(ax2)

    plt.tight_layout()
    return fig


def plot_spectrogram_with_psd(spectrogram, stage_name, fs=100, nfft=256):
    """Spectrogram (left) + time-averaged power with band annotations (right)."""
    spec = spectrogram.squeeze().detach().cpu().numpy() if hasattr(spectrogram, "detach") else np.asarray(spectrogram).squeeze()
    n_freq = spec.shape[1]
    freq_axis = np.linspace(0, fs / 2, n_freq)

    freq_max = 45  # cut off noise above 45 Hz

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 4.5),
        gridspec_kw={"width_ratios": [5, 2], "wspace": 0.08},
    )

    # Left: spectrogram
    ax1.imshow(spec.T, aspect="auto", origin="lower", cmap="viridis",
               extent=[0, 30, 0, fs / 2])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(f"Spectrogram — {stage_name}")
    ax1.set_ylim(0, freq_max)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: time-averaged power vs frequency (horizontal plot, shared Y)
    avg_power = spec.mean(axis=0)
    ax2.plot(avg_power, freq_axis, color="#2563EB", linewidth=1.2)
    _add_band_shading_h(ax2)
    ax2.set_xlabel("Avg Power (dB)")
    ax2.set_title(f"Mean Spectrum — {stage_name}")
    ax2.set_ylim(0, freq_max)
    ax2.set_yticklabels([])  # shared visual alignment with spectrogram
    ax2.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Band labels inside the band spans, right-aligned
    for band in DISPLAY_BANDS:
        mid_y = (band["low"] + band["high"]) / 2
        if mid_y <= freq_max:
            ax2.annotate(
                band["name"], xy=(0.92, mid_y),
                xycoords=("axes fraction", "data"),
                ha="center", va="center", fontsize=12, fontweight="bold",
                color=band["color"],
            )

    plt.tight_layout()
    return fig


def plot_pred_barplot(probs, true_label, model_name):
    """Bar chart of 5-class prediction probabilities."""
    p = probs.detach().cpu().numpy() if hasattr(probs, "detach") else np.asarray(probs)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(STAGE_NAMES, p, color=STAGE_COLORS,
                  edgecolor="#CCCCCC", linewidth=0.8, width=0.65)

    # Highlight true label with thick black border
    bars[true_label].set_edgecolor("black")
    bars[true_label].set_linewidth(2.5)

    # Percentage labels — ensure minimum height so small-bar labels don't collide
    for i, v in enumerate(p):
        label_y = max(v, 0.03) + 0.02
        ax.text(i, label_y, f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Probability")
    ax.set_ylim(0, min(1.0, p.max() * 1.3) + 0.08)
    ax.set_title(f"{model_name} — Prediction Confidence")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    _clean_spines(ax)

    plt.tight_layout()
    return fig


def plot_attribution_vs_psd(spectrogram, attribution, stage_name, fs=100):
    """Overlay mean |IG attribution| on the mean power spectrum per frequency bin."""
    spec = spectrogram.squeeze().detach().cpu().numpy() if hasattr(spectrogram, "detach") else np.asarray(spectrogram).squeeze()
    attr = attribution.squeeze().detach().cpu().numpy() if hasattr(attribution, "detach") else np.asarray(attribution).squeeze()

    n_freq = spec.shape[1]
    freq_axis = np.linspace(0, fs / 2, n_freq)

    avg_power = spec.mean(axis=0)
    avg_attr = np.abs(attr).mean(axis=0)

    freq_max = 35  # focus on physiologically relevant range

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    fig.subplots_adjust(top=0.82)

    # Band shading
    _add_band_shading_v(ax1, alpha=0.10)

    # Band labels — placed in a dedicated row above the axes, below the title
    for band in DISPLAY_BANDS:
        mid = (band["low"] + band["high"]) / 2
        if mid <= freq_max:
            ax1.annotate(
                band["name"], xy=(mid, 1.03),
                xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=band["color"], annotation_clip=False,
            )

    # Power spectrum (left axis)
    color_psd = "#2563EB"
    ax1.plot(freq_axis, avg_power, color=color_psd, linewidth=1.8, label="Avg Power (dB)")
    ax1.set_xlabel("Frequency (Hz)", fontsize=12)
    ax1.set_ylabel("Avg Power (dB)", color=color_psd, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_psd)
    ax1.set_xlim(0, freq_max)
    ax1.spines["top"].set_visible(False)

    # Attribution (right axis)
    ax2 = ax1.twinx()
    color_attr = "#E74C3C"
    ax2.fill_between(freq_axis, 0, avg_attr, color=color_attr, alpha=0.25)
    ax2.plot(freq_axis, avg_attr, color=color_attr, linewidth=1.8, label="Mean |IG Attribution|")
    ax2.set_ylabel("Mean |IG Attribution|", color=color_attr, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_attr)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Combined legend — compact, non-obtrusive
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=9, framealpha=0.85, edgecolor="none")

    ax1.set_title(f"Power Spectrum vs IG Attribution — {stage_name}",
                  fontsize=13, fontweight="bold", pad=25)

    fig.tight_layout(rect=[0, 0, 1, 0.82])
    return fig
