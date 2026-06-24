"""Executable version of the lecture notebook for debugging.

Run from the lecture directory:
    cd lectures/2026-06-24_ugent_xai
    python run_notebook.py
"""
import os
import sys
import traceback

# Non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

errors = []

def section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# === Cell 1: Setup ===
section("Cell 1: Setup")
try:
    import torch
    import numpy as np

    from util import (
        STAGE_NAMES, STAGE_COLORS,
        plot_hypnogram, plot_epoch_with_psd, plot_spectrogram_with_psd,
        plot_pred_barplot, plot_ig_raw, plot_ig_spectrogram,
        plot_attribution_vs_psd,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
except Exception as e:
    errors.append(("Cell 1: Setup", e))
    traceback.print_exc()


# === Cell 4: Load Sleep-EDF (raw) with N3 sample ===
section("Cell 4: Load Sleep-EDF (raw)")
try:
    from physioex.data.datasets import SleepEDFDataset, MASSDataset

    ds_raw = SleepEDFDataset(
        channels=["EEG"],
        pipelines="raw",
        sequence_length=20,
    )

    print(f"Sleep-EDF: {ds_raw.get_n_subjects()} subjects, {len(ds_raw)} sequences")

    # Find a sample with N3 epochs in mid-sequence
    sample_raw = None
    for i in range(0, min(20000, len(ds_raw))):
        s = ds_raw[i]
        labels = s["labels"]
        valid = labels[labels >= 0]
        n3_pos = (labels == 3).nonzero(as_tuple=True)[0]
        mid_n3 = [p.item() for p in n3_pos if 5 <= p.item() <= 15]
        if len(valid) >= 15 and len(valid.unique()) >= 3 and len(n3_pos) >= 3 and mid_n3:
            sample_raw = s
            print(f"Using sample index {i}, labels={labels.tolist()}")
            break
    assert sample_raw is not None, "No suitable N3 sample found"

    eeg_ch = sample_raw["channel_order"][0]
    print(f"Channel: {eeg_ch}")
    print(f"Signal shape: {sample_raw['signals'][eeg_ch].shape}")
    print(f"Labels: {sample_raw['labels'].tolist()}")
except Exception as e:
    errors.append(("Cell 4: Load Sleep-EDF", e))
    traceback.print_exc()


# === Cell 5: Raw N3 epoch + PSD ===
section("Cell 5: Raw N3 epoch + PSD")
try:
    # Pick a mid-sequence N3 epoch for visualization
    n3_pos_raw = (sample_raw["labels"] == 3).nonzero(as_tuple=True)[0]
    mid_n3_raw = [p.item() for p in n3_pos_raw if 5 <= p.item() <= 15]
    vis_epoch_raw = mid_n3_raw[len(mid_n3_raw) // 2]
    print(f"Visualizing epoch {vis_epoch_raw} (label=N3)")

    sig_epoch = sample_raw["signals"][eeg_ch][vis_epoch_raw]
    fig = plot_epoch_with_psd(sig_epoch, fs=100, stage_name="N3")
    fig.savefig("outputs/cell5_raw_n3_psd.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("Saved outputs/cell5_raw_n3_psd.png")
except Exception as e:
    errors.append(("Cell 5: Raw N3 PSD", e))
    traceback.print_exc()
    plt.close("all")


# === Cell 6: Load MASS (spectrogram) with N3 sample ===
section("Cell 6: Load MASS SS03 (seqsleepnet)")
try:
    ds_spec = MASSDataset(
        cohort=3,
        channels=["EEG"],
        pipelines="seqsleepnet",
        sequence_length=20,
    )

    print(f"MASS SS03: {ds_spec.get_n_subjects()} subjects, {len(ds_spec)} sequences")

    # Find a sample with many N3 epochs
    sample_spec = None
    for i in range(0, min(20000, len(ds_spec))):
        s = ds_spec[i]
        labels = s["labels"]
        valid = labels[labels >= 0]
        n3_count = (labels == 3).sum().item()
        if len(valid) >= 15 and len(valid.unique()) >= 3 and n3_count >= 5:
            sample_spec = s
            print(f"Using sample index {i}, labels={labels.tolist()}")
            break
    assert sample_spec is not None, "No suitable N3 sample found"

    eeg_ch_spec = sample_spec["channel_order"][0]
    print(f"Channel: {eeg_ch_spec}")
    print(f"Signal shape: {sample_spec['signals'][eeg_ch_spec].shape}")
    print(f"Labels: {sample_spec['labels'].tolist()}")
except Exception as e:
    errors.append(("Cell 6: Load MASS", e))
    traceback.print_exc()


# === Cell 7: Spectrogram N3 epoch + PSD ===
section("Cell 7: Spectrogram N3 epoch + PSD")
try:
    # Pick a mid-sequence N3 epoch
    n3_pos_spec = (sample_spec["labels"] == 3).nonzero(as_tuple=True)[0]
    vis_epoch_spec = n3_pos_spec[len(n3_pos_spec) // 2].item()
    print(f"Visualizing epoch {vis_epoch_spec} (label=N3)")

    spec_epoch = sample_spec["signals"][eeg_ch_spec][vis_epoch_spec]
    fig = plot_spectrogram_with_psd(spec_epoch, stage_name="N3")
    fig.savefig("outputs/cell7_spec_n3_psd.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("Saved outputs/cell7_spec_n3_psd.png")
except Exception as e:
    errors.append(("Cell 7: Spectrogram N3 PSD", e))
    traceback.print_exc()
    plt.close("all")


# === Cell 11: TinySleepNet inference ===
section("Cell 11: TinySleepNet inference")
try:
    from physioex.models import load_from_pretrained

    tiny = load_from_pretrained("tinysleepnet-supratak", device=device, verbose=True)

    x_raw = sample_raw["signals"][eeg_ch].unsqueeze(0).unsqueeze(2).to(device)
    print(f"Input shape: {x_raw.shape}")

    with torch.no_grad():
        logits = tiny(x_raw)
        probs_raw = logits.softmax(-1).squeeze(0).cpu()
        preds = probs_raw.argmax(-1)

    print(f"Preds: {preds.tolist()}")
    print(f"True:  {sample_raw['labels'].tolist()}")

    fig = plot_hypnogram(sample_raw["labels"], pred_labels=preds,
                         title="TinySleepNet on Sleep-EDF (in-domain)")
    fig.savefig("outputs/cell11_tiny_hypnogram.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Select highest-confidence correctly-predicted N3 epoch
    n3_mask = (sample_raw["labels"] == 3) & (preds == 3)
    n3_conf = probs_raw[:, 3] * n3_mask.float()
    epoch_idx_tiny = n3_conf.argmax().item()
    pred_label_tiny = preds[epoch_idx_tiny].item()
    conf_tiny = probs_raw[epoch_idx_tiny, pred_label_tiny].item()
    print(f"Target N3 epoch: {epoch_idx_tiny}, pred={STAGE_NAMES[pred_label_tiny]}, conf={conf_tiny:.4f}")

    fig = plot_pred_barplot(probs_raw[epoch_idx_tiny], true_label=3, model_name="TinySleepNet")
    fig.savefig("outputs/cell11_tiny_barplot.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("Saved outputs/cell11_tiny_hypnogram.png + cell11_tiny_barplot.png")
except Exception as e:
    errors.append(("Cell 11: TinySleepNet", e))
    traceback.print_exc()
    plt.close("all")


# === Cell 13: SeqSleepNet inference ===
section("Cell 13: SeqSleepNet inference")
try:
    seq = load_from_pretrained("seqsleepnet-phan", device=device, verbose=True)

    x_spec = sample_spec["signals"][eeg_ch_spec].unsqueeze(0).unsqueeze(2).to(device)
    print(f"Input shape: {x_spec.shape}")

    with torch.no_grad():
        logits = seq(x_spec)
        probs_spec = logits.softmax(-1).squeeze(0).cpu()
        preds_seq = probs_spec.argmax(-1)

    print(f"Preds: {preds_seq.tolist()}")
    print(f"True:  {sample_spec['labels'].tolist()}")

    fig = plot_hypnogram(sample_spec["labels"], pred_labels=preds_seq,
                         title="SeqSleepNet on MASS SS03 (in-domain)")
    fig.savefig("outputs/cell13_seq_hypnogram.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Select highest-confidence correctly-predicted N3 epoch
    n3_mask_s = (sample_spec["labels"] == 3) & (preds_seq == 3)
    n3_conf_s = probs_spec[:, 3] * n3_mask_s.float()
    epoch_idx_seq = n3_conf_s.argmax().item()
    pred_label_seq = preds_seq[epoch_idx_seq].item()
    conf_seq = probs_spec[epoch_idx_seq, pred_label_seq].item()
    print(f"Target N3 epoch: {epoch_idx_seq}, pred={STAGE_NAMES[pred_label_seq]}, conf={conf_seq:.4f}")

    fig = plot_pred_barplot(probs_spec[epoch_idx_seq], true_label=3, model_name="SeqSleepNet")
    fig.savefig("outputs/cell13_seq_barplot.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("Saved outputs/cell13_seq_hypnogram.png + cell13_seq_barplot.png")
except Exception as e:
    errors.append(("Cell 13: SeqSleepNet", e))
    traceback.print_exc()
    plt.close("all")


# === Cell 17: IG on TinySleepNet ===
section("Cell 17: IG on TinySleepNet (raw N3)")
try:
    from physioex.explain.posthoc import IntegratedGradients

    print(f"Epoch {epoch_idx_tiny}: true={STAGE_NAMES[sample_raw['labels'][epoch_idx_tiny].item()]}, "
          f"predicted={STAGE_NAMES[pred_label_tiny]}, confidence={conf_tiny:.4f}")

    def tiny_score(x_epoch):
        x = x_epoch.unsqueeze(1).unsqueeze(1)
        logits = tiny(x)
        return logits[:, 0, :].softmax(-1)[:, pred_label_tiny]

    ig = IntegratedGradients(f=tiny_score, steps=64, expects_batch=True)

    torch.backends.cudnn.enabled = False
    x_epoch = sample_raw["signals"][eeg_ch][epoch_idx_tiny].unsqueeze(0).to(device)
    attr = ig(x_epoch)
    torch.backends.cudnn.enabled = True
    print(f"attr shape: {attr.shape}")

    fig = plot_ig_raw(x_epoch, attr, pred_label_tiny, epoch_idx_tiny, model_name="TinySleepNet")
    fig.savefig("outputs/cell17_ig_tiny.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("Saved outputs/cell17_ig_tiny.png")
except Exception as e:
    errors.append(("Cell 17: IG TinySleepNet", e))
    traceback.print_exc()
    plt.close("all")


# === Cell 19: IG on SeqSleepNet ===
section("Cell 19: IG on SeqSleepNet (spectrogram N3)")
try:
    print(f"Epoch {epoch_idx_seq}: true={STAGE_NAMES[sample_spec['labels'][epoch_idx_seq].item()]}, "
          f"predicted={STAGE_NAMES[pred_label_seq]}, confidence={conf_seq:.4f}")

    def seq_score(x_epoch):
        x = x_epoch.unsqueeze(1).unsqueeze(1)
        logits = seq(x)
        return logits[:, 0, :].softmax(-1)[:, pred_label_seq]

    ig_seq = IntegratedGradients(f=seq_score, steps=64, expects_batch=True)

    torch.backends.cudnn.enabled = False
    x_epoch_s = sample_spec["signals"][eeg_ch_spec][epoch_idx_seq].unsqueeze(0).to(device)

    # Baseline = noise floor: mean power above 40 Hz (outside the 0.3-40 Hz bandpass)
    n_freq = x_epoch_s.shape[-1]  # 129 bins, 0-50 Hz
    bin_40hz = int(40.0 / 50.0 * n_freq)  # ~103
    noise_floor = x_epoch_s[..., bin_40hz:].mean()
    baseline_s = torch.full_like(x_epoch_s, noise_floor.item())
    print(f"Noise-floor baseline: {noise_floor.item():.2f} dB (mean of bins > 40 Hz)")

    attr_s = ig_seq(x_epoch_s, baseline=baseline_s)
    torch.backends.cudnn.enabled = True
    print(f"attr_s shape: {attr_s.shape}")

    fig = plot_ig_spectrogram(x_epoch_s, attr_s,
                              sample_spec["labels"][epoch_idx_seq].item(),
                              pred_label_seq, model_name="SeqSleepNet")
    fig.savefig("outputs/cell19_ig_seq.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("Saved outputs/cell19_ig_seq.png")

    # Attribution vs PSD overlay
    fig = plot_attribution_vs_psd(x_epoch_s, attr_s, stage_name="N3")
    fig.savefig("outputs/cell19b_attr_vs_psd.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("Saved outputs/cell19b_attr_vs_psd.png")
except Exception as e:
    errors.append(("Cell 19: IG SeqSleepNet", e))
    traceback.print_exc()
    plt.close("all")


# === Summary ===
section("SUMMARY")
if errors:
    print(f"\n{len(errors)} ERROR(S):")
    for name, err in errors:
        print(f"  FAIL: {name} — {type(err).__name__}: {err}")
    sys.exit(1)
else:
    print("\nAll cells executed successfully!")
    print("Output files:")
    for f in sorted(os.listdir("outputs")):
        size = os.path.getsize(f"outputs/{f}")
        print(f"  outputs/{f}  ({size:,} bytes)")
    sys.exit(0)
