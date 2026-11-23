"""Evaluate the SeqSleepNet baselines on the MASS fold-0 test set, with the protocol of seqsleepnet-phan.

Baselines (weights from HF 4rooms/physioex, copied to --weights because Sofia is offline):
- ``seqsleepnet-phan``: 1 channel (raw EEG), L = 20, trained with examples/pretrained/seqsleepnet-phan/train.py.
- ``seqsleepnet-EEG-EOG-EMG-L=21``: 3 channels (raw EEG, EOG, EMG), physioex 1.x Lightning checkpoint, 1-layer
  epoch LSTM + 1-layer sequence GRU; its keys are mapped onto ``SeqSleepNet(in_chan=3, seqnlayer1=1, seqnlayer2=1)``.
  It was trained on the physioex 1.x split, which may share subjects with this test set.

Each MASS cohort is evaluated with ``Trainer.voting_evaluate`` (fold 0 test subjects); totals are pooled from the
confusion matrices.

    python examples/pretrained/channelnet-mass/eval_baselines.py --weights <dir> --root $DATA/MASS/Original --out <dir>
"""
import argparse
import json
import os

import numpy as np
import torch

from physioex.data.datasets import get_dataset
from physioex.models.seqsleepnet import SeqSleepNet
from physioex.train.trainer import Trainer


def from_cm(cm):
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    tp = np.diag(cm)
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
    return {"acc": po, "kappa": (po - pe) / (1 - pe), "macro_f1": f1.mean(), "f1_per_class": f1.tolist(), "n": n}


def load_baselines(wdir):
    cfg = json.load(open(os.path.join(wdir, "seqsleepnet-phan", "config.json")))
    phan = SeqSleepNet(**cfg["model_kwargs"])
    phan.load_state_dict(torch.load(os.path.join(wdir, "seqsleepnet-phan", "model.pt"), map_location="cpu"))
    ck = torch.load(os.path.join(wdir, "seqsleepnet-EEG-EOG-EMG-L=21-test_acc=0.877.ckpt"), map_location="cpu",
                    weights_only=False)["state_dict"]
    ren = [("nn.epoch_encoder.F2_filtbank.", "filterbank."), ("nn.epoch_encoder.F2_birnn.", "seqn1."),
           ("nn.epoch_encoder.F2_attention.", "attention."), ("nn.sequence_encoder.LSTM.", "seqn2."),
           ("nn.sequence_encoder.clf.", "clf.")]
    sd = {}
    for k, v in ck.items():
        for a, b in ren:
            if k.startswith(a):
                sd[b + k[len(a):]] = v
    tri = SeqSleepNet(in_chan=3, seqnlayer1=1, seqnlayer2=1)
    tri.load_state_dict(sd, strict=True)
    return {"seqsleepnet-phan_1ch": (phan, ["EEG"], 20), "seqsleepnet_3ch_L21": (tri, ["EEG", "EOG", "EMG"], 21)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    MASS = get_dataset("mass")
    res = {}
    for name, (model, chans, L) in load_baselines(args.weights).items():
        per, cm = {}, np.zeros((5, 5))
        for c in range(1, 6):
            ds = MASS(cohort=c, channels=chans, pipelines="seqsleepnet", sequence_length=L, root=args.root)
            r = Trainer.voting_evaluate(model=model, dataset=ds, L=L, fold=0, gpu_id=args.gpu_id)
            m = np.asarray(r["confusion_matrix"].cpu() if hasattr(r["confusion_matrix"], "cpu") else r["confusion_matrix"])
            cm += m
            per[f"mass_ss{c:02d}"] = {k: float(v) for k, v in from_cm(m).items() if k != "f1_per_class"}
            print(name, c, per[f"mass_ss{c:02d}"], flush=True)
        res[name] = {"total": {k: (float(v) if k != "f1_per_class" else v) for k, v in from_cm(cm).items()},
                     "per_dataset": per, "confusion": cm.tolist(), "channels": chans, "L": L}
        print(name, "TOTAL", res[name]["total"], flush=True)
    json.dump(res, open(os.path.join(args.out, "baselines.json"), "w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
