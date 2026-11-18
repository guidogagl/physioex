"""The psg_gen models behind the ``Codec`` interface of ``concept_counterfactual`` / ``concept_leak``.

The generative decoder does not take latents: it takes a decomposition (events + background
levels). The adapter keeps the decomposition of every encoded epoch in a store and hands out
*references* in place of the latents, so the swaps of ``concept_leak`` become swaps of components:

* ``N`` (B, 1) = id of the epoch whose night gain is used (``N_other_night`` = gain of another night);
* ``E`` (B, 1) = id of the epoch whose background levels (logP) are used (``E_*`` = background swap);
* ``P`` (B, 60, 2) = [id, token] of the epoch whose events are used (``P_other_stage`` = events swap);
  ``P_perm`` permutes the 60 tokens, and every event moves with its 0.5 s token.

The gain enters as in P1 (R7): event amplitudes x g_N / g_P, background log10 power + 2 log10(g_N / g_E).
With the three ids equal the decode is the plain :meth:`PSGGenBase.decode` of that epoch.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from physioex.models.psg_gen.inference import Events


class GenCodec:
    def __init__(self, combo_dir, dev, mod="eeg"):
        from physioex.train.bin.pretrain_combo import load_s2

        run, self.m = load_s2(os.path.join(combo_dir, mod, "s2"), mod, dev)
        self.half = run["args"]["seq"] // 2
        self.dev = dev
        self.tok_sec = 30.0 / self.m.n_tok
        self.store: list[dict] = []          # per epoch: events (epoch = 0), logP (1, W, nb), log10 g

    def encode(self, xp, centres, batch=128):
        ids = []
        with torch.no_grad():
            for j in range(0, len(centres), batch):
                tb = centres[j:j + batch]
                w = torch.from_numpy(np.stack([xp[t - self.half:t + self.half + 1] for t in tb]).astype(np.float32))
                dec = self.m.encode(w[:, :, None].to(self.dev))
                ev: Events = dec["events"]
                for b in range(len(tb)):
                    e = ev.select(ev.epoch == b)
                    e.epoch = torch.zeros_like(e.epoch)
                    ids.append(len(self.store))
                    self.store.append({"events": e, "logP": dec["logP"][b:b + 1].float(),
                                       "lg": float(dec["N"][b, 0])})
        idx = torch.tensor(ids, dtype=torch.float32)
        tok = torch.arange(self.m.n_tok, dtype=torch.float32)
        P = torch.stack([idx[:, None].expand(-1, self.m.n_tok), tok[None].expand(len(ids), -1)], -1)
        return {"N": idx[:, None], "E": idx[:, None].clone(), "P": P}

    def decode(self, n, e, p, seed=0):
        out = []
        for i in range(len(n)):
            sn, se = self.store[int(n[i, 0])], self.store[int(e[i, 0])]
            sp = self.store[int(p[i, 0, 0])]
            ev = sp["events"].select(torch.ones(len(sp["events"]), dtype=torch.bool, device=sp["events"].epoch.device))
            # token permutation: position t now holds old token p[i, t, 1]; an event in old token o moves
            # to the position t with p[i, t, 1] == o, by (t - o) tokens
            old = p[i, :, 1].long()
            if len(ev) and not torch.equal(old, torch.arange(len(old))):
                newpos = torch.empty_like(old)
                newpos[old] = torch.arange(len(old))
                o = (ev.onset / self.tok_sec).long().clamp(0, len(old) - 1).cpu()
                ev.onset = ev.onset + ((newpos[o] - o).float() * self.tok_sec).to(ev.onset.device)
            ev.amp = ev.amp * 10.0 ** (sn["lg"] - sp["lg"])
            dec = {"events": ev, "logP": se["logP"] + 2.0 * (sn["lg"] - se["lg"])}
            with torch.no_grad():
                out.append(self.m.decode(dec, seed=seed)[0].float().cpu().numpy())
        return np.stack(out)

    def requant(self, p):
        return p
