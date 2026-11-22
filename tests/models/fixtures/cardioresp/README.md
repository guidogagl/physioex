# Golden outputs for cardiorespiratory sleep-staging models

Produced once on CPU by running the authors' own code with their released weights
(`make_goldens.py` kept outside the repo, `~/work/data/papers/ppg-ecg-sleep-staging/ref/`).
Inputs are regenerated in tests from `numpy.random.default_rng(seed)`; only outputs are stored.
E = 60 epochs of 30 s for every model.

| file | model / source | input recipe | stored output |
|---|---|---|---|
| `wav2sleep.npz` | HF `joncarter/wav2sleep` (`config.yaml` copied as `wav2sleep.config.yaml`) | signals in order `signals` = [ABD, ECG, PPG, THX]; signal *i* = `default_rng(100+i).standard_normal(spe*60).astype(float32)`, spe = 256 (ABD/THX) or 1024 (ECG/PPG), shape (1, spe*60); an absent signal = `full((1, spe*60), -inf)` | `logits_all`, `logits_ecg` (ECG only), `logits_ppg` (PPG only), `logits_ecg_thx`: (1, 60, 4) logits, classes [W, N1+N2, N3, REM] |
| `wav2sleep-eog.npz` | HF `joncarter/wav2sleep-eog` | signals [EOG-L, EOG-R], spe 4096, seeds 100, 101 | `logits_all` (1, 60, 5) |
| `sleepppgnet-replica.npz` | HF `sleep-is-all-you-need/SPPPPG-Net`, class `SleepPPGNetTorch` of SPPPPGNet 0.1.4 | `default_rng(200).standard_normal(1228800).astype(float32)` shaped (1, 1, 1228800) | `logits` (1, 4, 1200), class order [Wake, REM, Light, Deep] |
| `jones-ecg.npz` | `adammj/ecg-sleep-staging`, 7 checkpoints, keys `<group>/<variant>` | ecg = `clip(default_rng(300).standard_normal(60*7680).astype(float32)*0.2, -1, 1).reshape(1,60,1,7680)`; additional = `default_rng(301).random((1,60,4)).astype(float32)`; padding_eliminator = ones (1,60,1); `SleepNet.forward(ecg, additional, padding_eliminator)[0]` | softmax probabilities (1, 60, 5), classes [W, N1, N2, N3, REM] |
| `state_dict_shapes.json` | all checkpoints above | — | `{name: {key: shape}}` of the original state dicts (Jones: `ckpt["model_state_dict"]`) |
