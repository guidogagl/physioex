# xLSTM in SeqSleepNet — experiment runner

Research question: *is the hierarchical fold/unfold of L-SeqSleepNet necessary, or
does a flat mLSTM (matrix memory, exponential gating) recover it?* Plus: causal
single-pass whole-night inference (SeqSleepNet's stated limitation #1).

`run.py` executes one `(model, dataset, fold, seed)` cell and writes pooled metrics,
subject-level statistics with bootstrap CIs and per-subject predictions for paired
comparisons. See its docstring for the protocols (MASS 20-fold as Phan 2019,
SleepEDF-SC 2013 LOSO as Phan 2023, SHHS fixed split) and examples.

## Stages (Phase 1 = MASS)

| Stage | Runs | Gate |
|---|---|---|
| 0a parity | `--model seqsleepnet --channels EEG EOG EMG --L 20` | ~87.0 acc / κ 0.815 (pooled, Phan 2019) |
| 0b competitor | `--dataset sleepedf --subset 2013 --model lseqsleepnet --L 200 --n_valid 4` | 86.3 ± 0.2 / κ 0.813 / MF1 79.3 (Phan 2023) |
| 1 controls | `xseqsleepnet` with `gru`, `gru_wrapped`, `gru_matched`, `xlstm_bi` at L=20; `--epoch_encoder xlstm` | null expected |
| 2 main | L ∈ {20, 100, 200, 400}: `gru`, `gru_wrapped`, `gru_matched`, `xlstm_bi`, `lseqsleepnet` | pre-registered κ criterion |
| 3 baselines | S4D / Mamba sequence encoders (to add) | — |
| 4 causal | `xlstm_causal` vs `gru_uni`, `--eval_modes voting single_pass`; train L=20, test whole night | — |

Development: folds 0–2 × seeds 0–2. Final: 20 folds × 1 seed for the surviving
configurations. Primary metric κ; report per-subject paired differences, not pooled deltas.

`gru_matched` sizes the wrapped BiGRU to the parameter count of `xlstm_bi` built
with the same `--seq_kwargs`; pass `target_params` explicitly to pin it.
