# SleepTokenizer Pretraining Tests

Test funzionali per verificare il pretraining di SleepTokenizer su dataset singoli e multipli.

## Struttura

```
tests/
├── __init__.py
├── test_single_dataset.py   # Test su MASS SS1 (singolo dataset)
├── test_multi_dataset.py    # Test su combinazione di dataset
└── README.md
```

## Dipendenze

Tutti i test importano costanti e funzioni da `../pretrain.py`:
- `MODEL_KWARGS`, `TRAIN_CONFIG`
- `_sleeptokenizer_step`, `_sleeptokenizer_voting_eval_step`
- `_compute_channel_loss`

Non c'è duplicazione di codice — tutto è riutilizzato.

## Test Singolo Dataset

```bash
# Solo inferenza modalità (veloce, no GPU)
python -m examples.pretrained.protosleepnet_gagliardi.tests.test_single_dataset --skip_training

# Test completo con 1 epoch di training
python -m examples.pretrained.protosleepnet_gagliardi.tests.test_single_dataset --gpu_id 0

# Output custom
python -m examples.pretrained.protosleepnet_gagliardi.tests.test_single_dataset \
    --gpu_id 0 --output_dir /path/to/output
```

**Verifiche**:
- `build_modality_ids` classifica correttamente canali MASS
- Script parte senza errori
- Logging mostra `loss_main` e `loss_chan` separati
- Checkpoint salvato correttamente

## Test Multi-Dataset

```bash
# Analisi eterogeneità canali (no training)
python -m examples.pretrained.protosleepnet_gagliardi.tests.test_multi_dataset --skip_training

# Test con MASS + SleepEDF (default)
python -m examples.pretrained.protosleepnet_gagliardi.tests.test_multi_dataset --gpu_id 0

# Test con datasets specifici
python -m examples.pretrained.protosleepnet_gagliardi.tests.test_multi_dataset \
    --gpu_id 0 --datasets mass sleptedf hmc

# Test modality dropout safety
python -m examples.pretrained.protosleepnet_gagliardi.tests.test_multi_dataset \
    --gpu_id 0 --test_dropout --skip_training
```

**Verifiche**:
- MultiDataset combina datasets con canali diversi
- `data_mask` rileva canali assenti
- `build_modality_ids` gestisce naming eterogeneo
- Training non crasha su batch eterogenei
- Modality dropout mai droppa tutti i canali (safety check)

## Output

I risultati sono salvati in `tests/output/`:
- `test_mass_ss1/` — output test singolo dataset
- `test_multi_mass_sleeptedf/` — output test multi-dataset
- `checkpoints/` — checkpoint per epoch

## Risoluzione Problemi

**Dataset non trovato**: Verifica che i dati siano nel path corretto o usa `--dataset_root`

**GPU memory**: Riduci `batch_size` nel codice (default 8 per test)

**Import errors**: Assicurati di essere nella root del repo o usa `python -m` syntax
