"""Embedding extraction and linear probe training for foundation models.

Two main entry points:

1. ``extract_embeddings(model, dataset)`` — extract and cache embeddings
2. ``train_probe(model, dataset)`` — train a linear probe on cached embeddings

Both work as functions and as CLI scripts::

    # Function API
    from physioex.models.foundation.embed import extract_embeddings, train_probe
    extract_embeddings("cbramod", "hmc")
    results = train_probe("cbramod", "hmc", fold=0)

    # CLI
    python -m physioex.models.foundation.embed.extract --model cbramod --dataset hmc
    python -m physioex.models.foundation.embed.probe --model cbramod --dataset hmc
"""
from physioex.models.foundation.embed.extract import extract_embeddings
from physioex.models.foundation.embed.dataset import (
    EmbeddingDataset,
    embedding_collate_fn,
)
from physioex.models.foundation.embed.probe import LinearProbe, train_probe

__all__ = [
    "extract_embeddings",
    "EmbeddingDataset",
    "embedding_collate_fn",
    "LinearProbe",
    "train_probe",
]
