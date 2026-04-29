"""Vendored encoder architectures not available in braindecode.

These are self-contained PyTorch modules copied from their original
repositories. They have no external dependencies beyond torch and einops.
"""
from .sleepfm_encoder import SetTransformer
from .tfc_encoder import TFC, TFCConfig, load_pretrained_tfc, to_freq_domain
from .neurolm_encoder import NeuralTransformer, NTConfig
