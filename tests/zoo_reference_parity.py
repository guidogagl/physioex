"""Parity of the pure-torch zoo blocks with the reference implementations (xmai-lab/sleep-mechinterp#39).

The reference packages are not installed with physioex (nor on Sofia): run this script in a separate venv with
``torch``, ``mambapy`` and ``xlstm`` (CPU is enough), from the repository root:

    ~/Code/.venv-zoo-ref/bin/python tests/zoo_reference_parity.py

Random weights are mapped 1:1 into the reference modules / functions and the outputs compared (atol 1e-4):

- ``MambaMixer``  vs ``mambapy.mamba.MambaBlock`` (parallel scan), same parameter names
- ``mlstm_parallel`` vs ``xlstm.blocks.mlstm.backends.parallel_stabilized_simple``
- ``SLSTMCell``  vs ``xlstm.blocks.slstm.src.vanilla.slstm_forward`` with the sLSTM pointwise function
  (gate order i, f, z, o and the recurrent matrix layout converted)
"""

import importlib.util
import sys
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("zoo_blocks", Path(__file__).parents[1] / "physioex/models/zoo_blocks.py")
zb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zb)                      # no physioex import: the reference venv only has torch

from mambapy.mamba import MambaBlock, MambaConfig                                    # noqa: E402
from xlstm.blocks.mlstm.backends import parallel_stabilized_simple                   # noqa: E402
from xlstm.blocks.slstm.src.vanilla import slstm_forward                             # noqa: E402
from xlstm.blocks.slstm.src.vanilla.slstm import slstm_forward_pointwise             # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
results = {}

# --- Mamba --------------------------------------------------------------------------------------------------------
d = 32
mine = zb.MambaMixer(d, d_state=16, expand=2, d_conv=4)
ref = MambaBlock(MambaConfig(d_model=d, n_layers=1, d_state=16, expand_factor=2, d_conv=4, bias=False,
                             conv_bias=True, pscan=True))
sd = {k.replace("conv.", "conv1d."): v for k, v in mine.state_dict().items()}
missing = ref.load_state_dict(sd, strict=True)
x = torch.randn(3, 29, d)
results["mamba"] = (mine(x) - ref(x)).abs().max().item()

# --- mLSTM ---------------------------------------------------------------------------------------------------------
q, k, v = (torch.randn(2, 4, 29, 16) for _ in range(3))
ip, fp = torch.randn(2, 4, 29, 1), torch.randn(2, 4, 29, 1) + 2.0
results["mlstm"] = (zb.mlstm_parallel(q, k, v, ip, fp) - parallel_stabilized_simple(q, k, v, ip, fp)).abs().max().item()

# --- sLSTM ---------------------------------------------------------------------------------------------------------
H, NH, din, T, Bn = 32, 4, 24, 20, 3
Dh = H // NH
cell = zb.SLSTMCell(din, H, NH)
with torch.no_grad():
    cell.R.normal_(0, 0.5)
    cell.W.bias.normal_(0, 0.5)
x = torch.randn(Bn, T, din)
y_mine = cell(x)                                                     # (B, T, H)
order = [1, 2, 0, 3]                                                 # mine z, i, f, o -> ref i, f, z, o
Wx = torch.nn.functional.linear(x, cell.W.weight).view(Bn, T, 4, H)[:, :, order].reshape(Bn, T, 4 * H)
b = cell.W.bias.view(4, H)[order].reshape(-1)
# ref R[h] (4*Dh, Dh): Ry[g, h, e] = sum_d y[h, d] R[h][g*Dh + e, d]; mine R[g, h, d, e]
R = cell.R[order].permute(1, 0, 3, 2).reshape(NH, 4 * Dh, Dh)
states = torch.zeros(4, Bn, H)
all_states, _, _ = slstm_forward(Wx.transpose(0, 1), states, R, b, slstm_forward_pointwise)
y_ref = all_states[0, 1:].transpose(0, 1)
results["slstm"] = (y_mine - y_ref).abs().max().item()

ok = all(v < 1e-4 for v in results.values())
for name, err in results.items():
    print(f"{name:6s} max |mine - ref| = {err:.2e}")
print("PARITY OK" if ok else "PARITY FAILED")
sys.exit(0 if ok else 1)
