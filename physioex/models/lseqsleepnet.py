"""L-SeqSleepNet: whole-cycle long sequence modelling for sleep staging.

Paper: Phan et al., "L-SeqSleepNet: Whole-cycle Long Sequence Modelling for
Automatic Sleep Staging", IEEE JBHI 2023 (arXiv:2301.03441).

Two implementations live in this module:

``LSeqSleepNet``
    Paper-compliant model (Sec. III of the paper):

    1. Epoch encoder (Sec. III-B), shared across the L epochs:
       learnable filterbank (F=129 -> M=32) -> BLSTM **with recurrent batch
       normalization** (He/2 = 64 per direction) -> additive attention
       (Eqs. 2-4, A=64) -> epoch embedding x in R^{He}.
    2. Long sequence modelling (Sec. III-C): fold L = B x K, then
       intra-subsequence BLSTM_ss over each row of K, each output vector
       o~ passed through ``o_bar = o~ + LN(W o~ + b)`` (Eq. 9), then
       inter-subsequence BLSTM_ws over each column of B with the same
       fc + LN + residual (Eq. 12), then unfold (Eq. 14).
    3. Classification (Sec. III-D): two fc layers of N_fc = 512 units with
       ReLU, then the output layer. Dropout 0.1 on the LSTM cells and on the
       fc layers (Sec. IV-B).

``LSeqSleepNetDraft``
    The previous, non-compliant draft kept for backward compatibility with
    the ``lseqsleepnet-phan`` checkpoint on the Hub. It differs from the
    paper in the residual/LayerNorm order (post-LN on ``fc(x) + input``
    instead of ``x~ + LN(fc(x~))`` on the BLSTM output), has no recurrent
    batch normalization, and silently truncates sequences longer than
    ``B*K``. Do not use it as a baseline; it is deprecated.

PhysioEx integration (both classes):
    Input:  (batch, L, C, T, F) spectrograms from the ``"seqsleepnet"`` preset
    Output: (batch, L, n_classes) per-epoch logits (sequence-to-sequence)
    Default L = B*K = 200 epochs (~100 min, roughly one sleep cycle).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from physioex.models.seqsleepnet import AttentionLayer, LearnableFilterbank


# ---------------------------------------------------------------------------
# Recurrent batch normalization LSTM (Cooijmans et al., ICLR 2017)
# ---------------------------------------------------------------------------


class BNLSTMCell(nn.Module):
    """LSTM cell with recurrent batch normalization.

    Implements the "shared statistics" variant used by the SeqSleepNet family
    (Phan's TensorFlow ``BNLSTMCell``): the input-to-hidden and hidden-to-hidden
    pre-activations are batch-normalized separately (no shift, shared bias
    ``b``), and the cell state is batch-normalized before the output tanh::

        gates = BN(W_x x_t; gamma_x) + BN(W_h h_{t-1}; gamma_h) + b
        c_t   = sigmoid(f) * c_{t-1} + sigmoid(i) * tanh(g)
        h_t   = sigmoid(o) * tanh(BN(c_t; gamma_c, beta_c))

    Statistics are shared across time steps (running averages updated at every
    step during training, used at evaluation). ``gamma`` is initialised to 0.1
    as recommended by Cooijmans et al. to avoid saturating the gates; the forget
    gate bias is initialised to 1.
    """

    def __init__(self, input_size: int, hidden_size: int, gamma_init: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        H = hidden_size

        self.weight_ih = nn.Parameter(torch.empty(4 * H, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * H, H))
        self.bias = nn.Parameter(torch.zeros(4 * H))

        # No learnable shift on the pre-activation BNs: the shared bias plays
        # that role (Cooijmans et al., Eq. 6-7).
        self.bn_ih = nn.BatchNorm1d(4 * H, affine=True)
        self.bn_hh = nn.BatchNorm1d(4 * H, affine=True)
        self.bn_c = nn.BatchNorm1d(H, affine=True)

        self.reset_parameters(gamma_init)

    def reset_parameters(self, gamma_init: float = 0.1) -> None:
        nn.init.orthogonal_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        with torch.no_grad():
            self.bias.zero_()
            H = self.hidden_size
            self.bias[H : 2 * H].fill_(1.0)  # forget gate
            for bn in (self.bn_ih, self.bn_hh, self.bn_c):
                bn.weight.fill_(gamma_init)
                bn.bias.zero_()
            self.bn_ih.bias.requires_grad_(False)
            self.bn_hh.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor, state=None):
        """x: (N, S, input_size) -> outputs (N, S, H)."""
        N, S, _ = x.shape
        H = self.hidden_size
        if state is None:
            h = x.new_zeros(N, H)
            c = x.new_zeros(N, H)
        else:
            h, c = state

        # Input projection for all steps at once; BN statistics over (N*S).
        xh = torch.matmul(x, self.weight_ih.t())  # (N, S, 4H)
        xh = self.bn_ih(xh.reshape(N * S, 4 * H)).reshape(N, S, 4 * H)

        outputs = []
        for t in range(S):
            hh = self.bn_hh(torch.matmul(h, self.weight_hh.t()))
            gates = xh[:, t] + hh + self.bias
            i, f, g, o = gates.chunk(4, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(self.bn_c(c))
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)


class BiLSTM(nn.Module):
    """Bidirectional LSTM with optional recurrent batch normalization.

    ``recurrent_bn=True`` uses :class:`BNLSTMCell` (paper-compliant, Python
    time loop); ``recurrent_bn=False`` falls back to cuDNN ``nn.LSTM`` (fast,
    used as an ablation). Dropout is applied to the cell inputs and outputs
    (TensorFlow ``DropoutWrapper`` style), matching "dropout applied to the
    LSTM cells" in Sec. IV-B of the paper.

    Input:  (N, S, input_size)
    Output: (N, S, 2*hidden_size)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        recurrent_bn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.recurrent_bn = recurrent_bn
        self.output_size = 2 * hidden_size
        self.drop_in = nn.Dropout(dropout)
        self.drop_out = nn.Dropout(dropout)
        if recurrent_bn:
            self.fwd = BNLSTMCell(input_size, hidden_size)
            self.bwd = BNLSTMCell(input_size, hidden_size)
        else:
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_in(x)
        if self.recurrent_bn:
            out_f, _ = self.fwd(x)
            out_b, _ = self.bwd(torch.flip(x, dims=[1]))
            out = torch.cat([out_f, torch.flip(out_b, dims=[1])], dim=-1)
        else:
            out, _ = self.rnn(x)
        return self.drop_out(out)


# ---------------------------------------------------------------------------
# Paper-compliant L-SeqSleepNet
# ---------------------------------------------------------------------------


class PaperEpochEncoder(nn.Module):
    """Epoch encoder of Sec. III-B: filterbank -> BN-BLSTM -> attention.

    Input:  (N, C, T, F)
    Output: (N, He) with He = 2 * hidden_size
    """

    def __init__(
        self,
        in_chan: int = 1,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        sf: int = 100,
        lowfreq: int = 0,
        highfreq: int = 50,
        hidden_size: int = 64,
        attention_size: int = 64,
        recurrent_bn: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.filterbank = LearnableFilterbank(in_chan, F, D, nfft, sf, lowfreq, highfreq)
        self.blstm = BiLSTM(D * in_chan, hidden_size, recurrent_bn=recurrent_bn, dropout=dropout)
        # Eqs. (2)-(4): u_t = tanh(W_a x~_t + b_a), w_t = softmax(u_t^T a),
        # x = sum_t w_t x~_t -- the same additive attention as SeqSleepNet's layer.
        self.attention = AttentionLayer(2 * hidden_size, attention_size)
        self.output_size = 2 * hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, F = x.shape
        x = self.filterbank(x)  # (N, C, T, D)
        x = x.permute(0, 2, 1, 3).reshape(N, T, -1)  # (N, T, C*D)
        x = self.blstm(x)  # (N, T, 2H)
        return self.attention(x)  # (N, 2H)


class _ResidualLNBlock(nn.Module):
    """``o_bar = o~ + LN(W o~ + b)`` -- Eqs. (9) and (12) of the paper.

    The residual is taken from the *BLSTM output* ``o~`` and the LayerNorm wraps
    only the fc projection. Dropout (``"applied to the fc layers"``) acts on the
    fc output before normalization.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)

    def forward(self, o_tilde: torch.Tensor) -> torch.Tensor:
        return o_tilde + self.ln(self.drop(self.fc(o_tilde)))


class LongSequenceModel(nn.Module):
    """Fold -> intra-subsequence -> inter-subsequence -> unfold (Sec. III-C).

    Input:  (batch, L, d_model) with L == B*K
    Output: (batch, L, d_model)

    Sequences shorter than ``B*K`` are zero-padded and trimmed back (an
    evaluation convenience for short recordings, not defined by the paper -- a
    warning is emitted). Sequences longer than ``B*K`` raise: the paper
    requires ``L = B x K`` and truncating would silently drop epochs.
    """

    def __init__(
        self,
        d_model: int = 128,
        B: int = 10,
        K: int = 20,
        hidden_ss: int = 64,
        hidden_ws: int = 64,
        recurrent_bn: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if 2 * hidden_ss != d_model or 2 * hidden_ws != d_model:
            # Eqs. (9)/(12) use square fc matrices W in R^{H x H} and add the
            # residual to the BLSTM output, so H_ss = H_ws = d_model.
            raise ValueError(
                f"L-SeqSleepNet requires 2*hidden_ss == 2*hidden_ws == d_model "
                f"(got {hidden_ss}, {hidden_ws}, {d_model})"
            )
        self.B, self.K, self.d_model = B, K, d_model
        self.blstm_ss = BiLSTM(d_model, hidden_ss, recurrent_bn=recurrent_bn, dropout=dropout)
        self.res_ss = _ResidualLNBlock(d_model, dropout)
        self.blstm_ws = BiLSTM(d_model, hidden_ws, recurrent_bn=recurrent_bn, dropout=dropout)
        self.res_ws = _ResidualLNBlock(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, L, D = x.shape
        B, K = self.B, self.K
        target = B * K
        if L > target:
            raise ValueError(
                f"L-SeqSleepNet expects sequences of exactly B*K={target} epochs, got L={L}. "
                f"Set sequence_length={target} (or change B/K)."
            )
        if L < target:
            warnings.warn(
                f"L-SeqSleepNet: padding sequence of L={L} epochs to B*K={target} with zeros.",
                stacklevel=2,
            )
            x = torch.cat([x, x.new_zeros(batch, target - L, D)], dim=1)

        # (1) fold: (batch, B*K, D) -> (batch*B, K, D); element l -> (b, k), Eqs. (6)-(7)
        x = x.reshape(batch * B, K, D)
        # (2) intra-subsequence modelling, Eqs. (8)-(9)
        o = self.res_ss(self.blstm_ss(x))  # (batch*B, K, D)
        # (3) inter-subsequence modelling along b for each k, Eqs. (11)-(12)
        o = o.reshape(batch, B, K, D).permute(0, 2, 1, 3).reshape(batch * K, B, D)
        o = self.res_ws(self.blstm_ws(o))  # (batch*K, B, D)
        # (4) unfold, Eq. (14): l = (b-1)*K + k
        o = o.reshape(batch, K, B, D).permute(0, 2, 1, 3).reshape(batch, B * K, D)
        return o[:, :L]


class LSeqSleepNet(nn.Module):
    """Paper-compliant L-SeqSleepNet (Phan et al., 2023). See module docstring.

    Args:
        n_classes: number of sleep stages (5).
        in_chan: input channels (paper: 1, C4-A1 / Fpz-Cz EEG).
        F, D, nfft, sf, lowfreq, highfreq: filterbank settings (F=129 -> M=D=32).
        epoch_hidden: He/2, hidden size per direction of the epoch BLSTM (64).
        epoch_attention: attention size A (64).
        B, K: fold shape, L = B*K (10 x 20 = 200).
        seq_hidden_ss, seq_hidden_ms: Hss/2 and Hws/2 per direction (64, 64).
            (``seq_hidden_ms`` keeps the draft's kwarg name for config compatibility.)
        d_clf: N_fc, units of the two classification fc layers (512).
        dropout: dropout rate on LSTM cells and fc layers (0.1).
        recurrent_bn: recurrent batch normalization in all BLSTMs (paper: True).
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_chan: int = 1,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        sf: int = 100,
        lowfreq: int = 0,
        highfreq: int = 50,
        epoch_hidden: int = 64,
        epoch_attention: int = 64,
        B: int = 10,
        K: int = 20,
        seq_hidden_ss: int = 64,
        seq_hidden_ms: int = 64,
        d_clf: int = 512,
        dropout: float = 0.1,
        recurrent_bn: bool = True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.B, self.K = B, K
        d_model = 2 * epoch_hidden

        self.epoch_encoder = PaperEpochEncoder(
            in_chan=in_chan, F=F, D=D, nfft=nfft, sf=sf, lowfreq=lowfreq, highfreq=highfreq,
            hidden_size=epoch_hidden, attention_size=epoch_attention,
            recurrent_bn=recurrent_bn, dropout=dropout,
        )
        self.sequence_model = LongSequenceModel(
            d_model=d_model, B=B, K=K, hidden_ss=seq_hidden_ss, hidden_ws=seq_hidden_ms,
            recurrent_bn=recurrent_bn, dropout=dropout,
        )
        # Sec. III-D: two fc(N_fc)+ReLU layers, then the output layer; dropout on fc.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_clf), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_clf, d_clf), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_clf, n_classes),
        )

    @property
    def sequence_length(self) -> int:
        return self.B * self.K

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, L, C, T, F) -> (batch, L, d_model) contextualised embeddings."""
        batch, L, C, T, F = x.shape
        x = self.epoch_encoder(x.reshape(batch * L, C, T, F)).reshape(batch, L, -1)
        return self.sequence_model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, L, C, T, F) -> (batch, L, n_classes) logits."""
        z = self.encode(x)
        batch, L, D = z.shape
        return self.classifier(z.reshape(batch * L, D)).reshape(batch, L, -1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "sequence_model.ln_ss.weight" in state_dict and "sequence_model.res_ss.ln.weight" not in state_dict:
            raise RuntimeError(
                "This checkpoint was trained with the deprecated, non paper-compliant draft "
                "(post-LN residual, no recurrent batch norm). Load it with "
                "physioex.models.lseqsleepnet:LSeqSleepNetDraft instead."
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


# ---------------------------------------------------------------------------
# Deprecated draft (kept for the `lseqsleepnet-phan` Hub checkpoint)
# ---------------------------------------------------------------------------


class EpochEncoder(nn.Module):
    """Draft epoch encoder: filterbank -> 1-layer cuDNN BiLSTM -> attention (no recurrent BN)."""

    def __init__(
        self,
        in_chan: int = 1,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        sf: int = 100,
        lowfreq: int = 0,
        highfreq: int = 50,
        hidden_size: int = 64,
        attention_size: int = 64,
    ):
        super().__init__()
        self.filterbank = LearnableFilterbank(in_chan, F, D, nfft, sf, lowfreq, highfreq)
        self.blstm = nn.LSTM(
            D * in_chan, hidden_size, num_layers=1, batch_first=True, bidirectional=True
        )
        self.attention = AttentionLayer(2 * hidden_size, attention_size)

    def forward(self, x):
        B, C, T, F = x.shape
        x = self.filterbank(x)  # (B, C, T, D)
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, C*D)
        x, _ = self.blstm(x)  # (B, T, 2*hidden)
        return self.attention(x)  # (B, 2*hidden)


class FoldProcessUnfold(nn.Module):
    """Draft fold-process-unfold. NOT paper-compliant: post-LN ``LN(fc(x) + input)``
    instead of Eq. (9)/(12) ``x~ + LN(fc(x~))``; silently truncates L > B*K."""

    def __init__(
        self,
        d_model: int = 128,
        B: int = 10,
        K: int = 20,
        hidden_ss: int = 64,
        hidden_ms: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.B, self.K, self.d_model = B, K, d_model
        self.blstm_ss = nn.LSTM(d_model, hidden_ss, num_layers=1, batch_first=True, bidirectional=True)
        self.fc_ss = nn.Linear(2 * hidden_ss, d_model)
        self.ln_ss = nn.LayerNorm(d_model)
        self.drop_ss = nn.Dropout(dropout)
        self.blstm_ms = nn.LSTM(d_model, hidden_ms, num_layers=1, batch_first=True, bidirectional=True)
        self.fc_ms = nn.Linear(2 * hidden_ms, d_model)
        self.ln_ms = nn.LayerNorm(d_model)
        self.drop_ms = nn.Dropout(dropout)

    def forward(self, x):
        batch, L, D = x.shape
        target_len = self.B * self.K
        if L < target_len:
            x = torch.cat([x, x.new_zeros(batch, target_len - L, D)], dim=1)
        elif L > target_len:
            x = x[:, :target_len]

        x = x.reshape(batch, self.B, self.K, D)
        x_intra = x.reshape(batch * self.B, self.K, D)
        out_ss, _ = self.blstm_ss(x_intra)
        out_ss = self.drop_ss(self.ln_ss(self.fc_ss(out_ss) + x_intra))
        out_ss = out_ss.reshape(batch, self.B, self.K, D)

        x_inter = out_ss.permute(0, 2, 1, 3).reshape(batch * self.K, self.B, D)
        out_ms, _ = self.blstm_ms(x_inter)
        out_ms = self.drop_ms(self.ln_ms(self.fc_ms(out_ms) + x_inter))
        out_ms = out_ms.reshape(batch, self.K, self.B, D).permute(0, 2, 1, 3)
        return out_ms.reshape(batch, self.B * self.K, D)[:, :L]


class LSeqSleepNetDraft(nn.Module):
    """Deprecated draft of L-SeqSleepNet (pre-2026-09). See module docstring.

    Kept only so that the ``lseqsleepnet-phan`` checkpoint on the Hub remains
    loadable. Use :class:`LSeqSleepNet` for any new training or comparison.
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_chan: int = 1,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        sf: int = 100,
        lowfreq: int = 0,
        highfreq: int = 50,
        epoch_hidden: int = 64,
        epoch_attention: int = 64,
        B: int = 10,
        K: int = 20,
        seq_hidden_ss: int = 64,
        seq_hidden_ms: int = 64,
        d_clf: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        warnings.warn(
            "LSeqSleepNetDraft is deprecated and not compliant with Phan et al. 2023; "
            "use LSeqSleepNet.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.n_classes = n_classes
        d_model = 2 * epoch_hidden
        self.epoch_encoder = EpochEncoder(
            in_chan=in_chan, F=F, D=D, nfft=nfft, sf=sf, lowfreq=lowfreq, highfreq=highfreq,
            hidden_size=epoch_hidden, attention_size=epoch_attention,
        )
        self.sequence_model = FoldProcessUnfold(
            d_model=d_model, B=B, K=K, hidden_ss=seq_hidden_ss, hidden_ms=seq_hidden_ms,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_clf), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_clf, d_clf), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_clf, n_classes),
        )

    def encode(self, x):
        batch, L, C, T, F = x.shape
        x = self.epoch_encoder(x.reshape(batch * L, C, T, F)).reshape(batch, L, -1)
        return self.sequence_model(x)

    def forward(self, x):
        z = self.encode(x)
        batch, L, D = z.shape
        return self.classifier(z.reshape(batch * L, D)).reshape(batch, L, -1)


if __name__ == "__main__":
    model = LSeqSleepNet(n_classes=5, in_chan=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"L-SeqSleepNet (paper-compliant): {n_params:,} parameters")
    x = torch.randn(2, 200, 1, 29, 129)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (2, 200, 5)
    y.sum().backward()
    print("Gradient flow: OK")
