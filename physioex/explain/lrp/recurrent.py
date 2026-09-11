"""LRP for LSTM / GRU via the Arras *signal-take* rule.

LRP for recurrent nets is an established method (Arras et al., *Explaining
Recurrent Neural Network Predictions in Sentiment Analysis*, WASSA@EMNLP 2017;
*Evaluating Recurrent Neural Network Explanations*, BlackboxNLP@ACL 2019).  The
recipe:

* **weighted (linear) connections** — the gate pre-activations ``W x + U h + b``
  → ε-LRP (``lxt.explicit.functional.linear_epsilon``);
* **sums** — ``c = f⊙c₋₁ + i⊙g`` → relevance split *proportionally* to the
  summands (``lxt.explicit.functional.add2``);
* **multiplicative gates** ``z = gate ⊙ source`` → the **signal-take rule**: all
  relevance to the *source* (the information signal), none to the *gate* (a
  learned control).  LXT's ``mul2`` splits 50/50 (uniform), so the signal-take
  behaviour is a small custom autograd Function here.

PyTorch's ``nn.LSTM``/``nn.GRU`` are *fused* cuDNN kernels whose internal gate
products are invisible to hooks.  So :class:`LRPLSTM` / :class:`LRPGRU`
re-run the recurrence **at cell level** using the LRP functionals, loading the
trained module's weights unchanged (no retraining).  The forward pass is
numerically identical to the fused module; only the backward carries relevance.

Requires the ``explain`` extra (``lxt``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LRP primitives for the recurrence
# ---------------------------------------------------------------------------


class _MulSignalTake(torch.autograd.Function):
    """``y = gate * source`` — forward is the product; backward routes **all**
    relevance to ``source`` and **none** to ``gate`` (Arras signal-take)."""

    @staticmethod
    def forward(ctx, gate, source):
        return gate * source

    @staticmethod
    def backward(ctx, relevance):
        # (grad_gate, grad_source): gate gets zero relevance, source gets all.
        return torch.zeros_like(relevance), relevance


def mul_signal_take(gate: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Elementwise ``gate * source`` with the signal-take LRP backward."""
    return _MulSignalTake.apply(gate, source)


def _st_act(preact: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """Straight-through activation: forward value ``act``, backward identity to
    ``preact`` (relevance passes through the source nonlinearity unchanged)."""
    return preact + (act - preact).detach()


# ---------------------------------------------------------------------------
# LRP-instrumented LSTM
# ---------------------------------------------------------------------------


class LRPLSTM(nn.Module):
    """Cell-level LSTM that carries LRP relevance in its backward pass.

    Build it from a trained ``nn.LSTM`` with :meth:`from_torch`; the forward
    output matches the fused module (within floating point).  Supports
    ``num_layers``, ``bidirectional``, ``batch_first`` and biases.  ``dropout``
    is ignored (evaluation-time attribution).

    Only the ``(output, (h_n, c_n))`` sequence output is reproduced; initial
    states default to zeros.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        bidirectional: bool = False,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.epsilon = epsilon
        self._params = nn.ParameterDict()

    # -- construction -------------------------------------------------------

    @classmethod
    def from_torch(cls, lstm: nn.LSTM) -> "LRPLSTM":
        """Create an LRPLSTM sharing ``lstm``'s trained weights."""
        obj = cls(
            input_size=lstm.input_size,
            hidden_size=lstm.hidden_size,
            num_layers=lstm.num_layers,
            bias=lstm.bias,
            batch_first=lstm.batch_first,
            bidirectional=lstm.bidirectional,
        )
        for name, p in lstm.named_parameters():
            # names like weight_ih_l0, weight_hh_l0_reverse, bias_ih_l1, ...
            obj._params[name.replace(".", "_")] = nn.Parameter(
                p.detach().clone(), requires_grad=False
            )
        return obj

    def _p(self, name: str):
        return self._params.get(name.replace(".", "_"))

    # -- one directional pass over one layer --------------------------------

    def _layer_dir(self, x, w_ih, w_hh, b_ih, b_hh, reverse: bool):
        # x: (T, B, in).  Returns (T, B, H).
        from lxt.explicit.functional import add2, linear_epsilon

        T, B, _ = x.shape
        H = self.hidden_size
        h = x.new_zeros(B, H)
        c = x.new_zeros(B, H)
        steps = range(T - 1, -1, -1) if reverse else range(T)
        outs = [None] * T
        for t in steps:
            gi = linear_epsilon(x[t], w_ih, b_ih, epsilon=self.epsilon)
            gh = linear_epsilon(h, w_hh, b_hh, epsilon=self.epsilon)
            gates = add2(gi, gh)  # proportional split (input vs recurrent)
            ii, ff, gg, oo = gates.split(H, dim=-1)
            i = torch.sigmoid(ii)
            f = torch.sigmoid(ff)
            g = _st_act(gg, torch.tanh(gg))  # source nonlinearity → identity bwd
            o = torch.sigmoid(oo)
            # c' = f⊙c + i⊙g ; sources are c and g
            c = add2(mul_signal_take(f, c), mul_signal_take(i, g))
            # h' = o⊙tanh(c') ; source is tanh(c') → identity bwd to c'
            h = mul_signal_take(o, _st_act(c, torch.tanh(c)))
            outs[t] = h
        return torch.stack(outs, dim=0), h, c  # (T,B,H), final (h, c)

    def forward(self, x: torch.Tensor):
        """Returns ``(output, (h_n, c_n))`` like ``nn.LSTM``."""
        if self.batch_first:
            x = x.transpose(0, 1)  # (B, T, in) -> (T, B, in)
        layer_in = x
        h_states, c_states = [], []
        for layer in range(self.num_layers):
            suff = f"_l{layer}"
            fwd, hf, cf = self._layer_dir(
                layer_in,
                self._p("weight_ih" + suff),
                self._p("weight_hh" + suff),
                self._p("bias_ih" + suff),
                self._p("bias_hh" + suff),
                reverse=False,
            )
            if self.bidirectional:
                bwd, hb, cb = self._layer_dir(
                    layer_in,
                    self._p("weight_ih" + suff + "_reverse"),
                    self._p("weight_hh" + suff + "_reverse"),
                    self._p("bias_ih" + suff + "_reverse"),
                    self._p("bias_hh" + suff + "_reverse"),
                    reverse=True,
                )
                layer_in = torch.cat([fwd, bwd], dim=-1)  # (T, B, 2H)
                h_states += [hf, hb]
                c_states += [cf, cb]
            else:
                layer_in = fwd
                h_states.append(hf)
                c_states.append(cf)
        out = layer_in
        if self.batch_first:
            out = out.transpose(0, 1)  # (T, B, dir*H) -> (B, T, dir*H)
        return out, (torch.stack(h_states, 0), torch.stack(c_states, 0))


# ---------------------------------------------------------------------------
# LRP-instrumented GRU
# ---------------------------------------------------------------------------


class LRPGRU(nn.Module):
    """Cell-level GRU that carries LRP relevance (Arras signal-take).

    Build from a trained ``nn.GRU`` with :meth:`from_torch`.  Same options as
    :class:`LRPLSTM`.  PyTorch GRU gate order is ``[r, z, n]``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        bidirectional: bool = False,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.epsilon = epsilon
        self._params = nn.ParameterDict()

    @classmethod
    def from_torch(cls, gru: nn.GRU) -> "LRPGRU":
        obj = cls(
            input_size=gru.input_size,
            hidden_size=gru.hidden_size,
            num_layers=gru.num_layers,
            bias=gru.bias,
            batch_first=gru.batch_first,
            bidirectional=gru.bidirectional,
        )
        for name, p in gru.named_parameters():
            obj._params[name.replace(".", "_")] = nn.Parameter(
                p.detach().clone(), requires_grad=False
            )
        return obj

    def _p(self, name: str):
        return self._params.get(name.replace(".", "_"))

    def _layer_dir(self, x, w_ih, w_hh, b_ih, b_hh, reverse: bool):
        from lxt.explicit.functional import add2, linear_epsilon

        T, B, _ = x.shape
        H = self.hidden_size
        h = x.new_zeros(B, H)
        steps = range(T - 1, -1, -1) if reverse else range(T)
        outs = [None] * T
        b_ih_r = b_ih if b_ih is not None else None
        b_hh_r = b_hh if b_hh is not None else None
        for t in steps:
            gi = linear_epsilon(x[t], w_ih, b_ih_r, epsilon=self.epsilon)
            gh = linear_epsilon(h, w_hh, b_hh_r, epsilon=self.epsilon)
            i_r, i_z, i_n = gi.split(H, dim=-1)
            h_r, h_z, h_n = gh.split(H, dim=-1)
            r = torch.sigmoid(add2(i_r, h_r))
            z = torch.sigmoid(add2(i_z, h_z))
            # n = tanh(i_n + r ⊙ h_n) ; r gates the recurrent term (source h_n)
            n_pre = add2(i_n, mul_signal_take(r, h_n))
            n = _st_act(n_pre, torch.tanh(n_pre))
            # h' = (1-z)⊙n + z⊙h ; sources are n and h
            one_minus_z = 1.0 - z
            h = add2(mul_signal_take(one_minus_z, n), mul_signal_take(z, h))
            outs[t] = h
        return torch.stack(outs, dim=0), h  # (T,B,H), final h

    def forward(self, x: torch.Tensor):
        """Returns ``(output, h_n)`` like ``nn.GRU``."""
        if self.batch_first:
            x = x.transpose(0, 1)
        layer_in = x
        h_states = []
        for layer in range(self.num_layers):
            suff = f"_l{layer}"
            fwd, hf = self._layer_dir(
                layer_in,
                self._p("weight_ih" + suff),
                self._p("weight_hh" + suff),
                self._p("bias_ih" + suff),
                self._p("bias_hh" + suff),
                reverse=False,
            )
            if self.bidirectional:
                bwd, hb = self._layer_dir(
                    layer_in,
                    self._p("weight_ih" + suff + "_reverse"),
                    self._p("weight_hh" + suff + "_reverse"),
                    self._p("bias_ih" + suff + "_reverse"),
                    self._p("bias_hh" + suff + "_reverse"),
                    reverse=True,
                )
                layer_in = torch.cat([fwd, bwd], dim=-1)
                h_states += [hf, hb]
            else:
                layer_in = fwd
                h_states.append(hf)
        out = layer_in
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, torch.stack(h_states, 0)
