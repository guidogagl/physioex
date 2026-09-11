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
  learned control).  LXT's ``mul2`` splits 50/50 (uniform), so signal-take is a
  small custom autograd Function here.

PyTorch's ``nn.LSTM``/``nn.GRU`` are *fused* cuDNN kernels whose internal gate
products are invisible to hooks.  So :class:`LRPLSTM` / :class:`LRPGRU`
**subclass** ``nn.LSTM`` / ``nn.GRU`` (so ``isinstance`` checks and the
``(output, states)`` return interface keep working in host models) and override
``forward`` to re-run the recurrence at cell level with the LRP functionals.
Build with :meth:`from_torch`; the forward output is numerically identical to
the fused module, only the backward carries relevance.

Requires the ``explain`` extra (``lxt``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
        return torch.zeros_like(relevance), relevance


def mul_signal_take(gate: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Elementwise ``gate * source`` with the signal-take LRP backward."""
    return _MulSignalTake.apply(gate, source)


def _st_act(preact: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """Straight-through activation: forward value ``act``, backward identity to
    ``preact`` (relevance passes through the source nonlinearity unchanged)."""
    return preact + (act - preact).detach()


def _copy_config_and_weights(dst, src):
    """Load ``src``'s trained weights into ``dst`` and freeze them."""
    dst.load_state_dict(src.state_dict())
    dst.eval()
    for p in dst.parameters():
        p.requires_grad_(False)
    dst.epsilon = 1e-6
    return dst


# ---------------------------------------------------------------------------
# LRP-instrumented LSTM (subclasses nn.LSTM)
# ---------------------------------------------------------------------------


class LRPLSTM(nn.LSTM):
    """Cell-level LSTM carrying LRP relevance; drop-in for a trained ``nn.LSTM``.

    Build with :meth:`from_torch`.  ``forward`` returns ``(output, (h_n, c_n))``
    like ``nn.LSTM``.  Supports ``num_layers``, ``bidirectional``,
    ``batch_first`` and biases; ``dropout`` is ignored (eval-time attribution);
    initial states default to zeros.
    """

    epsilon: float = 1e-6

    @classmethod
    def from_torch(cls, lstm: nn.LSTM) -> "LRPLSTM":
        obj = cls(
            lstm.input_size,
            lstm.hidden_size,
            num_layers=lstm.num_layers,
            bias=lstm.bias,
            batch_first=lstm.batch_first,
            bidirectional=lstm.bidirectional,
        )
        return _copy_config_and_weights(obj, lstm)

    def _p(self, name):
        return getattr(self, name, None)

    def _layer_dir(self, x, w_ih, w_hh, b_ih, b_hh, reverse: bool):
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
            gates = add2(gi, gh)
            ii, ff, gg, oo = gates.split(H, dim=-1)
            i = torch.sigmoid(ii)
            f = torch.sigmoid(ff)
            g = _st_act(gg, torch.tanh(gg))
            o = torch.sigmoid(oo)
            c = add2(mul_signal_take(f, c), mul_signal_take(i, g))
            h = mul_signal_take(o, _st_act(c, torch.tanh(c)))
            outs[t] = h
        return torch.stack(outs, dim=0), h, c

    def forward(self, x, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        layer_in = x
        h_states, c_states = [], []
        for layer in range(self.num_layers):
            suff = f"_l{layer}"
            fwd, hf, cf = self._layer_dir(
                layer_in, self._p("weight_ih" + suff), self._p("weight_hh" + suff),
                self._p("bias_ih" + suff), self._p("bias_hh" + suff), reverse=False,
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
                layer_in = torch.cat([fwd, bwd], dim=-1)
                h_states += [hf, hb]
                c_states += [cf, cb]
            else:
                layer_in = fwd
                h_states.append(hf)
                c_states.append(cf)
        out = layer_in
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, (torch.stack(h_states, 0), torch.stack(c_states, 0))


# ---------------------------------------------------------------------------
# LRP-instrumented GRU (subclasses nn.GRU)
# ---------------------------------------------------------------------------


class LRPGRU(nn.GRU):
    """Cell-level GRU carrying LRP relevance; drop-in for a trained ``nn.GRU``.

    ``forward`` returns ``(output, h_n)`` like ``nn.GRU``.  PyTorch GRU gate
    order is ``[r, z, n]``.
    """

    epsilon: float = 1e-6

    @classmethod
    def from_torch(cls, gru: nn.GRU) -> "LRPGRU":
        obj = cls(
            gru.input_size,
            gru.hidden_size,
            num_layers=gru.num_layers,
            bias=gru.bias,
            batch_first=gru.batch_first,
            bidirectional=gru.bidirectional,
        )
        return _copy_config_and_weights(obj, gru)

    def _p(self, name):
        return getattr(self, name, None)

    def _layer_dir(self, x, w_ih, w_hh, b_ih, b_hh, reverse: bool):
        from lxt.explicit.functional import add2, linear_epsilon

        T, B, _ = x.shape
        H = self.hidden_size
        h = x.new_zeros(B, H)
        steps = range(T - 1, -1, -1) if reverse else range(T)
        outs = [None] * T
        for t in steps:
            gi = linear_epsilon(x[t], w_ih, b_ih, epsilon=self.epsilon)
            gh = linear_epsilon(h, w_hh, b_hh, epsilon=self.epsilon)
            i_r, i_z, i_n = gi.split(H, dim=-1)
            h_r, h_z, h_n = gh.split(H, dim=-1)
            r = torch.sigmoid(add2(i_r, h_r))
            z = torch.sigmoid(add2(i_z, h_z))
            n_pre = add2(i_n, mul_signal_take(r, h_n))
            n = _st_act(n_pre, torch.tanh(n_pre))
            one_minus_z = 1.0 - z
            h = add2(mul_signal_take(one_minus_z, n), mul_signal_take(z, h))
            outs[t] = h
        return torch.stack(outs, dim=0), h

    def forward(self, x, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        layer_in = x
        h_states = []
        for layer in range(self.num_layers):
            suff = f"_l{layer}"
            fwd, hf = self._layer_dir(
                layer_in, self._p("weight_ih" + suff), self._p("weight_hh" + suff),
                self._p("bias_ih" + suff), self._p("bias_hh" + suff), reverse=False,
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
