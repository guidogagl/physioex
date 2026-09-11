"""LRP for LSTM / GRU via the Arras *signal-take* rule.

LRP for recurrent nets is an established method (Arras et al., *Explaining
Recurrent Neural Network Predictions in Sentiment Analysis*, WASSA@EMNLP 2017;
*Evaluating Recurrent Neural Network Explanations*, BlackboxNLP@ACL 2019).  The
recipe:

* **weighted (linear) connections** — the gate pre-activations ``W x + U h + b``
  → ε-LRP (:func:`~physioex.explain.lrp._functional.linear_eps`);
* **sums** — ``c = f⊙c₋₁ + i⊙g`` → relevance split *proportionally* to the
  summands (:func:`~physioex.explain.lrp._functional.add_eps`);
* **multiplicative gates** ``z = gate ⊙ source`` → the **signal-take rule**: all
  relevance to the *source* (the information signal), none to the *gate* (a
  learned control);
* the *source* nonlinearities (``tanh``) pass relevance as identity
  (:func:`~physioex.explain.lrp._functional.st_identity`).

Applying ε-LRP separately to ``W x`` and ``U h`` and combining with the
proportional ``add_eps`` is equivalent to Arras's single ε-rule on the full
pre-activation up to ``O(ε)``.  Stabilisers are **signed** (``z + ε·sign z``),
as in Arras, so near-zero *negative* sums (cell states crossing zero) cannot
blow up.  Bias relevance is absorbed (LRP-ε convention, Arras δ=0): with biases,
``Σ R`` is a *fraction* of ``f`` — use the conservation diagnostics to see it.

PyTorch's ``nn.LSTM``/``nn.GRU`` are *fused* kernels whose gate products are
invisible to hooks.  :class:`LRPLSTM` / :class:`LRPGRU` **subclass** them (so
``isinstance`` checks and the ``(output, states)`` interface keep working in
host models) and override ``forward`` to re-run the recurrence at cell level.
Build with :meth:`from_torch`; the forward output is numerically identical to
the fused module, only the backward carries relevance.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from physioex.explain.lrp._functional import (
    add_eps,
    linear_eps,
    mul_signal_take,
    st_identity,
)

# backwards-compatible aliases (used by tests / older code)
_st_act = st_identity


def _load_frozen(dst, src, epsilon: float):
    dst.load_state_dict(src.state_dict())
    dst.eval()
    for p in dst.parameters():
        p.requires_grad_(False)
    dst.epsilon = float(epsilon)
    return dst


def _check_supported(rnn):
    if getattr(rnn, "proj_size", 0):
        raise NotImplementedError("LRP recurrent layers do not support proj_size > 0")


class LRPLSTM(nn.LSTM):
    """Cell-level LSTM carrying LRP relevance; drop-in for a trained ``nn.LSTM``.

    Build with :meth:`from_torch`.  ``forward(x)`` returns ``(output,
    (h_n, c_n))`` like ``nn.LSTM``; an explicit initial state ``hx`` is not
    supported (raises), ``dropout`` is ignored (eval-time attribution).
    """

    epsilon: float = 1e-6

    @classmethod
    def from_torch(cls, lstm: nn.LSTM, epsilon: float = 1e-6) -> "LRPLSTM":
        _check_supported(lstm)
        obj = cls(
            lstm.input_size, lstm.hidden_size, num_layers=lstm.num_layers,
            bias=lstm.bias, batch_first=lstm.batch_first,
            bidirectional=lstm.bidirectional,
        )
        return _load_frozen(obj, lstm, epsilon)

    def _p(self, name):
        return getattr(self, name, None)

    def _layer_dir(self, x, w_ih, w_hh, b_ih, b_hh, reverse: bool):
        T, B, _ = x.shape
        H, eps = self.hidden_size, self.epsilon
        h = x.new_zeros(B, H)
        c = x.new_zeros(B, H)
        steps = range(T - 1, -1, -1) if reverse else range(T)
        outs = [None] * T
        for t in steps:
            gates = add_eps(
                linear_eps(x[t], w_ih, b_ih, eps), linear_eps(h, w_hh, b_hh, eps), eps
            )
            ii, ff, gg, oo = gates.split(H, dim=-1)
            i, f, o = torch.sigmoid(ii), torch.sigmoid(ff), torch.sigmoid(oo)
            g = st_identity(gg, torch.tanh(gg))  # source nonlinearity → identity
            c = add_eps(mul_signal_take(f, c), mul_signal_take(i, g), eps)
            h = mul_signal_take(o, st_identity(c, torch.tanh(c)))
            outs[t] = h
        return torch.stack(outs, dim=0), h, c

    def forward(self, x, hx=None):
        if hx is not None:
            raise NotImplementedError("LRPLSTM: explicit initial state hx is not supported")
        if x.dim() != 3:
            raise ValueError("LRPLSTM expects a batched 3-D input")
        if self.batch_first:
            x = x.transpose(0, 1)
        layer_in, h_states, c_states = x, [], []
        for layer in range(self.num_layers):
            s = f"_l{layer}"
            fwd, hf, cf = self._layer_dir(
                layer_in, self._p("weight_ih" + s), self._p("weight_hh" + s),
                self._p("bias_ih" + s), self._p("bias_hh" + s), reverse=False,
            )
            if self.bidirectional:
                r = s + "_reverse"
                bwd, hb, cb = self._layer_dir(
                    layer_in, self._p("weight_ih" + r), self._p("weight_hh" + r),
                    self._p("bias_ih" + r), self._p("bias_hh" + r), reverse=True,
                )
                layer_in = torch.cat([fwd, bwd], dim=-1)
                h_states += [hf, hb]
                c_states += [cf, cb]
            else:
                layer_in = fwd
                h_states.append(hf)
                c_states.append(cf)
        out = layer_in.transpose(0, 1) if self.batch_first else layer_in
        return out, (torch.stack(h_states, 0), torch.stack(c_states, 0))


class LRPGRU(nn.GRU):
    """Cell-level GRU carrying LRP relevance; drop-in for a trained ``nn.GRU``.

    ``forward(x)`` returns ``(output, h_n)`` like ``nn.GRU``.  PyTorch gate order
    is ``[r, z, n]`` with ``n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))`` and
    ``h' = (1−z)⊙n + z⊙h``; sources are ``n``, ``h`` and the recurrent
    pre-activation gated by ``r``.
    """

    epsilon: float = 1e-6

    @classmethod
    def from_torch(cls, gru: nn.GRU, epsilon: float = 1e-6) -> "LRPGRU":
        _check_supported(gru)
        obj = cls(
            gru.input_size, gru.hidden_size, num_layers=gru.num_layers,
            bias=gru.bias, batch_first=gru.batch_first,
            bidirectional=gru.bidirectional,
        )
        return _load_frozen(obj, gru, epsilon)

    def _p(self, name):
        return getattr(self, name, None)

    def _layer_dir(self, x, w_ih, w_hh, b_ih, b_hh, reverse: bool):
        T, B, _ = x.shape
        H, eps = self.hidden_size, self.epsilon
        h = x.new_zeros(B, H)
        steps = range(T - 1, -1, -1) if reverse else range(T)
        outs = [None] * T
        for t in steps:
            gi = linear_eps(x[t], w_ih, b_ih, eps)
            gh = linear_eps(h, w_hh, b_hh, eps)
            i_r, i_z, i_n = gi.split(H, dim=-1)
            h_r, h_z, h_n = gh.split(H, dim=-1)
            r = torch.sigmoid(add_eps(i_r, h_r, eps))
            z = torch.sigmoid(add_eps(i_z, h_z, eps))
            n_pre = add_eps(i_n, mul_signal_take(r, h_n), eps)
            n = st_identity(n_pre, torch.tanh(n_pre))
            h = add_eps(mul_signal_take(1.0 - z, n), mul_signal_take(z, h), eps)
            outs[t] = h
        return torch.stack(outs, dim=0), h

    def forward(self, x, hx=None):
        if hx is not None:
            raise NotImplementedError("LRPGRU: explicit initial state hx is not supported")
        if x.dim() != 3:
            raise ValueError("LRPGRU expects a batched 3-D input")
        if self.batch_first:
            x = x.transpose(0, 1)
        layer_in, h_states = x, []
        for layer in range(self.num_layers):
            s = f"_l{layer}"
            fwd, hf = self._layer_dir(
                layer_in, self._p("weight_ih" + s), self._p("weight_hh" + s),
                self._p("bias_ih" + s), self._p("bias_hh" + s), reverse=False,
            )
            if self.bidirectional:
                r = s + "_reverse"
                bwd, hb = self._layer_dir(
                    layer_in, self._p("weight_ih" + r), self._p("weight_hh" + r),
                    self._p("bias_ih" + r), self._p("bias_hh" + r), reverse=True,
                )
                layer_in = torch.cat([fwd, bwd], dim=-1)
                h_states += [hf, hb]
            else:
                layer_in = fwd
                h_states.append(hf)
        out = layer_in.transpose(0, 1) if self.batch_first else layer_in
        return out, torch.stack(h_states, 0)
