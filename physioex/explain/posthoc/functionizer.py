# Post-Hoc explanable algorithms expects scalar functions as input.
# f : R^n -> R
# In PhysioEx we usually deal with Sequence-to-Sequence Models that are not scalar functions.
# f_seq : R^(L x n) -> R^(L x c)
# where L is the sequence length, n is the input feature dimension and c is the number of classes.

import torch


# Naive Functionizer R^n -> R^b x L x n -> R^c -> R
class Funct(torch.nn.Module):
    def __init__(
        self, model: torch.nn.Module, out_index: int = 0, softmax: bool = False
    ):
        super(Funct, self).__init__()

        self.model = model
        self.out_index = out_index
        self.softmax = softmax

    def forward(self, x):
        # x shape : n --> batch x L x n
        shape = x.shape
        shape = [1, 1] + list(shape)
        x = x.reshape(shape)
        out = self.model(x)
        if self.softmax:
            out = torch.nn.functional.softmax(out, dim=-1)
        return out[:, self.out_index].view(-1)


class SeqFunct(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        in_index: int = 0,
        out_index: int = 0,
        softmax: bool = False,
    ):
        super(SeqFunct, self).__init__()

        self.model = model
        self.in_index = in_index
        self.out_index = out_index
        self.softmax = softmax

    def forward(self, x):
        # x shape : L x n
        out = self.model(x.unsqueeze(0))
        if self.softmax:
            out = torch.nn.functional.softmax(out, dim=-1)

        return out[:, self.in_index, self.out_index].view(-1)
