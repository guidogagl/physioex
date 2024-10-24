import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError

    def summary(self, input_size, device, batch_size=-1):

        def register_hook(module):

            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[(-1)].split("'")[0]
                module_idx = len(summary)
                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    if class_name in ("GRU", "LSTM", "RNN"):
                        summary[m_key]["output_shape"] = [batch_size] + list(
                            output[0].size()
                        )[1:]
                    else:
                        summary[m_key]["output_shape"] = [
                            [-1] + list(o.size())[1:] for o in output
                        ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size
                summary[m_key]["trainable"] = any(
                    [p.requires_grad for p in module.parameters()]
                )
                params = np.sum(
                    [
                        np.prod(list(p.size()))
                        for p in module.parameters()
                        if p.requires_grad
                    ]
                )
                summary[m_key]["nb_params"] = int(params)

            if not isinstance(module, nn.Sequential):
                if not isinstance(module, nn.ModuleList):
                    pass
            if not module == self:
                hooks.append(module.register_forward_hook(hook))

        assert device.type in (
            "cuda",
            "cpu",
        ), "Input device is not valid, please specify 'cuda' or 'cpu'"
        if device.type == "cuda":
            if torch.cuda.is_available():
                dtype = torch.cuda.FloatTensor
            else:
                dtype = torch.FloatTensor
            if isinstance(input_size, tuple):
                input_size = [input_size]
            x = [(torch.rand)(*(2,), *in_size).type(dtype) for in_size in input_size]
            summary = OrderedDict()
            hooks = []
            self.apply(register_hook)
            self(*x)
            for h in hooks:
                h.remove()

            print("----------------------------------------------------------------")
            line_new = "{:>20}  {:>25} {:>15}".format(
                "Layer (type)", "Output Shape", "Param #"
            )
            print(line_new)
            print("================================================================")
            total_params = 0
            total_output = 0
            trainable_params = 0
            for layer in summary:
                line_new = "{:>20}  {:>25} {:>15}".format(
                    layer,
                    str(summary[layer]["output_shape"]),
                    "{0:,}".format(summary[layer]["nb_params"]),
                )
                total_params += summary[layer]["nb_params"]
                if any(isinstance(el, list) for el in summary[layer]["output_shape"]):
                    for list_out in summary[layer]["output_shape"]:
                        total_output += np.prod(list_out, dtype=np.int64)
                else:
                    total_output += np.prod(
                        summary[layer]["output_shape"], dtype=np.int64
                    )
                if "trainable" in summary[layer]:
                    if summary[layer]["trainable"] == True:
                        trainable_params += summary[layer]["nb_params"]
                    print(line_new)
            total_input_size = abs(
                np.sum([np.prod(x) for x in input_size])
                * batch_size
                * 4.0
                / 1073741824.0
            )
            total_output_size = abs(2.0 * total_output * 4.0 / 1073741824.0)
            total_params_size = abs(total_params * 4.0 / 1073741824.0)
            total_size = total_params_size + total_output_size + total_input_size
            print("================================================================")
            print("Total params: {0:,}".format(total_params))
            print("Trainable params: {0:,}".format(trainable_params))
            print("Non-trainable params: {0:,}".format(total_params - trainable_params))
            print("----------------------------------------------------------------")
            print("Input size (GB): %0.2f" % total_input_size)
            print("Forward/backward pass size (GB): %0.2f" % total_output_size)
            print("Params size (GB): %0.2f" % total_params_size)
            print("Estimated Total Size (GB): %0.2f" % total_size)
            print("----------------------------------------------------------------")

    def debug_model(self, input_size, device, cond_size=False):
        if cond_size or cond_size is 0:
            self.summary([input_size[1:], (cond_size,)], device, input_size[0])
            z = torch.rand((input_size[0], cond_size)).to(device)
        else:
            self.summary(input_size[1:], device, input_size[0])
        X = torch.rand(input_size).to(device)
        print("Input size: ", X.size)
        time_start = time.time()
        if cond_size or cond_size is 0:
            out = self(X, z)
        else:
            out = self(X)
        print("Batch time: {:.3f}".format(time.time() - time_start))
        for k, v in out.items():
            print("Key: ", k)
            print("Output size: ", v.size())


class M_PSG2FEAT(BaseModel):
    def __init__(self, config):
        """A model to process epochs of polysomnography data

        Args:
            config: An instance of the config class with set attributes.
        """
        super().__init__()
        # Attributes
        self.n_channels = config.n_channels
        self.n_class = config.pre_n_class
        self.n_label = len(config.pre_label)
        self.return_only_pred = config.return_only_pred
        self.return_pdf_shape = config.return_pdf_shape

        ### LAYERS ###
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(1, 32, (self.n_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        # self.channel_mixer = nn.Sequential(
        #         nn.Conv2d(self.n_channels, 32, (1, 1), bias = False),
        #         nn.BatchNorm2d(32),
        #         nn.ReLU6(inplace=True))
        self.MobileNetV2 = MobileNetV2(num_classes=self.n_class)
        self.LSTM = nn.LSTM(128, 128, num_layers=1, bidirectional=True)
        self.add_attention = AdditiveAttention(256, 512)
        self.linear_l = nn.Linear(256, 256)
        self.dropout = nn.Dropout(p=config.do_f)
        # Label specific layers
        self.classify_bias_init = [50.0, 10.0]
        self.classify_l = nn.Linear(256, self.n_class * 2)
        self.classify_l.bias.data = torch.Tensor(self.classify_bias_init)

    def forward(self, X):
        """Forward call of model

        Args:
            X (Tensor): Input polysomnography epoch of size [batch_size, n_channels, 38400]

        Returns:
            dict: A dict {'pred': age predictions,
                          'feat': latent space representation,
                          'alpha': additive attention weights,
                          'pdf_shape': shape of predicted age distribution (not used)}
        """
        # X.size() = [Batch_size, Channels = 13, Time = 5*60*128]
        X = torch.unsqueeze(X, 1)
        # X.size() = [Batch_size, Feature_maps = 1, Channels = 13, Time = 5*60*128]
        # Channel Mixer
        X = self.channel_mixer(X)
        # X.size() = [Batch_size, Feature_maps = 32, Channels = 1, Time = 5*60*128]
        # Modified MobileNetV2
        X = self.MobileNetV2(X)
        # X.size() = [Batch_size, Feature_maps = 320, Channels = 1, Time = 5*60*16]
        # LSTM layer
        X = X.view(-1, X.size(1), 1, int(X.size(3) / (5 * 4)), 5 * 4)
        X = torch.squeeze(X.mean([4]), 2)
        X = X.permute(2, 0, 1)
        self.LSTM.flatten_parameters()
        X, _ = self.LSTM(X)
        # Attention layer
        X = X.permute(1, 0, 2)
        # Averaged features
        X_avg = torch.mean(X, 1)
        X_a, alpha = self.add_attention(X)
        # Linear Transform
        X_a = self.linear_l(F.relu(X_a))
        # Dropout
        X_a = self.dropout(X_a)
        # Linear
        C = self.classify_l(X_a)
        C = torch.squeeze(C, 1)
        if self.return_only_pred:
            return torch.unsqueeze(C[:, 0], 1)
        else:
            return {
                "pred": C[:, 0],
                "feat": torch.cat((X_a, X_avg), 1),
                "alpha": alpha,
                "pdf_shape": C[:, 1],
            }


class M_PSG2FEAT_wn(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config.n_channels
        self.n_class = config.pre_n_class
        self.n_label = len(config.pre_label)
        self.return_only_pred = config.return_only_pred
        self.classify_bias_init = [36.7500]
        ### LAYERS ###
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(1, 32, (self.n_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.wavenet = nn.Sequential(
            dilated_conv_block(32, 64, (1, 9), (1, 1), (1, 1), 1, (1, 2), config.do_f),
            dilated_conv_block(64, 128, (1, 9), (1, 1), (1, 2), 1, (1, 2), config.do_f),
            dilated_conv_block(
                128, 128, (1, 9), (1, 1), (1, 4), 1, (1, 2), config.do_f
            ),
            dilated_conv_block(
                128, 256, (1, 9), (1, 1), (1, 8), 1, (1, 2), config.do_f
            ),
            dilated_conv_block(
                256, 256, (1, 9), (1, 1), (1, 16), 1, (1, 2), config.do_f
            ),
        )

        self.dropout = nn.Dropout(p=config.do_f)
        self.classify_l = nn.Linear(256, self.n_class)
        self.classify_l.bias.data = torch.Tensor(self.classify_bias_init)

    def forward(self, X):
        # X.size() = [Batch_size, Channels = 13, Time = 5*60*128]
        X = torch.unsqueeze(X, 1)
        # X.size() = [Batch_size, Feature_maps = 1, Channels = 13, Time = 5*60*128]
        # Channel Mixer
        X = self.channel_mixer(X)
        # X.size() = [Batch_size, Feature_maps = 32, Channels = 1, Time = 5*60*128]
        X = self.wavenet(X)
        # X.size() = [Batch_size, Feature_maps = 256, Channels = 1, Time = 5*60*16]
        X = X.mean([2, 3])
        X = self.dropout(X)
        C = self.classify_l(F.relu(X))
        C = torch.squeeze(C, 1)
        alpha = torch.ones(C.size(0), 5, 1) / 5.0
        if self.return_only_pred:
            return C
        else:
            return {"pred": C, "feat": X, "alpha": alpha}


class M_PSG2FEAT_simple(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config.n_channels
        self.n_class = config.pre_n_class
        self.n_label = len(config.pre_label)
        self.return_only_pred = config.return_only_pred
        self.classify_bias_init = [36.7500]

        ### LAYERS ###
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(1, 32, (self.n_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.CNN = nn.Sequential(
            ConvBNReLU(32, 64, (1, 3), (1, 2)),
            ConvBNReLU(64, 64, (1, 3), (1, 2)),
            ConvBNReLU(64, 128, (1, 3), (1, 2)),
            ConvBNReLU(128, 256, (1, 3), (1, 2)),
            ConvBNReLU(256, 256, (1, 3), (1, 1)),
        )
        self.classify_l = nn.Linear(256, self.n_class)
        self.dropout = nn.Dropout(p=config.do_f)
        self.classify_l.bias.data = torch.Tensor(self.classify_bias_init)

    def forward(self, X):
        # X.size() = [Batch_size, Channels = 13, Time = 5*60*128]
        X = torch.unsqueeze(X, 1)
        # X.size() = [Batch_size, Feature_maps = 1, Channels = 13, Time = 5*60*128]
        # Channel Mixer
        X = self.channel_mixer(X)
        # X.size() = [Batch_size, Feature_maps = 32, Channels = 1, Time = 5*60*128]
        X = self.CNN(X)
        # X.size() = [Batch_size, Feature_maps = 256, Channels = 1, Time = 5*60*16]
        X = X.mean([2, 3])
        # dropout
        X = self.dropout(X)
        # Classify layer
        C = self.classify_l(F.relu(X))
        C = torch.squeeze(C, 1)
        alpha = torch.ones(C.size(0), 5, 1) / 5.0
        if self.return_only_pred:
            return C
        else:
            return {"pred": C, "feat": X, "alpha": alpha}


class M_FEAT2LABEL(BaseModel):
    def __init__(self, config):
        """A model to process latent space representations of polysomnography data

        Args:
            config: An instance of the config class with set attributes.
        """
        super().__init__()
        self.n_channels = config.n_channels
        self.n_class = config.n_class
        self.label_cond = sum(config.label_cond_size)
        self.return_only_pred = config.return_only_pred
        self.return_pdf_shape = config.return_pdf_shape

        ### LAYERS ###
        self.LSTM = nn.LSTM(
            512,
            32 * config.net_size_scale,
            num_layers=config.lstm_n,
            bidirectional=True,
        )
        self.add_attention = AdditiveAttention(
            64 * config.net_size_scale, 128 * config.net_size_scale
        )
        self.linear_l = nn.Linear(
            64 * config.net_size_scale + self.label_cond, 64 * config.net_size_scale
        )
        self.dropout = nn.Dropout(p=config.do_l)
        self.classify_l = nn.Linear(64 * config.net_size_scale, self.n_class * 2)
        self.classify_bias_init = [50.0, 10.0]
        self.classify_l.bias.data = torch.Tensor(self.classify_bias_init)

    #        self.transformer = nn.TransformerEncoderLayer(256, nhead = 8)

    def forward(self, X, z):
        """Forward call of model

        Args:
            X (Tensor): latent space representation of size [batch_size, n_epochs, n_features]
            z (Tensor): additional input scalars of size [batch_size, n_z]

        Returns:
            dict: A dict {'pred': age predictions,
                          'alpha': additive attention weights,
                          'pdf_shape': shape of predicted age distribution (not used)}
        """
        # z.size() = [batch_size, label_cond]
        # X.size() = [batch_size, n_epochs, Features]
        X = X.permute(1, 0, 2)
        self.LSTM.flatten_parameters()
        X, _ = self.LSTM(X)
        # Attention layer
        X = X.permute(1, 0, 2)
        X, alpha = self.add_attention(X)
        # Concatenate with conditional labels
        X = torch.cat((X, z), 1)
        # Linear Transform
        X = self.linear_l(F.relu(X))
        # Dropout
        X = self.dropout(X)
        # Classify layer
        X = self.classify_l(F.relu(X))
        X = torch.squeeze(X, 1)
        if self.return_only_pred:
            return X[:, 0]
        else:
            return {"pred": X[:, 0], "alpha": alpha, "pdf_shape": X[:, 1]}


class M_FEAT2LABEL_simple(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config.n_channels
        self.n_class = config.n_class
        self.label_cond = sum(config.label_cond_size)
        self.return_only_pred = config.return_only_pred

        ### LAYERS ###
        self.linear_l = nn.Linear(512 + self.label_cond, 64 * config.net_size_scale)
        self.dropout = nn.Dropout(p=config.do_l)
        self.classify_l = nn.Linear(64 * config.net_size_scale, self.n_class)
        self.classify_bias_init = [36.7500]
        self.classify_l.bias.data = torch.Tensor(self.classify_bias_init)

    #        self.transformer = nn.TransformerEncoderLayer(256, nhead = 8)

    def forward(self, X, z):
        # z.size() = [batch_size, label_cond]
        # X.size() = [batch_size, n_epochs, Features]
        X = X.mean(1)
        X = torch.cat((X, z), 1)
        alpha = torch.ones(X.size(0), 120, 1) / 5.0
        # Linear Transform
        X = self.linear_l(F.relu(X))
        # Dropout
        X = self.dropout(X)
        # Classify layer
        X = self.classify_l(F.relu(X))
        X = torch.squeeze(X, 1)
        if self.return_only_pred:
            return X
        else:
            return {"pred": X, "alpha": alpha}


# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        width_mult=1.0,
        inverted_residual_setting=None,
        round_nearest=8,
    ):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, (1, 2)],
                [6, 32, 2, (1, 2)],
                [6, 64, 2, (1, 2)],
                [6, 128, 1, (1, 1)],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(inverted_residual_setting)
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        features = [ConvBNReLU(input_channel, input_channel, stride=(1, 2))]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else (1, 1)
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
        # building last several layers
        #        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=(1,1)))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        #        self.classifier = nn.Sequential(
        #            nn.Dropout(0.2),
        #            nn.Linear(self.last_channel, num_classes),
        #        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        #        x = x.mean([2, 3])
        #        x = self.classifier(x)
        return x


# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
class ConvBNReLU(nn.Sequential):
    def __init__(
        self, in_planes, out_planes, kernel_size=(1, 3), stride=(1, 1), groups=1
    ):
        padding = tuple((np.array(kernel_size) - 1) // 2)
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


# Dilated conv block (conv->bn->relu->maxpool->dropout)
class dilated_conv_block(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 3),
        stride=(1, 1),
        dilation=(1, 1),
        groups=1,
        max_pool=(1, 1),
        drop_chance=0.0,
    ):
        padding = tuple(
            (
                (np.array(kernel_size) - 1) * (np.array(dilation) - 1)
                + np.array(kernel_size)
                - np.array(stride)
            )
            // 2
        )
        if max_pool[1] > 1:
            super(dilated_conv_block, self).__init__(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_pool, stride=max_pool),
                nn.Dropout(p=drop_chance),
            )
        else:
            super(dilated_conv_block, self).__init__(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_chance),
            )


# Additive Attention
class AdditiveAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        """Additive attention module

        Args:
            input_size (int): Size of input
            hidden_size (int): Size of hidden layer
        """
        super(AdditiveAttention, self).__init__()
        self.linear_u = nn.Linear(input_size, hidden_size)
        self.linear_a = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h):
        """Forward call of model

        Args:
            h (Tensor): Input features of size [batch_size, ..., input_size]

        Returns:
            s (Tensor): Summary features
            a (Tensor): Additive attention weights
        """
        # h.size() = [Batch size, Sequence length, Hidden size]
        u = torch.tanh(self.linear_u(h))
        a = F.softmax(self.linear_a(u), dim=1)
        s = torch.sum(a * h, 1)
        return s, a


# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride[1] in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride[1] == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=(1, 1)))
        layers.extend(
            [
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, (1, 1), (1, 1), 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


import os
import numpy as np


class Config(object):
    def __init__(self):
        """Configuration class that contain attributes to set paths and network options.

        Dependencies:
            A txt file named 'profile.txt' in the same directory that matches a set of paths defined below.
        """

        # Get profile
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        with open(os.path.join(__location__, "profile.txt"), "r") as f:
            profile = f.readline()

        # Set local data directory
        if profile == "predator":
            self.data_dir = "H:\\nAge\\"
            self.tmp_dir = "H:\\nAge\\tmp\\"
            self.cfs_ds_path = "G:\\cfs\\datasets\\cfs-visit5-dataset-0.4.0.csv"
            self.cfs_ssc_path = "G:\\cfs\\polysomnography\\annotations-events-profusion"
            self.mros_ds_path = "G:\\mros\\datasets\\mros-visit1-dataset-0.3.0.csv"
            self.mros_ssc_path = (
                "G:\\mros\\polysomnography\\annotations-events-profusion\\visit1"
            )
            self.shhs_ds_path = "H:\\shhs\\datasets\\shhs1-dataset-0.15.0.csv"
            self.shhs_ssc_path = (
                "H:\\shhs\\polysomnography\\annotations-events-profusion\\shhs1"
            )
            self.wsc_ds_path = "G:\\WSC_PLM_ data_all.xlsx"
            self.wsc_ssc_path = "G:\\wsc\\polysomnography\\labels"
            self.stages_ds_path = "H:\\STAGES\\PatientDemographics.xlsx"
            self.stages_ssc_path = "H:\\STAGES\\polysomnograms"
            self.ssc_ds_path = "H:\\SSC\\ssc.xlsx"
            self.ssc_ssc_path = "H:\\SSC\\polysomnography\\labels"
            self.sof_ds_path = "H:\\sof\\datasets\\sof-visit-8-dataset-0.6.0.csv"
            self.sof_ssc_path = "H:\\sof\\polysomnography\\annotations-events-profusion"
            self.hpap_ds_path = (
                "H:\\homepap\\datasets\\homepap-baseline-dataset-0.1.0.csv"
            )
            self.hpap_ssc_path = (
                "H:\\homepap\\polysomnography\\annotations-events-profusion\\lab\\full"
            )
            self.list_split_train = "H:\\nAge\\X_train.csv"
            self.list_split_val = "H:\\nAge\\X_val.csv"
            self.list_split_test = "H:\\nAge\\X_test.csv"
        elif profile == "sherlock":
            self.data_dir = "/scratch/users/abk26/nAge/"
            self.tmp_dir = "/scratch/users/abk26/nAge/tmp/"
            self.cfs_ds_path = (
                "/oak/stanford/groups/mignot/cfs/datasets/cfs-visit5-dataset-0.4.0.csv"
            )
            self.cfs_ssc_path = "/oak/stanford/groups/mignot/cfs/polysomnography/annotations-events-profusion/"
            self.mros_ds_path = "/oak/stanford/groups/mignot/mros/datasets/mros-visit1-dataset-0.3.0.csv"
            self.mros_ssc_path = "/oak/stanford/groups/mignot/mros/polysomnography/annotations-events-profusion/visit1/"
            self.shhs_ds_path = (
                "/oak/stanford/groups/mignot/shhs/datasets/shhs1-dataset-0.14.0.csv"
            )
            self.shhs_ssc_path = (
                "/home/users/abk26/SleepAge/Scripts/data/shhs/polysomnography/shhs1/"
            )
            self.wsc_ds_path = (
                "/home/users/abk26/SleepAge/Scripts/data/WSC_PLM_ data_all.xlsx"
            )
            self.wsc_ssc_path = "/oak/stanford/groups/mignot/psg/WSC_EDF/"
            self.stages_ds_path = (
                "/home/users/abk26/SleepAge/Scripts/data/PatientDemographics.xlsx"
            )
            self.stages_ssc_path = "/oak/stanford/groups/mignot/psg/STAGES/deid/"
            self.ssc_ds_path = "/home/users/abk26/SleepAge/Scripts/data/ssc.xlsx"
            self.ssc_ssc_path = "/oak/stanford/groups/mignot/psg/SSC/APOE_deid/"
            self.list_split_train = os.path.join(self.data_dir, "X_train.csv")
            self.list_split_val = os.path.join(self.data_dir, "X_val.csv")
            self.list_split_test = os.path.join(self.data_dir, "X_test.csv")
        elif profile == "new":
            self.data_dir = os.path.join(__location__, "data")
        else:
            self.data_dir = ""
            self.tmp_dir = ""
            self.cfs_ds_path = ""
            self.mros_ds_path = ""
            self.shhs_ds_path = ""
            self.wsc_ds_path = ""
            self.stages_ds_path = ""

        # Datapaths
        self.model_dir = os.path.join(self.data_dir, "model")
        self.train_dir = os.path.join(self.data_dir, "train")
        self.val_dir = os.path.join(self.data_dir, "val")
        self.interp_dir = os.path.join(self.data_dir, "interpretation")
        self.am_dir = os.path.join(self.data_dir, "am")
        self.test_dir = os.path.join(self.data_dir, "test")
        self.train_cache_dir = os.path.join(self.data_dir, "train_cache")
        self.val_cache_dir = os.path.join(self.data_dir, "val_cache")
        self.test_cache_dir = os.path.join(self.data_dir, "test_cache")
        self.train_F_dir = os.path.join(self.data_dir, "train_F")
        self.val_F_dir = os.path.join(self.data_dir, "val_F")
        self.test_F_dir = os.path.join(self.data_dir, "test_F")
        self.pretrain_dir = os.path.join(self.data_dir, "all")
        self.F_train_dir = os.path.join(self.data_dir, "all_F")

        # Checkpoint
        self.save_dir = self.model_dir
        self.model_F_path = os.path.join(self.model_dir, "modelF")
        self.model_L_path = os.path.join(self.model_dir, "modelL")
        self.model_L_BO_path = self.model_L_path
        self.BO_expe_path = os.path.join(self.model_dir, "exp")

        self.return_only_pred = False
        self.return_pdf_shape = True

        # Pretraining
        # label-config
        self.pre_label = ["age"]  # ['age', 'bmi', 'sex']
        self.pre_label_size = [1]  # [1, 1, 2]
        self.pre_n_class = sum(self.pre_label_size)
        self.pre_only_sleep = 0
        # network-config
        self.n_channels = 12
        self.pre_model_num = 1
        # train-config
        self.pre_max_epochs = 20
        self.pre_patience = 3
        self.pre_batch_size = 32
        self.pre_lr = 1e-3
        self.pre_n_workers = 0
        self.do_f = 0.75
        self.pre_channel_drop = True
        self.pre_channel_drop_prob = 0.1
        self.loss_func = "huber"
        self.only_eeg = 1

        # Training
        # label-config
        self.label = ["age"]
        self.label_cond = []  # ['q_low', 'q_high'] #['q_low', 'q_high', 'bmi', 'sex']
        self.label_cond_size = []  # [7, 7] #[7, 7, 1, 1]
        self.n_class = 1
        # train-config
        self.do_l = 0.5
        self.max_epochs = 200
        self.patience = 20
        self.batch_size = 64
        self.lr = 5e-4
        self.l2 = 1e-5
        self.n_workers = 0
        self.pad_length = 120
        self.cond_drop = False
        self.cond_drop_prob = 0.5
        # network-config
        self.net_size_scale = 4
        self.lstm_n = 1
        self.epoch_size = 5 * 60 * 128
        self.return_att_weights = False
