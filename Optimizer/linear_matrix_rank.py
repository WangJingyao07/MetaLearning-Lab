'''
All of the details are available at:
https://wangjingyao07.github.io/Awesome-Meta-Learning-Platform/
'''

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, n_units, init_scale=1.0):
        super(Linear, self).__init__()

        self.dim = n_units
        # copy the n_units from the parameter
        self._layers = []
        # initialize the layers of MLP
        layer = nn.Linear(1, n_units, bias=False)
        # the size of each layer
        variance = math.sqrt(2.0 / (1 + self.dim))
        layer.weight.data.normal_(0.0, init_scale * variance)
        # initialize the layer with normal distribution of mean and variance
        self._layers.append(layer)
        # add the layer into self._layers

        name = 'fc'  # the prediction layer is just called fc
        self.add_module(name, layer)
        # add the name and layer into torch module

    def forward(self, x, layers=None):
        if layers is None:
            layers = self._layers
            out = layers[0](x)
        else:
            w=list(layers)[0]
            out = F.linear(x, w)
        return out
