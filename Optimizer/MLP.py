'''
All of the details are available at:
https://wangjingyao07.github.io/Awesome-Meta-Learning-Platform/
'''

import math
import copy
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_units, init_scale=1.0):
        super(MLP, self).__init__()

        self._n_units = copy.copy(n_units)
        # copy the n_units from the parameter
        self._layers = []
        # initialize the layers of MLP
        for i in range(1, len(n_units)):
            layer = nn.Linear(n_units[i-1], n_units[i], bias=False)
            # the size of each layer
            variance = math.sqrt(2.0 / (n_units[i-1] + n_units[i]))
            layer.weight.data.normal_(0.0, init_scale * variance)
            # initialize the layer with normal distribution of mean and variance
            self._layers.append(layer)
            # add the layer into self._layers

            name = 'fc%d' % i
            if i == len(n_units) - 1:
                name = 'fc'  # the prediction layer is just called fc
            self.add_module(name, layer)
            # add the name and layer into torch module

    def forward(self, x):
        x = x.view(-1, self._n_units[0])
        # change the data into appropriate size
        out = self._layers[0](x)
        for layer in self._layers[1:]:
            out = F.relu(out)
            out = layer(out)
        return out
