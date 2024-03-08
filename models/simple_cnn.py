import torch.nn as nn
from models.ds_modules.basemodel import BaseModel
from models.conv_sn_chen import conv_spectral_norm
import torch


def make_conv_layer(in_channels, out_channels, kernel_size, padding, bias, spectral_norm):
    """
    """
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
    if (spectral_norm):
        return conv_spectral_norm(conv_layer, sigma=1.0, n_power_iterations=1)
    else:
        return conv_layer


class SimpleCNN(BaseModel):
    """
    params:
    - activation_type
    - spline_init
    - spline_size
    - spline_range
    - save_memory
    """
    def __init__(self, num_layers=4, num_channels=64, kernel_size=3, padding=1, bias=True, spectral_norm=False, **params):
        
        super().__init__(**params)

        layers = []

        # First Block
        layers.append(make_conv_layer(1, num_channels, kernel_size, padding, bias, spectral_norm))
        layers.append(self.init_activation(('conv', num_channels), bias=False))

        # Middle blocks
        for _ in range(num_layers-2):
            layers.append(make_conv_layer(num_channels, num_channels, kernel_size, padding, bias, spectral_norm))
            layers.append(self.init_activation(('conv', num_channels), bias=False))

        # Last convolutional layer
        layers.append(make_conv_layer(num_channels, 1, kernel_size, padding, True, spectral_norm))

        self.net = nn.Sequential(*layers)

        self.initialization(init_type='custom_normal')
        self.num_params = self.get_num_params()


    def forward(self, x):
        """ """
        out = self.net(x)
        return out