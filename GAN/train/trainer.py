import torch.nn as nn


class WightInitializer:
    def __init__(self, conv = (0, 0.02), bias = 0.01, momentum = 0.1):
        self.bias = bias
        self.conv_std, self.conv_mean = conv
        self.momentum = momentum

    def __call__(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, self.conv_mean, self.conv_std)
            m.bias.data.fill_(self.bias)
        if type(m) == nn.BatchNorm2d:
            m.momentum = self.momentum


class WeightClipper:
    def __init__(self, interval):
        self.min_value, self.max_value = interval

    def __call__(self, m):
        if hasattr(m, 'weight'):
            w = m.weight.data
            w[w > self.max_value] = self.max_value
            w[w < self.min_value] = self.min_value