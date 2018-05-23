# -*- coding: utf-8 -*-
import torch.nn as nn

# DCGAN generator
class Generator(nn.Module):

    def __init__(self, width=256, n_ch=3, latent_size=100, n_ker_first=128):
        super(Generator, self).__init__()
        self.nhidden = 5
        self.n_ker_first = n_ker_first
        self.first_layer_width = int(width/(pow(2,self.nhidden-1)))
        self.first_layer = nn.Linear(latent_size, pow(self.first_layer_width,2)*self.n_ker_first*8)
        self.model = nn.Sequential(
                        nn.ReLU(),
                        nn.ConvTranspose2d(n_ker_first * 8, n_ker_first*4, 4, padding=1, stride=2),
                        nn.BatchNorm2d(n_ker_first * 4),
                        nn.ReLU(),
                        nn.ConvTranspose2d(n_ker_first * 4, n_ker_first*2, 4, padding=1, stride=2),
                        nn.BatchNorm2d(n_ker_first * 2),
                        nn.ReLU(),
                        nn.ConvTranspose2d(n_ker_first * 2, n_ker_first, 4, padding=1, stride=2),
                        nn.BatchNorm2d(n_ker_first),
                        nn.ReLU(),
                        nn.ConvTranspose2d(n_ker_first, n_ch, 4, padding=1, stride=2),
                        nn.BatchNorm2d(n_ch),
                        nn.Sigmoid()
                    )
        self.apply(init_wights)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.model(x.view((-1, self.n_ker_first * 8, self.first_layer_width, self.first_layer_width)))
        return x

# WDCGAN critique
class Critique(nn.Module):

    def __init__(self, width=256, n_ch=3, n_ker_first=128):
        super(Critique, self).__init__()
        self.size_last = width//16
        self.model = nn.Sequential(
                        nn.Conv2d(n_ch, n_ker_first, 4, stride=2, padding=1),
                        nn.BatchNorm2d(n_ker_first),
                        nn.LeakyReLU(0.02),
                        nn.Conv2d(n_ker_first, n_ker_first * 2, 4, stride=2, padding=1),
                        nn.BatchNorm2d(n_ker_first * 2),
                        nn.LeakyReLU(0.02),
                        nn.Conv2d(n_ker_first * 2, n_ker_first * 4, 4, stride=2, padding=1),
                        nn.BatchNorm2d(n_ker_first * 4),
                        nn.LeakyReLU(0.02),
                        nn.Conv2d(n_ker_first * 4, n_ker_first * 8, 4, stride=2, padding=1),
                        nn.BatchNorm2d(n_ker_first * 8),
                        nn.LeakyReLU(0.02),
                        nn.Conv2d(n_ker_first * 8, 1, self.size_last, stride=2)
                    )
        self.apply(init_wights)

    def forward(self, x):
        return self.model(x).view((-1,1))

def init_wights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, 0, 0.02)
        m.bias.data.fill_(0.01)
    if type(m) == nn.BatchNorm2d:
        m.momentum = 0.4