# -*- coding: utf-8 -*-

from visdom import Visdom
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.datasets as dst
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

viz = Visdom()

startup_sec = 1
while not viz.check_connection() and startup_sec > 0:
    time.sleep(0.1)
    startup_sec -= 0.1
assert viz.check_connection(), 'No connection could be formed quickly'


def reltol(w, wprev):
    return (w-wprev).norm()/wprev.norm()


root = '/mnt/DATA/TorchData'
trans = transforms.Compose([transforms.ToTensor()])
train_set = dst.CIFAR10(root = root, train=True, transform=trans, download=True)
first = torch.Tensor(train_set.train_data[800, :, :, :]).mean(2)
sobelx = torch.Tensor([[1.0,0.0,-1.0], [2.0, 0, 2.0], [1.0,0.0,-1.0]])
gx = F.conv2d(Variable(first.view([1, 1, 32, 32])), Variable(sobelx.view([1, 1, 3, 3])))
sobely = sobelx.transpose(0,1).clone()
gy = F.conv2d(Variable(first.view([1, 1, 32, 32])), Variable(sobely.view([1, 1, 3, 3])))
norm = (gx.pow(2) + gy.pow(2)).sqrt()
norm = norm/norm.max()
prova = torch.Tensor([1.0,4,5])
viz.line(prova)

x = np.tile(np.arange(1, 101), (100, 1))
y = x.transpose()
X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
viz.contour(X=X, opts=dict(colormap='Viridis'))

# surface
viz.surf(X=X, opts=dict(colormap='Hot'))

# N = 1000;
# D = 2;
# X = torch.rand(N, D)
# W = torch.rand(D,1)
# B = 10
# x = Variable(X, requires_grad = False)
# y = Variable(X.mm(W)+B, requires_grad = False);
#
# w = Variable(torch.rand(D,1), requires_grad = True)
# b = Variable(9*torch.ones(1), requires_grad = True)
#
# niter = 1000
# loss_array = np.zeros(niter)
# reltol_array = np.zeros(niter)
# for i in range(niter):
#    ypred = x.mm(w) + b
#    loss = (y-ypred).pow(2).mean()
#    loss_array[i] = loss.data
#    loss.backward()
#    wprev = w.data.clone()
#    w.data -= 0.01*w.grad.data
#    b.data -= 0.01*b.grad.data
#    w.grad.data.zero_()
#    b.grad.data.zero_()
#    reltol_array[i] = reltol(w.data, wprev)
#
# viz.line(loss_array)
# viz.line(reltol_array)
#
#
#
    