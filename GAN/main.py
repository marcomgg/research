# -*- coding: utf-8 -*-
from visdom import Visdom
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.datasets as dst
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.models import *
import time

def update_window(vz: Visdom, win, x, y):
    if vz.win_exists(win):
        vz.line(
            X=np.array(x),
            Y=np.array(y),
            win=win,
            update='append'
        )
        return win
    else:
        win = vz.line(
            X=np.array(x),
            Y=np.array(y)
            )
        return win

viz = Visdom()

startup_sec = 1
while not viz.check_connection() and startup_sec > 0:
    time.sleep(0.1)
    startup_sec -= 0.1
assert viz.check_connection(), 'No connection could be formed quickly'

w = 128; h = w; latent_size = 100
batch_size = 128; num_workers = 4; k = 10

root = '/mnt/DATA/TorchData'
trans = transforms.Compose([transforms.Resize((w, h)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
train_set = dst.LSUN(root=root, classes=['bridge_train'], transform=trans)
data = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

it = iter(data)
for i in range(10):
    first = next(it)[0]
    viz.image(np.squeeze(first[1,:,:].numpy()))

critique = Critique(w)
generator = Generator(w)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

critique.to(device)
generator.to(device)

num_epochs = 1

optimizer_generator = Adam(generator.parameters())
optimizer_critique = Adam(critique.parameters())

win_loss_critique = []
win_loss_generator = []

for i in range(num_epochs):
    for i, batch in enumerate(data, 0):
        x = batch[0]
        z = torch.randn((batch_size, latent_size))
        x, z = x.to(device), z.to(device)

        generator.zero_grad()
        critique.zero_grad()

        loss_critique = -torch.mean(critique(generator(z)) - critique(x))
        loss_critique.backward()
        optimizer_critique.step()

        if i%k == 0:
            generator.zero_grad()
            z = torch.randn((batch_size, latent_size))
            z = z.to(device)
            loss_generator = torch.mean(critique(generator(z)))
            loss_generator.backward()
            optimizer_generator.step()
        break




