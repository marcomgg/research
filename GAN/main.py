# -*- coding: utf-8 -*-
import sys
sys.path.extend(['GAN', 'GAN'])
from visdom import Visdom
import numpy as np
import torch
import torchvision.datasets as dst
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop
from models.models import *
from train.trainer import *
from train.losses import  *
from util import visualmanager as vm

#%%
w = 128; h = w; latent_size = 100
batch_size = 64; num_workers = 4; k = 5

print('Loading data')
root = '/mnt/DATA/TorchData'
trans = transforms.Compose([transforms.Resize((w, h)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
train_set = dst.LSUN(root=root, classes=['bridge_train'], transform=trans)
data = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, drop_last= True, shuffle=True)

#%%
viz = Visdom()
vm.check_connection(viz)

#%%
it = iter(data)
for i in range(10):
    first = next(it)[0]
    viz.image(np.squeeze(first[1,:,:].numpy()), env='marco')

#%%
critique = Critique(w)
generator = Generator(w)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Loading model')
critique.to(device)
generator.to(device)

print('Initializing weights')
initializer = WightInitializer()
critique.apply(initializer)
generator.apply(initializer)

num_epochs = 30

optimizer_generator = RMSprop(generator.parameters(), lr=0.00005)
optimizer_critique = RMSprop(critique.parameters(), lr=0.00005)

win_loss_critique = []
win_loss_generator = []

c=0.01
clipper = WeightClipper((-c, c))

#%%
for epoch in range(num_epochs):
    for i, batch in enumerate(data, 0):
        x = batch[0]
        z = torch.randn((batch_size, latent_size))
        x, z = x.to(device), z.to(device)

        generator.zero_grad()
        critique.zero_grad()

        loss_critique = wgan_critique_loss(critique, generator, x, z)
        win_loss_critique = vm.update_line(viz, win_loss_critique,i, loss_critique.data.tolist())
        loss_critique.backward()
        optimizer_critique.step()
        critique.apply(clipper)

        if i%k == 0:
            generator.zero_grad()
            z = torch.randn((batch_size, latent_size))
            z = z.to(device)
            loss_generator = wgan_generator_loss(critique, generator, z)
            win_loss_generator = vm.update_line(viz, win_loss_generator, i, loss_generator.data.tolist())
            loss_generator.backward()
            optimizer_generator.step()

    z = torch.randn((1, latent_size))
    z = z.to(device)
    im = generator.forward(z).data.cpu().numpy()
    viz.image(np.squeeze(im))






