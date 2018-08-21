import torch


def wgan_generator_loss(critique, generator, z):
    return -torch.mean(critique(generator(z)))

def wgan_critique_loss(critique, generator, x, z):
    -torch.mean(critique(x) - critique(generator(z)))