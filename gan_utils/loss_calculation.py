import torch


def discriminator_loss(criterion, real, fake):
    dis_real_loss = criterion(real, torch.ones_like(real))
    dis_fake_loss = criterion(fake, torch.zeros_like(fake))

    return (dis_fake_loss + dis_real_loss) / 2


def generator_loss(criterion, real_out, fake_out, target, fake, l):
    dis_fake_loss = criterion(fake_out, torch.ones_like(fake_out))
    target_loss = torch.mean(torch.abs(target - fake))

    return dis_fake_loss + l * target_loss
