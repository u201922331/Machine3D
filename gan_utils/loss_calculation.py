import torch


def discriminator_loss(criterion, gen, dis, _in, _target):
    fake = gen(_in)
    fake_pred = dis(_in, fake)
    fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))

    real_pred = dis(_in, _target)
    real_loss = criterion(real_pred, torch.ones_like(real_pred))

    return (real_loss + fake_loss) / 2


def generator_loss(criterion, gen, dis, _in, _target, l):
    fake = gen(_in)
    fake_pred = dis(_in, fake)

    real_pred = dis(_in, _target)

    fake_loss = criterion(fake_pred, torch.ones_like(fake_pred))
    target_loss = torch.mean(torch.abs(_target - fake))

    return fake_loss + l * target_loss
