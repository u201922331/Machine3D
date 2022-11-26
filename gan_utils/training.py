from tqdm.auto import tqdm
from utils.constants import *
from gan_utils.loss_calculation import *
import torch


def train(gen, dis, gen_opt, dis_opt, _dataloader, _epochs, _lambda=100, _display_step=10, _save_each=5):
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    steps = []
    ys_gen = []
    ys_dis = []

    for epoch in range(_epochs):
        if epoch % _save_each == 0 or epoch == 0:
            print('--Creating checpoint...--')
            torch.save(gen.state_dict(), MODELS + f'gen_{epoch}.chk')
            torch.save(dis.state_dict(), MODELS + f'dis_{epoch}.chk')

        for _in, _target in tqdm(_dataloader):
            dis_opt.zero_grad()
            dis_loss = discriminator_loss(criterion, gen, dis, _in, _target)
            dis_loss.backward(retain_graph=True)
            dis_opt.step()

            gen_opt.zero_grad()
            gen_loss = generator_loss(criterion, gen, dis, _in, _target, _lambda)
            gen_loss.backward()
            gen_opt.step()

            mean_discriminator_loss += dis_loss.item() / _display_step
            mean_generator_loss += gen_loss.item() / _display_step
            if cur_step % _display_step == 0 and cur_step > 0:
                print(f'Step {cur_step} - Epoch {epoch + 1}:'
                      f'Generator loss: {mean_generator_loss},'
                      f'Discriminator loss: {mean_discriminator_loss}')
                steps.append(cur_step)

                ys_dis.append(mean_discriminator_loss)
                ys_gen.append(mean_generator_loss)

                mean_discriminator_loss = 0
                mean_generator_loss = 0
            cur_step += 1

    return gen, dis, steps, ys_gen, ys_dis
