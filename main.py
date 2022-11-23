from gan_utils.generator import Generator
from gan_utils.discriminator import Discriminator
from gan_utils.custom_loader import load
from gan_utils.loss_calculation import *
from utils.constants import *
import torch
from tqdm.auto import tqdm


def main():
    ltr, lts, mtr_x, mtr_y, mts_x, mts_y = load('./datasets/dataset.npy')

    gen = Generator()
    dis = Discriminator()
    gen_opt = torch.optim.Adam(gen.parameters(), lr=learning_rate)
    dis_opt = torch.optim.Adam(dis.parameters(), lr=learning_rate)

    cur_step = 0
    mean_gen_loss = 0
    mean_dis_loss = 0
    for i in range(epochs):
        if i % 5 == 0:
            torch.save(gen.state_dict(), PATH_CHECKPOINTS_GEN + f'/gen_{i}.pth')
            torch.save(dis.state_dict(), PATH_CHECKPOINTS_DIS + f'/dis_{i}.pth')

        for in_data, tg_data in tqdm(ltr):
            dis_opt.zero_grad()
            fake = gen(in_data)
            o_fake = dis(in_data, fake)
            o_real = dis(in_data, tg_data)
            dis_loss = discriminator_loss(criterion, o_real, o_fake)
            dis_loss.backward(retain_graph=True)
            dis_opt.step()

            gen_opt.zero_grad()
            fake = gen(in_data)
            o_fake = dis(in_data, fake)
            o_real = dis(in_data, tg_data)
            gen_loss = generator_loss(criterion, o_real, o_fake, tg_data, fake, LAMBDA)
            gen_loss.backward()
            gen_opt.step()

            mean_dis_loss += dis_loss.item() / display_step
            mean_gen_loss += gen_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print(f'Step {cur_step}: Generator loss: {mean_gen_loss}, Discriminator loss: {mean_dis_loss}')
                mean_gen_loss = 0
                mean_dis_loss = 0
            cur_step += 1

    torch.save(dis.state_dict(), PATH_MODELS + '/dis.pth')
    torch.save(gen.state_dict(), PATH_MODELS + '/gen.pth')


if __name__ == '__main__':
    main()
