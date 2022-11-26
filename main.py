import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import *
from utils.load_as_tensor import load_binvox_as_tensor
from utils.save_voxel import save_voxel
from gan_utils.generator import Generator
from gan_utils.discriminator import Discriminator
from gan_utils.training import train


torch.manual_seed(tseed_)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    dataset = np.load(DATASETS+'dataset.npy', allow_pickle=True).item()

    mdl_tr_br = [torch.from_numpy(np.array([model]) * 2 - 1).float().to(device) for model in dataset['train']['broken']]
    mdl_tr_tg = [torch.from_numpy(np.array([model]) * 2 - 1).float().to(device) for model in dataset['train']['target']]

    dataloader = DataLoader(list(zip(mdl_tr_br, mdl_tr_tg)), batch_size=batch_size, shuffle=True)

    gen = Generator().to(device)
    dis = Discriminator().to(device)

    should_load = True if input('Load backups? (y/n): ').lower() == 'y' else False
    if should_load:
        gen_chk_path = input('Generator: ')
        dis_chk_path = input('Discriminator: ')

        gen.load_state_dict(torch.load(gen_chk_path, map_location=device))
        dis.load_state_dict(torch.load(dis_chk_path, map_location=device))

    gen_optimizer = torch.optim.Adam(gen.parameters(), lr)
    dis_optimizer = torch.optim.Adam(dis.parameters(), lr)

    should_train = True if input('Train? (y/n): ').lower() == 'y' else False
    x_steps, y_gen, y_dis = [], [], []
    if should_train:
        gen, dis, x_steps, y_gen, y_dis = train(gen, dis, gen_optimizer, dis_optimizer, dataloader, 100)

        should_save = True if input('Want to save progress? (y/n): ') else False
        if should_save:
            torch.save(dis.state_dict(), MODELS+'dis.pth')
            torch.save(gen.state_dict(), MODELS+'gen.pth')

    obj = load_binvox_as_tensor(MESHES+'jarron.binvox', device)
    obj = torch.stack([obj, ])
    pcs = gen(obj)

    obj = obj[0][0].detach().cpu().numpy() > 0
    pcs = pcs[0][0].detach().cpu().numpy() > 0

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(obj, facecolors='darkslategray', alpha=0.5)
    ax.voxels(pcs & ~obj, facecolors='orange', edgecolors='darkorange')
    plt.show()

    should_save_generated = True if input('Want to save generated? (y/n): ') else False
    if should_save_generated:
        export_path_broken = input('Path of broken: ')
        export_path_result = input('Path of generated: ')
        save_voxel(export_path_broken, obj)
        save_voxel(export_path_result, pcs & ~obj)


if __name__ == '__main__':
    main()
