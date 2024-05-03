import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from preprocess import preprocess_data
from model2 import init_model, get_rays, render_rays

import os

def main():
    train_images, train_poses, test_image, test_pose, focal = preprocess_data()
    H, W = train_images.shape[1:3]

    # convert from numpy to torch tensor
    train_images = torch.tensor(train_images, dtype=torch.float32)
    train_poses = torch.tensor(train_poses, dtype=torch.float32)
    test_image = torch.tensor(test_image, dtype=torch.float32)
    test_pose = torch.tensor(test_pose, dtype=torch.float32)
    focal = torch.tensor(focal, dtype=torch.float64)

    model = init_model()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    N_samples = 64
    N_iters = 1000
    psnrs = []
    iternums = []
    i_plot = 25

    import time
    t = time.time()
    for i in range(N_iters + 1):

        img_i = np.random.randint(train_images.shape[0])
        target = train_images[img_i]
        pose = train_poses[img_i]
        rays_o, rays_d = get_rays(H, W, focal, pose)
        optimizer.zero_grad()
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples, rand=True)
        loss = torch.mean((rgb - target) ** 2)
        loss.backward()
        optimizer.step()

        if i % i_plot == 0:
            print(i, (time.time() - t) / i_plot, 'secs per iter')
            t = time.time()

            # Render the holdout view for logging
            rays_o, rays_d = get_rays(H, W, focal, test_pose)
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
            loss = torch.mean((rgb - test_image) ** 2)
            psnr = -10. * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            # plotting
            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(rgb.detach().cpu().numpy())
            plt.title(f'Iteration: {i}')
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title('PSNR')
            # plt.show()

            # save figure to directory
            results_dir = './results/'
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, f'plot_{i}.png'))

    print('Done')

if __name__ == "__main__":
    main()
