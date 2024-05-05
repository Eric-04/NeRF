import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from preprocess import preprocess_data
from model import init_model, get_rays, render_rays, create_interactive_plot, generate_video

import os

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bouncingballs")
    args = parser.parse_args()
    return args

def main(args):
    nerf_obj = args.dataset
    train_images, train_poses, test_image, test_pose, focal = preprocess_data(nerf_obj)
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
    N_iters = 25
    psnrs = []
    iternums = []
    i_plot = 25

    # create interactive plot
    if os.path.exists(f'./model/{nerf_obj}.pth'):

        print("model pre-trained. creating interactive plot...")
        model.load_state_dict(torch.load(f'./model/{nerf_obj}.pth'))
        model.eval()

        create_interactive_plot(H, W, focal, model, N_samples=N_samples)
        return

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
            nerf_obj_results_dir = results_dir + f'{nerf_obj}'
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(nerf_obj_results_dir, exist_ok=True)
            plt.savefig(os.path.join(nerf_obj_results_dir, f'plot_{i}.png'))
    
    # save model to directory
    model_dir = './model/'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, f'{nerf_obj}.pth'))

    # generate video
    video_dir = './video/'
    os.makedirs(video_dir, exist_ok=True)
    generate_video(model, H, W, focal, N_samples, output_file=f'{video_dir}{nerf_obj}.mp4')

    # create interactive plot
    create_interactive_plot(H, W, focal, model, N_samples=N_samples)

    print('Done')

if __name__ == "__main__":
    args = parseArguments()
    main(args)
