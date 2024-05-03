import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, widgets
import imageio
from tqdm import tqdm

L_embed = 6

import torch.nn as nn

def init_model(D=8, W=256):
    class InitModel(nn.Module):
        def __init__(self):
            super(InitModel, self).__init__()

            self.relu = nn.ReLU()
            self.dense = nn.ModuleList([nn.Linear(3 + 3 * 2 * L_embed, W)])
            for i in range(1, D):
                if i % 4 == 1 and i > 1:
                    self.dense.append(nn.Linear(W+3 + 3 * 2 * L_embed, W))
                else:
                    self.dense.append(nn.Linear(W, W))
            self.final_layer = nn.Linear(W, 4)

        def forward(self, inputs):
            outputs = inputs
            for i, layer in enumerate(self.dense):
                outputs = layer(outputs)
                outputs = self.relu(outputs)
                if i % 4 == 0 and i > 0:
                    outputs = torch.cat([outputs, inputs], -1)
            outputs = self.final_layer(outputs)
            return outputs

    return InitModel()

def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32))
    dirs = torch.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = torch.unsqueeze(torch.unsqueeze(c2w[:3, -1], 0), 1).expand_as(rays_d)

    return rays_o, rays_d

def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    def batchify(fn, chunk=1024*32):
        return lambda inputs : torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    def posenc(x):
        rets = [x]
        for i in range(L_embed):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2.**i * x))
        return torch.cat(rets, -1)

    z_vals = torch.linspace(near, far, N_samples)
    if rand:
        z_vals = torch.rand(*rays_o.shape[:-1], N_samples) * (far - near) / N_samples + z_vals
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)

    pts_flat = pts.view(-1, 3)
    pts_flat = posenc(pts_flat)
    raw = batchify(network_fn)(pts_flat)
    raw = raw.view(pts.shape[:-1] + (4,))

    sigma_a = F.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3])

    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.full_like(z_vals[..., :1], 1e10)], -1)
    alpha = 1. - torch.exp(-sigma_a * dists)
    weights = alpha * torch.cumprod(1. - alpha + 1e-10, -1)

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    return rgb_map, depth_map, acc_map

# Define transformation matrices as PyTorch tensors
trans_t = lambda t: torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1],
], dtype=torch.float32)

rot_phi = lambda phi: torch.tensor([
    [1, 0, 0, 0],
    [0, torch.cos(phi), -torch.sin(phi), 0],
    [0, torch.sin(phi), torch.cos(phi), 0],
    [0, 0, 0, 1],
], dtype=torch.float32)

rot_theta = lambda th: torch.tensor([
    [torch.cos(th), 0, -torch.sin(th), 0],
    [0, 1, 0, 0],
    [torch.sin(th), 0, torch.cos(th), 0],
    [0, 0, 0, 1],
], dtype=torch.float32)

def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi / 180. * np.pi) @ c2w
        c2w = rot_theta(theta / 180. * np.pi) @ c2w
        c2w = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32) @ c2w
        return c2w

def create_interactive_plot(H, W, focal, model, N_samples):
    
    def f(theta, phi, radius):
        c2w = pose_spherical(theta, phi, radius)
        rays_o, rays_d = get_rays(H, W, focal, c2w[:3, :4])
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        img = np.clip(rgb, 0, 1)

        plt.figure(2, figsize=(20, 6))
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

    sldr = lambda v, mi, ma: widgets.FloatSlider(
        value=v,
        min=mi,
        max=ma,
        step=.01,
    )

    names = [
        ['theta', 100., 0., 360],
        ['phi', -30., -90, 0],
        ['radius', 4., 3., 5.],
    ]

    interactive_plot = interactive(f, **{name: sldr(v, mi, ma) for name, v, mi, ma in names})
    output = interactive_plot.children[-1]
    output.layout.height = '350px'

    return interactive_plot

def generate_video(model, H, W, focal, N_samples, output_file='video.mp4'):
    theta_range=(0., 360.)
    frames = []
    for th in tqdm(np.linspace(theta_range[0], theta_range[1], 120, endpoint=False)):
        c2w = pose_spherical(th, phi=-30., radius=4.)
        rays_o, rays_d = get_rays(H, W, focal, c2w[:3, :4])
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        frames.append((255 * np.clip(rgb, 0, 1)).astype(np.uint8))

    imageio.mimwrite(output_file, frames, fps=30, quality=7)
    print(f"Video saved to '{output_file}'")

# Usage example:
# generate_video(model, pose_spherical, get_rays, render_rays, H, W, focal, N_samples)
