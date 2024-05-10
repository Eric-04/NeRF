import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.widgets import Slider
import cv2
import os

L_embed = 6

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
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy')
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
    c2w = trans_t(torch.tensor(radius))
    c2w = rot_phi(torch.tensor(phi / 180. * np.pi)) @ c2w
    c2w = rot_theta(torch.tensor(theta / 180. * np.pi)) @ c2w
    c2w = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32) @ c2w
    return c2w

def create_interactive_plot(H, W, focal, model, N_samples, nerf_obj,
                            theta_intv = 18, phi_intv = 9, radius_intv = 4, pretrained = True):
    
    def f(theta, phi, radius):
        c2w = pose_spherical(theta, phi, radius)
        rays_o, rays_d = get_rays(H, W, focal, c2w[:3, :4])
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        img = np.clip(rgb.detach().cpu().numpy(), 0, 1)
        return img

    def generate_frames():
        frames = {}

        # if frames have been generated 
        video_dir = './frames/'
        os.makedirs(video_dir, exist_ok=True)
        filename = f'{video_dir}{nerf_obj}:{theta_intv},{phi_intv},{radius_intv}.npy'
        if os.path.exists(filename):
            frames = np.load(filename, allow_pickle=True).item()
        else:
            for theta in tqdm(np.linspace(0., 360., theta_intv+1)):
                for phi in np.linspace(-90., 0., phi_intv+1):
                    for radius in np.linspace(3., 5., radius_intv+1):
                        key = f'{theta},{phi},{radius}'
                        frames[key] = f(theta, phi, radius)
            np.save(filename, frames)
                
        
        return frames
    
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    ax.imshow(f(120, -30, 4))

    ax_theta = plt.axes([0.1, 0.1, 0.65, 0.03])
    ax_phi = plt.axes([0.1, 0.05, 0.65, 0.03])
    ax_radius = plt.axes([0.1, 0.15, 0.65, 0.03])

    sldr_theta = Slider(ax_theta, 'Theta', 0, 360, valinit=120, valstep=360/phi_intv)
    sldr_phi = Slider(ax_phi, 'Phi', -90, 0, valinit=-30, valstep=90/phi_intv)
    sldr_radius = Slider(ax_radius, 'Radius', 3, 5, valinit=4, valstep=2/radius_intv)

    if pretrained == True: frames = generate_frames()

    def update(val):
        theta = sldr_theta.val
        phi = sldr_phi.val
        radius = sldr_radius.val
        if pretrained == False:
            ax.imshow(f(theta, phi, radius))
        else:
            ax.imshow(frames[f'{theta},{phi},{radius}'])


    sldr_theta.on_changed(update)
    sldr_phi.on_changed(update)
    sldr_radius.on_changed(update)

    plt.show()

def generate_video(model, H, W, focal, N_samples, output_file='video.mp4'):
    frames = []
    fps = 30

    # theta range (0 - 360), in 120 intervals
    for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
        c2w = pose_spherical(th, phi=-30., radius=4.)
        rays_o, rays_d = get_rays(H, W, focal, c2w[:3, :4])
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        frames.append((255 * np.clip(rgb.detach().numpy(), 0, 1)).astype(np.uint8))

    # Determine the size of the first frame
    height, width, _ = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write frames to video
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    print(f"Video saved to '{output_file}'")