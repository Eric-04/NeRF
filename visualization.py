import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from IPython.display import HTML
from base64 import b64encode
import imageio
from ipywidgets import interactive, widgets

# Load data
data = np.load('tiny_nerf_data.npz')
images = torch.tensor(data['images'], dtype=torch.float32)
poses = torch.tensor(data['poses'], dtype=torch.float32)
focal = torch.tensor(data['focal'], dtype=torch.float32)
H, W = images.shape[1:3]
print(images.shape, poses.shape, focal)

# Select test image and pose
testimg, testpose = images[101], poses[101]

# Trim data for training
images = images[:100, ..., :3]
poses = poses[:100]

# Display test image
plt.imshow(testimg.numpy())
plt.show()

def posenc(x):
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i * x))
    return torch.cat(rets, dim=-1)

L_embed = 6
embed_fn = posenc

class Model(nn.Module):
    def __init__(self, D=8, W=256):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.dense = nn.Linear(W, W)
        self.fc_out = nn.Linear(W, 4)
        
    def forward(self, inputs, D):
        outputs = inputs
        for i in range(D):
            outputs = self.dense(outputs)
            if i % 4 == 0 and i > 0:
                outputs = torch.cat([outputs, inputs], dim=-1)
        outputs = self.fc_out(outputs)
        return outputs

def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32))
    dirs = torch.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -torch.ones_like(i)], dim=-1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, -1].unsqueeze(0).expand_as(rays_d)
    return rays_o, rays_d

def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    def batchify(fn, chunk=1024*32):
        return lambda inputs : torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], dim=0)
    
    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples) 
    if rand:
        z_vals += torch.rand(list(rays_o.shape[:-1]) + [N_samples]) * (far-near)/N_samples
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    # Run network
    pts_flat = pts.view(-1, 3)
    pts_flat = embed_fn(pts_flat)
    raw = batchify(network_fn)(pts_flat)
    raw = raw.view(pts.shape[:-1] + (4,))
    
    # Compute opacities and colors
    sigma_a = nn.functional.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3]) 
    
    # Do volume rendering
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.tensor([1e10], device=z_vals.device).expand(z_vals[..., :1].shape)], dim=-1) 
    alpha = 1. - torch.exp(-sigma_a * dists)  
    weights = alpha * torch.cumprod(1. - alpha + 1e-10, dim=-1)
    
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2) 
    depth_map = torch.sum(weights * z_vals, dim=-1) 
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map

model = Model()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

N_samples = 64
N_iters = 1000
psnrs = []
iternums = []
i_plot = 25

t = time.time()
for i in range(N_iters+1):
    
    img_i = np.random.randint(images.shape[0])
    target = torch.tensor(images[img_i], dtype=torch.float32)
    pose = torch.tensor(poses[img_i], dtype=torch.float32)
    rays_o, rays_d = get_rays(H, W, focal, pose)
    
    optimizer.zero_grad()
    rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples, rand=True)
    loss = torch.mean((rgb - target)**2)
    loss.backward()
    optimizer.step()
    
    if i % i_plot == 0:
        print(i, (time.time() - t) / i_plot, 'secs per iter')
        t = time.time()
        
        # Render the holdout view for logging
        rays_o, rays_d = get_rays(H, W, focal, testpose)
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        loss = torch.mean((rgb - testimg)**2)
        psnr = -10. * torch.log10(loss)

        psnrs.append(psnr.item())
        iternums.append(i)
        
        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.imshow(rgb.cpu().detach().numpy())
        plt.title(f'Iteration: {i}')
        plt.subplot(122)
        plt.plot(iternums, psnrs)
        plt.title('PSNR')
        plt.show()

print('Done')

def trans_t(t):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ], dtype=torch.float32)

def rot_phi(phi):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, torch.cos(phi), -torch.sin(phi), 0],
        [0, torch.sin(phi), torch.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)

def rot_theta(th):
    return torch.tensor([
        [torch.cos(th), 0, -torch.sin(th), 0],
        [0, 1, 0, 0],
        [torch.sin(th), 0, torch.cos(th), 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.tensor([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def f(**kwargs):
    c2w = pose_spherical(**kwargs)
    rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
    rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
    img = torch.clamp(rgb, 0, 1)
    
    plt.figure(2, figsize=(20,6))
    plt.imshow(img.cpu().detach().numpy())
    plt.show()

sldr = lambda v, mi, ma: widgets.FloatSlider(
    value=v,
    min=mi,
    max=ma,
    step=.01,
)

names = [
    ['theta', [100., 0., 360]],
    ['phi', [-30., -90, 0]],
    ['radius', [4., 3., 5.]],
]

interactive_plot = interactive(f, **{s[0] : sldr(*s[1]) for s in names})
output = interactive_plot.children[-1]
output.layout.height = '350px'
interactive_plot

frames = []
for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
    c2w = pose_spherical(th, -30., 4.)
    rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
    rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
    img = torch.tensor(255 * np.clip(rgb, 0, 1), dtype=torch.uint8).numpy()
    frames.append(img)

f = 'video.mp4'
imageio.mimwrite(f, frames, fps=30, quality=7)

mp4 = open('video.mp4', 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML("""
<video width=400 controls autoplay loop>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
