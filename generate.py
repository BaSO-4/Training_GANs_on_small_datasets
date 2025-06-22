import os, math
import torch
from torchvision import utils as vutils

from networks import MappingNetwork, Generator, get_w, get_noise

def load_model(model, models_dir, file_name, device):
    path = os.path.join(models_dir, file_name)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict).to(device)
    model.eval()
    return model

G = load_model(Generator, 'G.pth')
M = load_model(MappingNetwork, 'M.pth')

def generate(models_path, output_dir, num, seed, resolution=128, W_dim=256, Z_dim=256, device='cuda'):
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    G = Generator(log_resolution, W_dim)
    M = MappingNetwork(Z_dim, W_dim)

    load_model(G, models_path, 'G.pth', device)
    load_model(M, models_path, 'M.pth', device)

    torch.manual_seed(seed)
    log_resolution = int(math.log2(resolution))

    w = get_w(num, W_dim, log_resolution, M, device)
    noise = get_noise(num, log_resolution, device)
    fakes = G(w, noise)

    grid = vutils.make_grid(fakes, nrow=int(math.sqrt(num)), padding=2, normalize=False)
    output_path = os.path.join(output_dir, f"grid_seed_{seed}.png")
    vutils.save_image(grid, output_path)
    print(f"Saved sample grid to {output_path}")