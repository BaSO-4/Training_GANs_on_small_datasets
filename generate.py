import os, math
import torch
from torchvision import utils as vutils

from networks import get_generator

def generate(checkpoint,num=25,latent_dim=512,seed=42,output_dir=".\\generations",device="cuda"):
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    G = get_generator(latent_dim=latent_dim, style_dim=latent_dim).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    if 'G_ema' in ckpt:
        G.load_state_dict(ckpt['G_ema'])
    elif 'G' in ckpt:
        G.load_state_dict(ckpt['G'])
    else:
        raise KeyError("Checkpoint must contain 'G_ema' or 'G' key.")
    G.eval()

    torch.manual_seed(seed)
    z = torch.randn(num, latent_dim, device=device)
    with torch.no_grad():
        samples = G(z).clamp(-1, 1)
    samples = (samples + 1) / 2

    # This is just for combining generations into one image and saving
    grid = vutils.make_grid(samples, nrow=int(math.sqrt(num)), padding=2, normalize=False)
    output_path = os.path.join(output_dir, "grid.png")
    vutils.save_image(grid, output_path)
    print(f"Saved sample grid to {output_path}")