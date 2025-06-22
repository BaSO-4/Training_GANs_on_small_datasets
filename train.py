import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from networks import MappingNetwork, Generator, Discriminator, PathLengthPenalty, get_noise, get_w
from augment import augment
from math import log2
import os


def gradient_penalty(critic, real, fake,device="cpu"):
    B, C, H, W = real.shape
    beta = torch.rand((B, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def train(models_dir, data_dir, batch_size, lr=1e-3, epochs=300, resolution=128, W_dim=256, Z_dim=256, lmbd=10.0, ada_num_img=500000, device="cuda"):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x - 1)
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    log_resolution = int(log2(resolution))

    G = Generator(log_resolution, W_dim).to(device)
    G.train()
    D = Discriminator(log_resolution).to(device)
    D.train()
    M = MappingNetwork(Z_dim, W_dim).to(device)
    M.train()
    opt_G = optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=lr, betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.99))
    opt_M = optim.Adam(M.parameters(), lr=lr, betas=(0.0, 0.99))
    path_length_penalty = PathLengthPenalty(0.99).to(device)

    p_aug = 0.0
    N = int(64 / batch_size)
    delta_p = N * batch_size / ada_num_img
    last_N = torch.empty(0, device=device)
    h = lambda last_N: torch.mean(last_N)
    h_goal = 0.6

    for epoch in range(epochs):
      print(f"epoch {epoch}/{epochs}")
      for i, (real, _) in enumerate(loader):
          B = real.shape[0]
          real = real.to(device, non_blocking=True)
          real_aug = augment(real, p_aug)

          w = get_w(B, W_dim, log_resolution, M, device)
          noise = get_noise(B, log_resolution, device)
          fake = G(w, noise)
          fake_aug = augment(fake, p_aug)
          D_fake_pred = D(fake_aug.detach())
          D_real_pred = D(real_aug.detach())
          grad = gradient_penalty(D, real_aug, fake_aug, device)
          loss_D = (
              -(torch.mean(D_real_pred) - torch.mean(D_fake_pred))
              + lmbd * grad
              + (0.001 * torch.mean(D_real_pred ** 2))
          )
          D.zero_grad()
          loss_D.backward()
          opt_D.step()

          D_fake_pred = D(fake_aug)
          loss_G = -torch.mean(D_fake_pred)

          if i % 16 == 0:
              plp = path_length_penalty(w, fake)
              if not torch.isnan(plp):
                  loss_G = loss_G + plp

          M.zero_grad()
          G.zero_grad()
          loss_G.backward()
          opt_G.step()
          opt_M.step()

          # p_aug update
          with torch.no_grad():
              real_pred = (D_real_pred > 0).float().mean().item()
              if len(last_N) == N:
                  last_N = torch.cat([last_N[1:], torch.tensor([real_pred], device=device)])
              else:
                  last_N = torch.cat([last_N, torch.tensor([real_pred], device=device)])
              h_N = h(last_N)
              p_aug = max(0, p_aug + (int(h_N > h_goal)*2-1) * delta_p)

          if i % 2000 == 0:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(((fake[0].clamp(-1, 1) + 1)/2).permute(1, 2, 0).cpu().detach().numpy())
            axes[0].axis('off')
            axes[1].imshow((((real_aug[0].clamp(-1, 1) + 1)/2).permute(1, 2, 0).cpu().detach().numpy()))
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()

    os.makedirs(models_dir, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(models_dir, 'G.pth'))
    torch.save(M.state_dict(), os.path.join(models_dir, 'M.pth'))
    torch.save(D.state_dict(), os.path.join(models_dir, 'D.pth'))