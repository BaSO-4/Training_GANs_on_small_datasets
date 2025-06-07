import os
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from networks import get_generator, get_discriminator
from augment import augmentation

def update_ema(G, G_ema, beta):
    with torch.no_grad():
        for p, p_ema in zip(G.parameters(), G_ema.parameters()):
            p_ema.copy_(beta * p_ema + (1 - beta) * p)


def compute_r1_penalty(D, real_images):
    real_images = real_images.requires_grad_(True)
    real_scores = D(real_images)
    grad_real = torch.autograd.grad(outputs=real_scores.sum(), inputs=real_images, create_graph=True)[0]
    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
    return grad_penalty


def train(data_dir,outdir,batch_size=32,resolution=256,latent_dim=512,r1_gamma=10.0,ema_beta=0.999,lr=2.5e-4,total_kimg=2500,ada_target=0.6,ada_interval=4,ada_kimg=500,log_interval=100,device="cuda",pretrained_g=None,pretrained_d=None,freeze_upto=None):
    device = torch.device(device)
    os.makedirs(outdir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )

    G = get_generator(l_dim=latent_dim, s_dim=latent_dim).to(device)
    D = get_discriminator().to(device)
    G_ema = get_generator(l_dim=latent_dim, s_dim=latent_dim).to(device)
    G_ema.load_state_dict(G.state_dict())
    for p in G_ema.parameters():
        p.requires_grad_(False)

    if pretrained_g is not None:
        print(f"Loading pretrained Generator from {pretrained_g} …")
        ckpt = torch.load(pretrained_g, map_location=device)
        if 'G_ema' in ckpt:
            G.load_state_dict(ckpt['G_ema'], strict=True)
            G_ema.load_state_dict(ckpt['G_ema'], strict=True)
        elif 'G' in ckpt:
            G.load_state_dict(ckpt['G'], strict=True)
            G_ema.load_state_dict(ckpt['G'], strict=True)
        else:
            raise KeyError("Pretrained Generator checkpoint must contain 'G_ema' or 'G' key.")

    if pretrained_d is not None:
        print(f"Loading pretrained Discriminator from {pretrained_d} …")
        ckpt_d = torch.load(pretrained_d, map_location=device)
        if 'D' in ckpt_d:
            D.load_state_dict(ckpt_d['D'], strict=True)
        else:
            raise KeyError("Pretrained Discriminator checkpoint must contain 'D' key.")

    if freeze_upto is not None:
        for idx in range(freeze_upto):
            for param in G.blocks[idx].parameters():
                param.requires_grad_(False)
        G.constant_input.requires_grad_(False)
        for param in G.initial_conv.parameters():
            param.requires_grad_(False)
        print(f"Frozen first {freeze_upto} generator blocks + constant/input conv")

    trainable_G = [p for p in G.parameters() if p.requires_grad]
    trainable_D = [p for p in D.parameters() if p.requires_grad]

    opt_G = optim.Adam(trainable_G, lr=lr, betas=(0.0, 0.99))
    opt_D = optim.Adam(trainable_D, lr=lr, betas=(0.0, 0.99))

    p = 0.0 # AUgmentation probability
    tau = ada_target
    ada_alpha = 1.0 / (ada_kimg * 1000.0 / batch_size)

    cur_nimg = 0
    step = 0
    start_time = time.time()

    while cur_nimg < total_kimg * 1000:
        for real_uint8, _ in loader:
            real_uint8 = real_uint8.to('cpu')
            B = real_uint8.shape[0]
            cur_nimg += B
            step += 1

            real_aug = augmentation(real_uint8, p).to(device)

            z = torch.randn(B, latent_dim, device=device)
            fake = G(z).to('cpu')
            fake_uint8 = ((fake.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
            fake_aug = augmentation(fake_uint8, p).to(device)

            logits_real = D(real_aug)
            logits_fake = D(fake_aug.detach())

            loss_D_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
            loss_D_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
            loss_D = loss_D_real + loss_D_fake

            real_float = real_uint8.to(torch.float32) / 127.5 - 1.0
            r1_penalty = compute_r1_penalty(D, real_float)
            loss_D = loss_D + (r1_gamma / 2.0) * r1_penalty

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            with torch.no_grad():
                real_pred = (logits_real > 0).float().mean().item()
            p = p + ada_alpha * (real_pred - tau)
            p = max(0.0, min(1.0, p))

            z2 = torch.randn(B, latent_dim, device=device)
            fake2 = G(z2)
            fake2_uint8 = ((fake2.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
            fake2_aug = augmentation(fake2_uint8, p).to(device)

            logits_fake2 = D(fake2_aug)
            loss_G = F.binary_cross_entropy_with_logits(logits_fake2, torch.ones_like(logits_fake2))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            update_ema(G, G_ema, ema_beta)

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Step {step:<6} | "
                      f"D loss: {loss_D.item():.4f} | G loss: {loss_G.item():.4f} | "
                      f"r_t: {real_pred:.3f} | p: {p:.3f} | "
                      f"{cur_nimg//1000} kimg | {elapsed:.1f}s")

            if cur_nimg >= total_kimg * 1000:
                break

        ckpt = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'G_ema': G_ema.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'p': p,
            'step': step,
            'cur_nimg': cur_nimg
        }
        torch.save(ckpt, os.path.join(outdir, f'checkpoint_{cur_nimg//1000}kimg.pth'))

    torch.save({'G_ema': G_ema.state_dict()}, os.path.join(outdir, 'G_ema_final.pth'))