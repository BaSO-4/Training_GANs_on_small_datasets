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
    return (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()

def train(data_dir, outdir, batch_size=32, resolution=256, latent_dim=512, r1_gamma=10.0, ema_beta=0.999, lr=2.5e-4, total_kimg=2500, ada_target=0.6, ada_interval=4, ada_kimg=500, log_interval=100, device="cuda", pretrained_g=None, pretrained_d=None, freeze_upto=None):
    os.makedirs(outdir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)
    G = get_generator(l_dim=latent_dim, s_dim=latent_dim).to('cuda')
    G_ema = get_generator(l_dim=latent_dim, s_dim=latent_dim).to('cuda')
    G_ema.load_state_dict(G.state_dict())
    for p in G_ema.parameters():
        p.requires_grad_(False)
    D = get_discriminator().to('cuda')
    if pretrained_g:
        ckpt = torch.load(pretrained_g, map_location='cuda')
        key = 'G_ema' if 'G_ema' in ckpt else 'G'
        G.load_state_dict(ckpt[key], strict=True)
        G_ema.load_state_dict(ckpt[key], strict=True)
    if pretrained_d:
        ckpt_d = torch.load(pretrained_d, map_location='cuda')
        D.load_state_dict(ckpt_d['D'], strict=True)
    if freeze_upto is not None:
        for idx in range(freeze_upto):
            for param in G.blocks[idx].parameters():
                param.requires_grad_(False)
        G.constant_input.requires_grad_(False)
        for param in G.initial_conv.parameters():
            param.requires_grad_(False)
    opt_G = optim.Adam([p for p in G.parameters() if p.requires_grad], lr=lr, betas=(0.0,0.99))
    opt_D = optim.Adam([p for p in D.parameters() if p.requires_grad], lr=lr, betas=(0.0,0.99))
    p_aug = 0.0
    tau = ada_target
    ada_alpha = 1.0 / (ada_kimg * 1000.0 / batch_size)
    cur_nimg = 0
    step = 0
    start = time.time()
    scaler = torch.cuda.amp.GradScaler()
    i = 1
    while cur_nimg < total_kimg * 1000:
        print("current: ", cur_nimg)
        for real_uint8, _ in loader:
            print(i)
            i += 1
            B = real_uint8.size(0)
            cur_nimg += B
            step += 1

            real_aug_cpu = augmentation(real_uint8, p_aug)
            real_aug = real_aug_cpu.to('cuda', dtype=torch.float32)
            real_aug = real_aug / 127.5 - 1.0
            z = torch.randn(B, latent_dim, device='cuda')
            fake = G(z)
            fake_uint8_cpu = ((fake.clamp(-1,1) + 1) * 127.5).cpu().to(torch.uint8)
            fake_aug_cpu = augmentation(fake_uint8_cpu, p_aug)
            fake_aug = fake_aug_cpu.to('cuda', dtype=torch.float32)
            fake_aug = fake_aug / 127.5 - 1.0
            print("+++++++++")
            D.train()
            logits_real = D(real_aug)
            print("dic 1")
            logits_fake = D(fake_aug.detach())
            print("disc 2")
            loss_D = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real)) + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
            real_float_cpu = (real_uint8.to(torch.float32)/127.5 - 1.0)
            loss_D = loss_D + (r1_gamma/2.0) * compute_r1_penalty(D, real_float_cpu)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            with torch.no_grad():
                real_pred = (logits_real > 0).float().mean().item()
            p_aug = max(0.0, min(1.0, p_aug + ada_alpha * (real_pred - tau)))

            D.to('cuda')
            torch.cuda.empty_cache()
            G.train()
            z2 = torch.randn(B, latent_dim, device='cuda')
            fake2 = G(z2)
            fake2_uint8_cpu = ((fake2.clamp(-1,1) + 1) * 127.5).cpu().to(torch.uint8)
            fake2_aug_cpu = augmentation(fake2_uint8_cpu, p_aug)
            fake2_aug = fake2_aug_cpu.to('cuda')
            logits_fake2 = D(fake2_aug)
            loss_G = F.binary_cross_entropy_with_logits(logits_fake2, torch.ones_like(logits_fake2))
            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()
            update_ema(G, G_ema, ema_beta)
            D.to('cpu')
            torch.cuda.empty_cache()

            if step % log_interval == 0:
                elapsed = time.time() - start
                print(f"Step {step} | D_loss {loss_D.item():.4f} | G_loss {loss_G.item():.4f} | p {p_aug:.3f} | {cur_nimg//1000}kimg | {elapsed:.1f}s")
            if cur_nimg >= total_kimg * 1000:
                break

        ckpt = {'G': G.state_dict(), 'D': D.state_dict(), 'G_ema': G_ema.state_dict(), 'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict(), 'p': p_aug, 'step': step, 'cur_nimg': cur_nimg}
        torch.save(ckpt, os.path.join(outdir, f'checkpoint_{cur_nimg//1000}kimg.pth'))

    torch.save({'G_ema': G_ema.state_dict()}, os.path.join(outdir, 'G_ema_final.pth'))