import os
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from networks import get_generator, get_discriminator
from augment import augment

def update_ema(G, G_ema, beta):
    with torch.no_grad():
        for p, p_ema in zip(G.parameters(), G_ema.parameters()):
            p_ema.copy_(beta * p_ema + (1 - beta) * p)

def compute_r1_penalty(D, real_images):
    real_images = real_images.requires_grad_(True)
    real_scores = D(real_images)
    grad_real = torch.autograd.grad(outputs=real_scores.sum(), inputs=real_images, create_graph=True)[0]
    return (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()

def train(data_dir,outdir,batch_size=32,resolution=256,latent_dim=512,r1_gamma=10.0,ema_beta=0.999,lr=2.5e-4,total_kimg=2500,ada_target=0.6,ada_interval=4,ada_kimg=500,log_interval=100,device="cuda",pretrained_g=None,pretrained_d=None,freeze_upto=None):
    os.makedirs(outdir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    G = get_generator(latent_dim, latent_dim).to(device)
    G_ema = get_generator(latent_dim, latent_dim).to(device)
    G_ema.load_state_dict(G.state_dict())
    for p in G_ema.parameters():
        p.requires_grad_(False)
    D = get_discriminator().to(device)
    if pretrained_g:
        ckpt = torch.load(pretrained_g, map_location=device)
        key = 'G_ema' if 'G_ema' in ckpt else 'G'
        G.load_state_dict(ckpt[key])
        G_ema.load_state_dict(ckpt[key])
    if pretrained_d:
        ckpt_d = torch.load(pretrained_d, map_location=device)
        D.load_state_dict(ckpt_d['D'])
    if freeze_upto is not None:
        for idx in range(freeze_upto):
            for param in G.blocks[idx].parameters():
                param.requires_grad_(False)
        G.constant_input.requires_grad_(False)
        for param in G.initial_conv.parameters():
            param.requires_grad_(False)
    opt_G = optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=lr, betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.99))
    p_aug = 0.0
    ada_alpha = 1.0 / (ada_kimg * 1000.0 / batch_size)
    cur_nimg, step = 0, 0
    start_time = time.time()
    while cur_nimg < total_kimg * 1000:
        for real_uint8, _ in loader:
            print(cur_nimg)
            B = real_uint8.size(0)
            # X = real_uint8[0].to(torch.float32) / 127.5 - 1.0
            # samples = X.clamp(-1, 1)
            # samples = (samples + 1) / 2
            # plt.figure(figsize=(8, 8))
            # plt.axis('off')
            # plt.imshow(samples.permute(1, 2, 0).cpu().numpy())
            # plt.show()
            cur_nimg += B
            step += 1
            real_uint8 = real_uint8.to(device, non_blocking=True)
            # print("aug:")
            real_aug = augment(real_uint8, p_aug)
            # img = real_aug[0].clamp(-1, 1)         # (Optional) Ensure values are in [-1, 1]
            # img = (img + 1) / 2            # Map to [0, 1]
            # img = img.permute(1, 2, 0)     # [C, H, W] → [H, W, C] for matplotlib
            # img = img.cpu()
            # plt.imshow(img.numpy())
            # plt.axis("off")
            # plt.show()
            # break

            z = torch.randn(B, latent_dim, device=device)
            fake = G(z)
            fake_uint8 = ((fake.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
            fake_aug = augment(fake_uint8, p_aug)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(((fake_aug[0].clamp(-1, 1) + 1)/2).permute(1, 2, 0).cpu().detach().numpy())
            axes[0].axis('off')
            axes[1].imshow((((real_aug[0].clamp(-1, 1) + 1)/2).permute(1, 2, 0).cpu().detach().numpy()))
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
            logits_real = D(real_aug)
            logits_fake = D(fake_aug.detach())
            print("real_logits", logits_real.mean().item())
            print("fake_logits", logits_fake.mean().item())
            loss_D = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real)) \
                     + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
            real_float = (real_uint8.to(torch.float32) / 127.5 - 1.0)
            loss_D += (r1_gamma / 2.0) * compute_r1_penalty(D, real_float)
            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()
            with torch.no_grad():
                real_pred = (logits_real > 0).float().mean().item()
                p_aug = max(0.0, min(1.0, p_aug + ada_alpha * (real_pred - ada_target)))
            z2 = torch.randn(B, latent_dim, device=device)
            fake2 = G(z2)
            # img = fake2[0].clamp(-1, 1)         # (Optional) Ensure values are in [-1, 1]
            # img = (img + 1) / 2            # Map to [0, 1]
            # img = img.permute(1, 2, 0)     # [C, H, W] → [H, W, C] for matplotlib
            # img = img.cpu()
            # plt.imshow(img.detach().numpy())
            # plt.axis("off")
            # plt.show()
            # fake2_norm = (fake2.clamp(-1, 1) + 1) / 2
            fake2_uint8 = ((fake2.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
            fake2_aug = augment(fake2_uint8, p_aug).to(device, non_blocking=True)
            logits_fake2 = D(fake2_aug)
            print("G fake_logits", logits_fake2.mean().item())
            loss_G = F.binary_cross_entropy_with_logits(logits_fake2, torch.ones_like(logits_fake2))
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            # for name, param in G.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradients for {name}: {param.grad.norm().item():.6f}")
            #     else:
            #         print(f"Gradients for {name}: None")
            # print(fake2.requires_grad)
            opt_G.step()
            update_ema(G, G_ema, ema_beta)
            if step % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Step {step} | D_loss {loss_D.item():.4f} | G_loss {loss_G.item():.4f} | p {p_aug:.3f} | {cur_nimg//1000}kimg | {elapsed:.1f}s")
            if cur_nimg >= total_kimg * 1000:
                break
        torch.save({
            'G': G.state_dict(),
            'D': D.state_dict(),
            'G_ema': G_ema.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'p': p_aug,
            'step': step,
            'cur_nimg': cur_nimg,
        }, os.path.join(outdir, f'checkpoint_{cur_nimg // 1000}kimg.pth'))
    torch.save({'G_ema': G_ema.state_dict()}, os.path.join(outdir, 'G_ema_final.pth'))