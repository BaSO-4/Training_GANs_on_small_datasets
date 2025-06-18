import torch
import torch.nn.functional as F
import math, random

def augment(image_uint8: torch.Tensor, p: float) -> torch.Tensor:
    B, C, H, W = image_uint8.shape
    device = image_uint8.device

    image = image_uint8.float() / 127.5 - 1.0

    def get_affine_matrix():
        G = torch.eye(3, device=device)

        def apply_prob(fn):
            return fn() if torch.rand(1) < p else torch.eye(3, device=device)

        # Flip X
        G = apply_prob(lambda: torch.tensor([[ -1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32)) @ G

        # Rotate 90° * k
        G = apply_prob(lambda: {
            0: torch.eye(3, device=device, dtype=torch.float32),
            1: torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=device, dtype=torch.float32),
            2: torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device, dtype=torch.float32),
            3: torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=device, dtype=torch.float32),
        }[random.randint(0, 3)]) @ G

        # Integer translation
        G = apply_prob(lambda: torch.tensor([
            [1, 0, round(random.uniform(-0.125, 0.125) * W)],
            [0, 1, round(random.uniform(-0.125, 0.125) * H)],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)) @ G

        # Isotropic scaling
        G = apply_prob(lambda: torch.tensor([
            [s := torch.distributions.LogNormal(0, 0.2 * math.log(2)).sample().item(), 0, 0],
            [0, s, 0],
            [0, 0, 1]
        ], device=device)) @ G

        # Pre-rotation
        prot = 1 - math.sqrt(1 - p)
        if torch.rand(1).item() < prot:
            theta = random.uniform(-math.pi, math.pi)
            R = torch.tensor([
                [math.cos(-theta), -math.sin(-theta), 0],
                [math.sin(-theta), math.cos(-theta), 0],
                [0, 0, 1]
            ], device=device)
            G = R @ G

        # Anisotropic scaling
        G = apply_prob(lambda: torch.tensor([
            [s := torch.distributions.LogNormal(0, 0.2 * math.log(2)).sample().item(), 0, 0],
            [0, 1/s, 0],
            [0, 0, 1]
        ], device=device)) @ G

        # Post-rotation
        if torch.rand(1).item() < prot:
            theta = random.uniform(-math.pi, math.pi)
            R = torch.tensor([
                [math.cos(-theta), -math.sin(-theta), 0],
                [math.sin(-theta), math.cos(-theta), 0],
                [0, 0, 1]
            ], device=device)
            G = R @ G

        # Fractional translation
        G = apply_prob(lambda: torch.tensor([
            [1, 0, random.gauss(0, 0.125) * W],
            [0, 1, random.gauss(0, 0.125) * H],
            [0, 0, 1]
        ], device=device)) @ G

        return G[:2, :]

    # Apply affine transform
    affine_matrices = torch.stack([get_affine_matrix() for _ in range(B)])
    grid = F.affine_grid(affine_matrices, image.size(), align_corners=False)
    image = F.grid_sample(image, grid, padding_mode='reflection', align_corners=False)

    # Color transformations
    def apply_color(image):
        def apply_prob(fn):
            return fn(image) if torch.rand(1) < p else image

        # Brightness
        image = apply_prob(lambda img: img + torch.randn(1, device=device).item() * 0.2)

        # Contrast
        image = apply_prob(lambda img: img * torch.distributions.LogNormal(0, 0.5 * math.log(2)).sample().item())

        # Luma flip
        v = torch.tensor([1.0, 1.0, 1.0], device=device) / math.sqrt(3)
        if torch.rand(1) < p:
            H = torch.eye(3, device=device) - 2 * v[:, None] @ v[None, :]
            image = image.permute(0, 2, 3, 1)
            image = image @ H.T
            image = image.permute(0, 3, 1, 2)

        # Hue rotation
        if torch.rand(1) < p:
            theta = random.uniform(-math.pi, math.pi)
            K = torch.tensor([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ], device=device)
            R = torch.eye(3, device=device) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K
            image = image.permute(0, 2, 3, 1)
            image = image @ R.T
            image = image.permute(0, 3, 1, 2)

        # Saturation
        if torch.rand(1) < p:
            image = (image.permute(0, 2, 3, 1) @ (
                    v[:, None] @ v[None, :] +
                    (torch.eye(3, device=device) - v[:, None] @ v[None, :]) *
                    torch.distributions.LogNormal(0, math.log(2)).sample().item()
                ).T).permute(0, 3, 1, 2)
        return image

    image = apply_color(image)

  # image space filtering
    bands = [0, math.pi/8, math.pi/4, math.pi/2]
    g = torch.ones(4, device=device)
    lambdas = torch.tensor([10, 1, 1, 1], device=device) / 13

    for i, b in enumerate(bands):
        if torch.rand(1).item() < p:
            t = torch.ones(4, device=device)
            t[i] = torch.distributions.LogNormal(0, math.log(2)).sample().to(device)
            t = t / torch.sqrt(torch.sum((lambdas * t ** 2)))
            g *= t

    def gaussian_bandpass_filter(img, sigma):
        k = int(4 * sigma + 1)
        padding = k // 2
        gauss = torch.arange(-padding, padding + 1, device=img.device, dtype=torch.float32)
        gauss = torch.exp(-0.5 * (gauss / sigma) ** 2)
        gauss = gauss / gauss.sum()
        gauss = gauss.view(1, 1, -1).repeat(img.size(1), 1, 1)
        img = F.pad(img, (padding, padding, padding, padding), mode='reflect')
        img = F.conv2d(img, gauss.unsqueeze(2), groups=img.size(1))
        img = F.conv2d(img, gauss.unsqueeze(3), groups=img.size(1))
        return img

    band_sigmas = [2.0, 1.0, 0.5, 0.25]
    filtered = sum(gaussian_bandpass_filter(image, sigma) * gain
                   for sigma, gain in zip(band_sigmas, g))
    image = filtered

    # additive rgb noise
    if torch.rand(1).item() < p:
        sigma = torch.distributions.HalfNormal(0.1).sample().item()
        noise = torch.randn_like(image) * sigma
        image = image + noise

    # cutout
    if torch.rand(1).item() < p:
        B, C, H, W = image.shape
        cx = torch.randint(W, (1,)).item()
        cy = torch.randint(H, (1,)).item()
        cut_w = W // 2
        cut_h = H // 2
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)
        mask = torch.ones_like(image)
        mask[:, :, y1:y2, x1:x2] = 0
        image = image * mask

    return torch.clamp(image, -1.0, 1.0)