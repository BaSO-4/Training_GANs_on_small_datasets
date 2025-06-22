import torch
import torch.nn.functional as F
from math import pi, sin, cos, log, sqrt


def transformations(Y, p):
    device = Y.device
    B, C, H, W = Y.shape

    G = torch.eye(3, device=device)

    # pixel blitting
    if torch.rand(1).item() < p:
        i = torch.randint(0, 2, ()).item()
        G = torch.tensor([
            [1-2*i, 0, 0],
            [ 0, 1, 0],
            [ 0, 0, 1]
        ], device=device, dtype=torch.float32) @ G

    # rotations
    if torch.rand(1).item() < p:
        i = torch.randint(0, 4, ()).item()
        theta = -pi / 2 * i
        G = torch.tensor([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]
        ], device=device, dtype=torch.float32) @ G

    # translation
    if torch.rand(1).item() < p:
        tx = torch.round(torch.rand(1, device=device) * 0.25 - 0.125).item() * W
        ty = torch.round(torch.rand(1, device=device) * 0.25 - 0.125).item() * H
        G = torch.tensor([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], device=device, dtype=torch.float32) @ G

    # isotropic scaling
    if torch.rand(1).item() < p:
        s = torch.distributions.LogNormal(0.0, (0.2 * log(2))).sample().item()
        G = torch.tensor([
            [s, 0, 0],
            [0, s, 0],
            [0, 0, 1]
        ], device=device, dtype=torch.float32) @ G

    p_rot = 1 - torch.sqrt(torch.tensor(1 - p, device=device))

    # pre rotation
    if torch.rand(1).item() < p_rot:
        theta = (torch.rand(1, device=device) * 2 * pi) - pi
        G = torch.tensor([
            [cos(-theta), -sin(-theta), 0],
            [sin(-theta), cos(-theta), 0],
            [0, 0, 1]
        ], device=device, dtype=torch.float32) @ G

    # anisotropic scaling
    if torch.rand(1).item() < p:
        s = torch.distributions.LogNormal(0.0, (0.2 * log(2))).sample().item()
        G = torch.tensor([
            [s, 0, 0],
            [0, 1/s, 0],
            [0, 0, 1]
        ], device=device, dtype=torch.float32) @ G

    # post rotation
    if torch.rand(1).item() < p_rot:
        theta = (torch.rand(1, device=device) * 2 * pi) - pi
        G = torch.tensor([
            [cos(-theta), -sin(-theta), 0],
            [sin(-theta), cos(-theta), 0],
            [0, 0, 1]
        ], device=device, dtype=torch.float32) @ G

    # fractional translation
    if torch.rand(1).item() < p:
        tx = torch.round(torch.rand(1, device=device) * 0.125).item() * W
        ty = torch.round(torch.rand(1, device=device) * 0.125).item() * H
        G = torch.tensor([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], device=device, dtype=torch.float32) @ G

    # pad image and adjust origin
    filter_size = 12
    pad_x = pad_y = filter_size // 2
    m_lo = torch.tensor([pad_x, pad_y], device=device, dtype=torch.uint8)
    m_hi = torch.tensor([pad_x, pad_y], device=device, dtype=torch.uint8)
    Y = F.pad(Y, [pad_x, pad_x, pad_y, pad_y], mode='reflect')
    cx = W / 2 - 0.5 + pad_x
    cy = H / 2 - 0.5 + pad_y
    T = torch.tensor([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)
    G = T @ G @ torch.inverse(T)

    # execute geometric transformations
    Y = F.interpolate(Y, scale_factor=2, mode='bilinear', align_corners=False)
    S = torch.tensor([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
    ], device=G.device, dtype=torch.float32)
    G = S @ G @ torch.inverse(S)
    affine = G[:2, :].unsqueeze(0).repeat(B, 1, 1)
    grid = F.affine_grid(affine, size=Y.size(), align_corners=False)
    Y = F.grid_sample(Y, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    Y = F.avg_pool2d(Y, kernel_size=2, stride=2)
    pad = filter_size // 2
    Y = Y[:, :, pad:-pad, pad:-pad]


    C = torch.eye(4, device=device)

    # brightness
    if torch.rand(1).item() < p:
        b = torch.randn(1, device=device) * 0.2
        C = torch.tensor([
            [1, 0, 0, b.item()],
            [0, 1, 0, b.item()],
            [0, 0, 1, b.item()],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32) @ C

    # contrast
    if torch.rand(1).item() < p:
        c = torch.distributions.LogNormal(0.0, (0.5 * log(2))).sample().item()
        C = torch.tensor([
            [c, 0, 0, 0],
            [0, c, 0, 0],
            [0, 0, c, 0],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32) @ C

    v = torch.tensor([1, 1, 1, 0], device=device) / sqrt(3)

    # luma flip
    if torch.rand(1).item() < p:
        i = torch.randint(0, 2, ()).item()
        C = (torch.eye(4, device=device) - 2 * torch.outer(v, v) * i) @ C

    # hue rotation
    if torch.rand(1).item() < p:
        theta = (torch.rand(1, device=device) * 2 * pi) - pi
        C = torch.tensor([
            [0, -v[2], v[1], 0],
            [v[2], 0, -v[0], 0],
            [-v[1], v[0], 0, 0],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32) @ C

    # saturation
    if torch.rand(1).item() < p:
        s = torch.distributions.LogNormal(0.0, (log(2))).sample().item()
        C = (torch.outer(v, v) + (torch.eye(4, device=device) - torch.outer(v, v)) * s) @ C

    # execute color transformations
    Y = Y.permute(0, 2, 3, 1).reshape(-1, 3)
    ones = torch.ones((Y.shape[0], 1), device=Y.device)
    Y = torch.cat([Y, ones], dim=1)
    Y = Y @ C.T
    Y = Y[:, :3]
    Y = Y.view(B, H, W, 3).permute(0, 3, 1, 2)
    return Y

def corruptions(Y, p):
    B, C, H, W = Y.shape
    device = Y.device
    b = torch.tensor([[0, pi/8], [pi/8, pi/4], [pi/4, pi/2], [pi/2, pi]], device=device)
    g = torch.ones(4, device=device)
    lmbd = torch.tensor([10, 1, 1, 1], device=device) / 13

    for i in range(4):
        if torch.rand(1).item() < p:
            t = torch.ones(4, device=device)
            t[i] = torch.distributions.LogNormal(0.0, (log(2))).sample().item()
            t = t / sqrt(torch.dot(lmbd, torch.pow(t, 2)))
            g = g * t

    def gaussian_kernel_2d(kernel_size=5, sigma=1.0, device='cpu'):
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    # execute image-space filtering
    H_z = gaussian_kernel_2d(device=device)
    g = torch.tensor([1.0, 0.5, 0.25, 0.1], device=device)
    H_z_prime = torch.zeros_like(H_z, device=device)
    for i in range(4):
        y = torch.arange(H_z.shape[0], device=device, dtype=torch.float32)
        x = torch.arange(H_z.shape[1], device=device, dtype=torch.float32)
        Ys, Xs = torch.meshgrid(y, x, indexing='ij')
        phase = b[i][0] * Xs + b[i][1] * Ys
        modulation = torch.cos(phase)
        bandpassed = H_z * modulation
        H_z_prime += bandpassed * g[i]
    m_lo, m_hi = H_z.shape[0] // 2, H_z.shape[1] // 2
    Y = F.pad(Y, (m_lo, m_hi, m_lo, m_hi), mode='reflect')
    H_z_prime = H_z_prime.view(1, 1, H_z_prime.shape[0], H_z_prime.shape[1])
    Y = F.conv2d(Y, weight=H_z_prime.expand(C, 1, -1, -1), groups=C, padding=(m_lo, m_hi))
    Y = Y[:, :, m_lo:-m_hi, m_lo:-m_hi]

    # additive rgb noise
    if torch.rand(1).item() < p:
        sig = torch.abs(torch.randn(1, device=device)) * 0.1
        noise = torch.randn_like(Y) * sig
        Y = Y + noise
        Y = Y.clamp(-1, 1)

    # cutout
    if torch.rand(1).item() < p:
        cx = torch.rand(B, device=device)
        cy = torch.rand(B, device=device)
        r_lo_x = torch.round((cx - 0.25) * W).clamp(0, W - 1).long()
        r_lo_y = torch.round((cy - 0.25) * H).clamp(0, H - 1).long()
        r_hi_x = torch.round((cx + 0.25) * W).clamp(0, W).long()
        r_hi_y = torch.round((cy + 0.25) * H).clamp(0, H).long()
        mask = torch.ones_like(Y, device=device)
        for i in range(B):
            mask[i, :, r_lo_y[i]:r_hi_y[i], r_lo_x[i]:r_hi_x[i]] = 0
        Y = Y * mask
    return Y

def augment(image, p):
    image = transformations(image, p)
    image = corruptions(image, p)
    return image