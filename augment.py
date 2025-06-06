import torch
import torch.nn.functional as F
import math

def init_G(device, batch_size):
    G = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    return G


def pixel_blitting(G, p, w, h):
    B = G.shape[0]
    device = G.device

    # x-flip with probability p
    flip_mask = (torch.rand(B, device=device) < p).float()
    i_flip = (torch.randint(0, 2, (B,), device=device).float() * flip_mask)
    sx = 1.0 - 2.0 * i_flip
    sy = torch.ones(B, device=device)
    S_flip = torch.zeros((B, 3, 3), device=device)
    S_flip[:, 0, 0] = sx
    S_flip[:, 1, 1] = sy
    S_flip[:, 2, 2] = 1.0
    G = torch.bmm(S_flip, G)

    # 90-degree rotations with probability p
    rot_mask = (torch.rand(B, device=device) < p).float()
    i_rot = (torch.randint(0, 4, (B,), device=device).float() * rot_mask)
    cos_i = torch.cos(-0.5 * math.pi * i_rot)
    sin_i = torch.sin(-0.5 * math.pi * i_rot)
    R90 = torch.zeros((B, 3, 3), device=device)
    R90[:, 0, 0] = cos_i
    R90[:, 0, 1] = -sin_i
    R90[:, 1, 0] = sin_i
    R90[:, 1, 1] = cos_i
    R90[:, 2, 2] = 1.0
    G = torch.bmm(R90, G)

    # Integer translation with probability p
    trans_mask = (torch.rand(B, device=device) < p).float()
    tx = torch.empty(B, device=device).uniform_(-0.125, 0.125) * trans_mask
    ty = torch.empty(B, device=device).uniform_(-0.125, 0.125) * trans_mask
    tx_pix = torch.round(tx * w)
    ty_pix = torch.round(ty * h)
    T_int = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_int[:, 0, 2] = tx_pix
    T_int[:, 1, 2] = ty_pix
    G = torch.bmm(T_int, G)
    return G


def general_geometric(G, p, w, h):
    B = G.shape[0]
    device = G.device

    # Isotropic scaling with probability p
    scale_mask = (torch.rand(B, device=device) < p).float()
    sigma = 0.2 * math.log(2.0)
    ln_s = torch.randn(B, device=device) * sigma
    s_iso = torch.exp(ln_s) * scale_mask + (1.0 - scale_mask)
    S_iso = torch.zeros((B, 3, 3), device=device)
    S_iso[:, 0, 0] = s_iso
    S_iso[:, 1, 1] = s_iso
    S_iso[:, 2, 2] = 1.0
    G = torch.bmm(S_iso, G)

    # Pre-rotation with probability prot = 1 - sqrt(1 - p)
    prot = 1.0 - math.sqrt(1.0 - p)
    pre_rot_mask = (torch.rand(B, device=device) < prot).float()
    theta_pre = (torch.rand(B, device=device) * 2 * math.pi - math.pi) * pre_rot_mask
    cos_pre = torch.cos(-theta_pre)
    sin_pre = torch.sin(-theta_pre)
    R_pre = torch.zeros((B, 3, 3), device=device)
    R_pre[:, 0, 0] = cos_pre
    R_pre[:, 0, 1] = -sin_pre
    R_pre[:, 1, 0] = sin_pre
    R_pre[:, 1, 1] = cos_pre
    R_pre[:, 2, 2] = 1.0
    G = torch.bmm(R_pre, G)

    # Anisotropic scaling with probability p
    aniso_mask = (torch.rand(B, device=device) < p).float()
    ln_s_aniso = torch.randn(B, device=device) * sigma
    s_aniso = torch.exp(ln_s_aniso) * aniso_mask + (1.0 - aniso_mask)
    inv_s = 1.0 / s_aniso
    S_aniso = torch.zeros((B, 3, 3), device=device)
    S_aniso[:, 0, 0] = s_aniso
    S_aniso[:, 1, 1] = inv_s
    S_aniso[:, 2, 2] = 1.0
    G = torch.bmm(S_aniso, G)

    # Post-rotation with same probability prot
    post_rot_mask = pre_rot_mask
    theta_post = (torch.rand(B, device=device) * 2 * math.pi - math.pi) * post_rot_mask
    cos_post = torch.cos(-theta_post)
    sin_post = torch.sin(-theta_post)
    R_post = torch.zeros((B, 3, 3), device=device)
    R_post[:, 0, 0] = cos_post
    R_post[:, 0, 1] = -sin_post
    R_post[:, 1, 0] = sin_post
    R_post[:, 1, 1] = cos_post
    R_post[:, 2, 2] = 1.0
    G = torch.bmm(R_post, G)

    # Fractional translation with probability p
    frac_mask = (torch.rand(B, device=device) < p).float()
    tx_f = torch.randn(B, device=device) * 0.125 * frac_mask
    ty_f = torch.randn(B, device=device) * 0.125 * frac_mask
    tx_pix_f = tx_f * w
    ty_pix_f = ty_f * h
    T_frac = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_frac[:, 0, 2] = tx_pix_f
    T_frac[:, 1, 2] = ty_pix_f
    G = torch.bmm(T_frac, G)
    return G


def pad_and_warp_image(Y, G, w, h):
    B, C, H, W = Y.shape
    device = Y.device

    # Approximate SYM6 with 3-tap lowpass for padding calculation
    pad_lo = torch.tensor([1, 1], device=device)
    pad_hi = torch.tensor([1, 1], device=device)

    pad_vals = (pad_lo[0].item(), pad_hi[0].item(), pad_lo[1].item(), pad_hi[1].item())
    Y_pad = F.pad(Y, pad_vals, mode='reflect')

    # Center shift G via T and T^{-1}
    cx = (w - 1) / 2.0 + pad_lo[0].item()
    cy = (h - 1) / 2.0 + pad_lo[1].item()
    T = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_inv = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    T[:, 0, 2] = cx
    T[:, 1, 2] = cy
    T_inv[:, 0, 2] = -cx
    T_inv[:, 1, 2] = -cy
    G = torch.bmm(T, torch.bmm(G, T_inv))

    Y0 = F.interpolate(Y_pad, size=(Y_pad.shape[2] * 2, Y_pad.shape[3] * 2), mode='bilinear', align_corners=False)
    S_up = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    S_inv = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    S_up[:, 0, 0] = 2.0
    S_up[:, 1, 1] = 2.0
    S_inv[:, 0, 0] = 0.5
    S_inv[:, 1, 1] = 0.5
    G = torch.bmm(S_up, torch.bmm(G, S_inv))

    # Warp via grid_sample using G^{-1}
    G_inv = torch.inverse(G)
    nH, nW = H, W
    xs = torch.linspace(0, nW - 1, nW, device=device)
    ys = torch.linspace(0, nH - 1, nH, device=device)
    grid_xy = torch.stack(torch.meshgrid(ys, xs), dim=-1)
    ones = torch.ones((nH, nW, 1), device=device)
    grid_homo = torch.cat([grid_xy[..., [1]], grid_xy[..., [0]], ones], dim=-1)
    grid_flat = grid_homo.view(-1, 3).t().unsqueeze(0).repeat(B, 1, 1)
    src_coords = torch.bmm(G_inv, grid_flat)
    xi = src_coords[:, 0, :] / (src_coords[:, 2, :] + 1e-8)
    yi = src_coords[:, 1, :] / (src_coords[:, 2, :] + 1e-8)
    xi_norm = (xi / (Y0.shape[3] - 1) - 0.5) * 2.0
    yi_norm = (yi / (Y0.shape[2] - 1) - 0.5) * 2.0
    grid_norm = torch.stack([xi_norm, yi_norm], dim=-1).view(B, nH, nW, 2)
    Y_warped = F.grid_sample(Y0, grid_norm, mode='bilinear', padding_mode='reflection', align_corners=True)

    Y_down = F.avg_pool2d(Y_warped, kernel_size=2, stride=2)
    Y_crop = Y_down[:, :, pad_lo[1]:pad_lo[1] + h, pad_lo[0]:pad_lo[0] + w]
    return Y_crop


def color_transform(Y, p):
    B, C, H, W = Y.shape
    device = Y.device

    Y01 = (Y + 1.0) / 2
    Cmat = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)

    # Brightness with probability p
    bright_mask = (torch.rand(B, device=device) < p).float().unsqueeze(1)
    b = torch.randn(B, device=device) * 0.2 * bright_mask.squeeze(1)
    T_b = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_b[:, 0, 3] = b
    T_b[:, 1, 3] = b
    T_b[:, 2, 3] = b
    Cmat = torch.bmm(T_b, Cmat)

    # Contrast with probability p
    contrast_mask = (torch.rand(B, device=device) < p).float().unsqueeze(1)
    sigma_c = 0.5 * math.log(2.0)
    ln_c = torch.randn(B, device=device) * sigma_c
    c_scale = torch.exp(ln_c) * contrast_mask.squeeze(1)
    c_scale = 1.0 - contrast_mask.squeeze(1) + c_scale
    S_c = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    S_c[:, 0, 0] = c_scale
    S_c[:, 1, 1] = c_scale
    S_c[:, 2, 2] = c_scale
    Cmat = torch.bmm(S_c, Cmat)

    # Luma flip with probability p
    luma_mask = (torch.rand(B, device=device) < p).float()
    v = torch.tensor([1.0, 1.0, 1.0, 0.0], device=device) / math.sqrt(3.0)
    for idx in range(B):
        if luma_mask[idx] == 1.0:
            v_mat = v.unsqueeze(1) @ v.unsqueeze(0)
            H_i = torch.eye(4, device=device) - 2.0 * v_mat
            Cmat[idx] = H_i @ Cmat[idx]
        else:
            Cmat[idx] = Cmat[idx]

    # Hue rotation with probability p
    hue_mask = (torch.rand(B, device=device) < p).float()
    theta_hue = (torch.rand(B, device=device) * 2 * math.pi - math.pi) * hue_mask

    def build_hue_rot(theta_batch):
        v3 = torch.tensor([1.0, 1.0, 1.0], device=device) / math.sqrt(3.0)
        vx, vy, vz = v3
        cos_t = torch.cos(theta_batch)
        sin_t = torch.sin(theta_batch)
        one_c = 1.0 - cos_t
        R = torch.zeros((B, 3, 3), device=device)
        for b_idx in range(B):
            c = cos_t[b_idx]
            s = sin_t[b_idx]
            oc = one_c[b_idx]
            R[b_idx, 0, 0] = c + oc * vx * vx
            R[b_idx, 0, 1] = oc * vx * vy - s * vz
            R[b_idx, 0, 2] = oc * vx * vz + s * vy
            R[b_idx, 1, 0] = oc * vy * vx + s * vz
            R[b_idx, 1, 1] = c + oc * vy * vy
            R[b_idx, 1, 2] = oc * vy * vz - s * vx
            R[b_idx, 2, 0] = oc * vz * vx - s * vy
            R[b_idx, 2, 1] = oc * vz * vy + s * vx
            R[b_idx, 2, 2] = c + oc * vz * vz
        return R

    R3d = build_hue_rot(theta_hue)
    R_hue = torch.zeros((B, 4, 4), device=device)
    R_hue[:, :3, :3] = R3d
    R_hue[:, 3, 3] = 1.0
    Cmat = torch.bmm(R_hue, Cmat)

    # Saturation with probability p
    sat_mask = (torch.rand(B, device=device) < p).float().unsqueeze(1)
    sigma_s = math.log(2.0)
    ln_s_sat = torch.randn(B, device=device) * sigma_s
    s_sat = torch.exp(ln_s_sat) * sat_mask.squeeze(1)
    s_sat = 1.0 - sat_mask.squeeze(1) + s_sat
    v4 = torch.tensor([1.0, 1.0, 1.0, 0.0], device=device) / math.sqrt(3.0)
    vTv = v4.unsqueeze(1) @ v4.unsqueeze(0)
    I4 = torch.eye(4, device=device)
    Sat_mat = torch.zeros((B, 4, 4), device=device)
    for b_idx in range(B):
        Sat_mat[b_idx] = vTv + (I4 - vTv) * s_sat[b_idx]
    Cmat = torch.bmm(Sat_mat, Cmat)

    # Color transform per pixel
    flat_rgb = Y01.permute(0, 2, 3, 1).reshape(B, -1, 3)
    ones = torch.ones((B, flat_rgb.shape[1], 1), device=device)
    rgb1 = torch.cat([flat_rgb, ones], dim=-1).permute(0, 2, 1)
    out4 = torch.bmm(Cmat, rgb1)
    out3 = out4[:, :3, :].view(B, 3, H, W)
    Y_color = out3 * 2.0 - 1.0
    return Y_color


def image_space_filter_and_corrupt(Y, p):
    B, C, H, W = Y.shape
    device = Y.device

    Y01 = (Y + 1.0) / 2

    # Freq bands and lambda
    lam = torch.tensor([10.0, 1.0, 1.0, 1.0], device=device) / 13.0

    # Build 4 bandpass filters by dilating a 2-tap lowpass
    low1d = torch.tensor([0.5, 0.5], device=device)
    def make_bandpass(dilate):
        L = low1d.shape[0]
        size1 = dilate * (L - 1) + 1
        filt1d = torch.zeros((size1,), device=device)
        filt1d[::dilate] = low1d
        filt2d = filt1d.unsqueeze(1) @ filt1d.unsqueeze(0)
        filt2d = filt2d / filt2d.sum()
        size2 = (dilate * 2) * (L - 1) + 1
        filt1d2 = torch.zeros((size2,), device=device)
        filt1d2[:: (dilate * 2)] = low1d
        filt2d2 = filt1d2.unsqueeze(1) @ filt1d2.unsqueeze(0)
        filt2d2 = filt2d2 / filt2d2.sum()
        if size1 < size2:
            diff = size2 - size1
            pad0 = diff // 2
            pad1 = diff - pad0
            filt2d = F.pad(filt2d, (pad0, pad1, pad0, pad1), mode='constant', value=0.0)
        elif size2 < size1:
            diff = size1 - size2
            pad0 = diff // 2
            pad1 = diff - pad0
            filt2d2 = F.pad(filt2d2, (pad0, pad1, pad0, pad1), mode='constant', value=0.0)
        return filt2d - filt2d2


    bp1 = make_bandpass(1)
    bp2 = make_bandpass(2)
    bp3 = make_bandpass(4)
    bp4 = make_bandpass(8)
    bandpasses = [bp1, bp2, bp3, bp4]

    # Compute gain vector g
    g = torch.ones((B, 4), device=device)
    for i, bp in enumerate(bandpasses):
        mask_i = (torch.rand(B, device=device) < p).float()
        sigma = math.log(2.0)
        ln_ti = torch.randn(B, device=device) * sigma
        ti = torch.exp(ln_ti) * mask_i
        t = torch.ones((B, 4), device=device)
        t[:, i] = ti
        denom = torch.sqrt(torch.sum((t ** 2) * lam.unsqueeze(0), dim=1, keepdim=True)) + 1e-8
        t_norm = t / denom
        g = g * t_norm

    # Build combined bandpass kernel H0 and apply separable conv
    def make_bp_kernel(bp, target_h, target_w):
        h, w = bp.shape
        pad_h = target_h - h
        pad_w = target_w - w
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        bp_padded = F.pad(bp, (pad_left, pad_right, pad_top, pad_bot), mode='constant', value=0.0)
        bp_3ch = bp_padded.unsqueeze(0).repeat(3, 1, 1)
        return bp_3ch

    max_kh = max(bp.shape[0] for bp in bandpasses)
    max_kw = max(bp.shape[1] for bp in bandpasses)
    bp_kernels = [make_bp_kernel(bp, max_kh, max_kw) for bp in bandpasses]

    pad_y = max_kh // 2
    pad_x = max_kw // 2
    Y_padded = F.pad(Y01, (pad_x, pad_x, pad_y, pad_y), mode='reflect')

    combined_kernels = []
    for b_idx in range(B):
        ck = torch.zeros((3, max_kh, max_kw), device=device)
        for i, bk in enumerate(bp_kernels):
            gi = g[b_idx, i]
            ck += gi * bk
        combined_kernels.append(ck.unsqueeze(1))
    combined_kernels = torch.cat(combined_kernels, dim=0)

    Y_dw = Y_padded.view(1, B*3, Y_padded.shape[2], Y_padded.shape[3])
    Y_filt_dw = F.conv2d(Y_dw, combined_kernels, bias=None, stride=1, padding=0, groups=B*3)
    Y_filt = Y_filt_dw.view(B, 3, H, W)

    # RGB noise with probability p
    noise_mask = (torch.rand(B, device=device) < p).float().view(B, 1, 1, 1)
    sigma_noise = torch.abs(torch.randn(B, device=device)) * 0.1
    sigma_noise = sigma_noise.view(B, 1, 1, 1)
    noise = torch.randn_like(Y_filt) * sigma_noise * noise_mask
    Y_filt = Y_filt + noise

    # Cutout with probability p
    cut_mask = (torch.rand(B, device=device) < p).float()
    cx = torch.rand(B, device=device)
    cy = torch.rand(B, device=device)
    rlo_x = torch.clamp(torch.round((cx - 0.25) * W), 0, W).long()
    rlo_y = torch.clamp(torch.round((cy - 0.25) * H), 0, H).long()
    rhi_x = torch.clamp(torch.round((cx + 0.25) * W), 0, W).long()
    rhi_y = torch.clamp(torch.round((cy + 0.25) * H), 0, H).long()
    for b_idx in range(B):
        if cut_mask[b_idx] == 1.0:
            y0 = rlo_y[b_idx].item()
            y1 = rhi_y[b_idx].item()
            x0 = rlo_x[b_idx].item()
            x1 = rhi_x[b_idx].item()
            Y_filt[b_idx, :, y0:y1, x0:x1] = 0.0

    Y_out = Y_filt * 2.0 - 1.0
    return Y_out

def augmentation(X, p):
    B, C, H, W = X.shape
    device = X.device

    X = X.to(torch.float32) / 127.5 - 1.0

    G = init_G(device, B)
    G = pixel_blitting(G, p, W, H)
    G = general_geometric(G, p, W, H)
    Y = pad_and_warp_image(X, G, W, H)
    Y = color_transform(Y, p)
    Y = image_space_filter_and_corrupt(Y, p)
    return Y
