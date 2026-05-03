import os
import sys
import math
import random
import cv2
import numpy as np
import pygame
import torch

# --------------------------
# SETTINGS
# --------------------------
FRAME_FOLDER = "frames"      # folder with black/white frames
WIDTH, HEIGHT = 640, 480     # display size
NUM_PARTICLES = 4000         # tune to GPU memory and performance
PARTICLE_RADIUS = 4          # radius in pixels (larger particles)
FPS = 30

# Physics
# GRAVITY = 800.0              # px / s^2
GRAVITY = 1.0
DT = 1.0 / FPS
BOUNCE_RESTITUTION = 0.6
FLOW_STRENGTH = 220.0
MAX_SEEK_SPEED = 600.0

# Colors (RGB)
COLOR_ON_WHITE = torch.tensor([255, 200, 60], dtype=torch.uint8)   # particle color when on white

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# LOAD FRAMES
# --------------------------
def load_frames(folder, width, height):
    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png"))
    ])
    frames = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        frames.append((img, bw))
    return frames

frames = load_frames(FRAME_FOLDER, WIDTH, HEIGHT)
if not frames:
    print("No frames found in", FRAME_FOLDER)
    sys.exit(1)

# Convert frames to torch tensors on device for fast sampling
color_tensors = []
bw_tensors = []
for img_bgr, bw in frames:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    color_tensors.append(torch.from_numpy(rgb).to(device=device, dtype=torch.uint8))
    bw_tensors.append(torch.from_numpy(bw).to(device=device, dtype=torch.uint8))

# --------------------------
# PRECOMPUTE DIRECTION FIELDS (to nearest white) as torch tensors
# --------------------------
def compute_dir_field_torch(bw_np):
    # bw_np: numpy uint8 (0 or 255)
    inv = (bw_np == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)
    gy, gx = np.gradient(dist)
    vx = -gx
    vy = -gy
    mag = np.sqrt(vx*vx + vy*vy) + 1e-8
    vx /= mag
    vy /= mag
    mask_white = (bw_np == 255)
    vx[mask_white] = 0.0
    vy[mask_white] = 0.0
    return torch.from_numpy(vx).to(device=device, dtype=torch.float32), \
           torch.from_numpy(vy).to(device=device, dtype=torch.float32)

dir_fields = [compute_dir_field_torch(bw.cpu().numpy()) for bw in bw_tensors]

# --------------------------
# PARTICLE STATE (torch tensors)
# --------------------------
N = NUM_PARTICLES
rng = np.random.default_rng()

# spawn only on white pixels of first frame
bw0 = bw_tensors[0].cpu().numpy()
white_indices = np.argwhere(bw0 == 255)
if len(white_indices) == 0:
    raise RuntimeError("No white pixels in first frame to spawn particles.")

idx = rng.integers(0, len(white_indices), size=N)
ys0, xs0 = white_indices[idx].T
pos_x = torch.from_numpy(xs0.astype(np.float32) + rng.random(N).astype(np.float32)).to(device=device)
pos_y = torch.from_numpy(ys0.astype(np.float32) + rng.random(N).astype(np.float32)).to(device=device)

vel_x = torch.from_numpy(rng.uniform(-30, 30, size=N).astype(np.float32)).to(device=device)
vel_y = torch.from_numpy(rng.uniform(-10, 10, size=N).astype(np.float32)).to(device=device)

on_white = torch.ones(N, dtype=torch.bool, device=device)

# --------------------------
# PREPARE PARTICLE STAMP OFFSETS (disk)
# --------------------------
r = PARTICLE_RADIUS
# create disk mask offsets
ys_off, xs_off = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1), indexing='ij')
mask_disk = (xs_off**2 + ys_off**2) <= (r*r)
offs_y = torch.from_numpy(ys_off[mask_disk].astype(np.int32)).to(device=device)
offs_x = torch.from_numpy(xs_off[mask_disk].astype(np.int32)).to(device=device)
K = offs_x.numel()
print(f"Particle stamp radius {r}, kernel size {K} pixels")

# --------------------------
# PYGAME INIT
# --------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyTorch GPU White-only Particles")
clock = pygame.time.Clock()

frame_index = 0
running = True

# Pre-allocate canvas on GPU (uint8)
canvas = torch.empty((HEIGHT, WIDTH, 3), dtype=torch.uint8, device=device)

# --------------------------
# MAIN LOOP
# --------------------------
while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # current and next frames
    color_img = color_tensors[frame_index]        # (H,W,3) uint8 on device
    bw_current = bw_tensors[frame_index]          # (H,W) uint8 on device
    bw_next = bw_tensors[(frame_index + 1) % len(frames)]
    dir_x_next, dir_y_next = dir_fields[(frame_index + 1) % len(frames)]

    # --------------------------
    # PHYSICS (vectorized on GPU)
    # --------------------------
    # gravity
    vel_y = vel_y + (GRAVITY * DT)

    # sample integer indices
    ix = pos_x.to(torch.int64).clamp(0, WIDTH - 1)
    iy = pos_y.to(torch.int64).clamp(0, HEIGHT - 1)

    # currently white and next white masks
    currently_white = (bw_current[iy, ix] == 255)
    next_white = (bw_next[iy, ix] == 255)

    # need to seek if currently on black or white disappears next frame
    need_seek = (~currently_white) | (currently_white & (~next_white))
    if need_seek.any():
        sx = dir_x_next[iy[need_seek], ix[need_seek]]
        sy = dir_y_next[iy[need_seek], ix[need_seek]]
        vel_x[need_seek] = vel_x[need_seek] + sx * (FLOW_STRENGTH * DT)
        vel_y[need_seek] = vel_y[need_seek] + sy * (FLOW_STRENGTH * DT)
        # cap speed
        speed = torch.hypot(vel_x[need_seek], vel_y[need_seek])
        over = speed > MAX_SEEK_SPEED
        if over.any():
            scale = (MAX_SEEK_SPEED / (speed[over] + 1e-8)).to(device)
            vel_x[need_seek][over] *= scale
            vel_y[need_seek][over] *= scale

    # integrate
    new_x = pos_x + vel_x * DT
    new_y = pos_y + vel_y * DT

    # screen edge collisions
    left_mask = new_x < 0
    if left_mask.any():
        new_x[left_mask] = -new_x[left_mask]
        vel_x[left_mask] = -vel_x[left_mask] * BOUNCE_RESTITUTION

    right_mask = new_x >= WIDTH
    if right_mask.any():
        new_x[right_mask] = 2*(WIDTH-1) - new_x[right_mask]
        vel_x[right_mask] = -vel_x[right_mask] * BOUNCE_RESTITUTION

    top_mask = new_y < 0
    if top_mask.any():
        new_y[top_mask] = -new_y[top_mask]
        vel_y[top_mask] = -vel_y[top_mask] * BOUNCE_RESTITUTION

    bottom_mask = new_y >= HEIGHT
    if bottom_mask.any():
        new_y[bottom_mask] = 2*(HEIGHT-1) - new_y[bottom_mask]
        vel_y[bottom_mask] = -vel_y[bottom_mask] * BOUNCE_RESTITUTION

    # boundary collision with black pixels (sample bw_current at new positions)
    nx = new_x.to(torch.int64).clamp(0, WIDTH - 1)
    ny = new_y.to(torch.int64).clamp(0, HEIGHT - 1)
    hits_black = (bw_current[ny, nx] == 0)

    if hits_black.any():
        # compute gradient of bw_current on CPU once per frame (cheap)
        # (we could precompute gradients as torch tensors; do it here for clarity)
        f = (bw_current.to(torch.float32) / 255.0).cpu().numpy()
        gy, gx = np.gradient(f)
        gx_t = torch.from_numpy(gx).to(device=device, dtype=torch.float32)
        gy_t = torch.from_numpy(gy).to(device=device, dtype=torch.float32)

        gx_vals = gx_t[ny[hits_black], nx[hits_black]]
        gy_vals = gy_t[ny[hits_black], nx[hits_black]]
        nmag = torch.hypot(gx_vals, gy_vals)
        valid = nmag > 1e-6
        if valid.any():
            nx_g = gx_vals[valid] / nmag[valid]
            ny_g = gy_vals[valid] / nmag[valid]
            idxs = torch.nonzero(hits_black, as_tuple=False).squeeze(1)[valid]
            vdotn = vel_x[idxs] * nx_g + vel_y[idxs] * ny_g
            vel_x[idxs] = (vel_x[idxs] - 2 * vdotn * nx_g) * BOUNCE_RESTITUTION
            vel_y[idxs] = (vel_y[idxs] - 2 * vdotn * ny_g) * BOUNCE_RESTITUTION
            new_x[idxs] = new_x[idxs] + nx_g * 1.5
            new_y[idxs] = new_y[idxs] + ny_g * 1.5
        # fallback invert vertical for remaining hits
        if (~valid).any():
            idxs2 = torch.nonzero(hits_black, as_tuple=False).squeeze(1)[~valid]
            vel_y[idxs2] = -vel_y[idxs2] * BOUNCE_RESTITUTION
            new_y[idxs2] = pos_y[idxs2] + vel_y[idxs2] * DT

    # commit positions
    pos_x = new_x.clamp(0, WIDTH - 1)
    pos_y = new_y.clamp(0, HEIGHT - 1)

    # update on_white flags (based on current frame)
    ix = pos_x.to(torch.int64)
    iy = pos_y.to(torch.int64)
    on_white = (bw_current[iy, ix] == 255)

    # --------------------------
    # RENDERING (stamp particles into GPU canvas)
    # --------------------------
    # copy background color frame into canvas
    canvas[:] = color_img  # (H,W,3) uint8 on device

    if on_white.any():
        # indices of particles to draw
        draw_idx = torch.nonzero(on_white, as_tuple=False).squeeze(1)
        xs = ix[draw_idx]  # int64
        ys = iy[draw_idx]

        # For each offset in the disk kernel, compute flattened indices and set color
        # We'll do this loop over K offsets (K small for small radius)
        H = HEIGHT
        W = WIDTH
        canvas_flat = canvas.view(-1, 3)  # (H*W, 3)

        base_flat = (ys * W + xs).to(torch.int64)  # (M,)
        # Expand base_flat to (M, K) by adding offsets
        # For each offset, compute flat indices and write color
        for k in range(K):
            dx = offs_x[k]
            dy = offs_y[k]
            xs_k = (xs + dx).clamp(0, W - 1)
            ys_k = (ys + dy).clamp(0, H - 1)
            flat_k = (ys_k * W + xs_k).to(torch.int64)
            # scatter: last write wins for overlapping particles
            canvas_flat[flat_k] = COLOR_ON_WHITE.to(device)

    # Transfer canvas to CPU and display via pygame
    canvas_cpu = canvas.cpu().numpy()
    surf = pygame.surfarray.make_surface(np.transpose(canvas_cpu, (1, 0, 2)))
    screen.blit(surf, (0, 0))
    pygame.display.flip()

    frame_index = (frame_index + 1) % len(frames)

pygame.quit()
