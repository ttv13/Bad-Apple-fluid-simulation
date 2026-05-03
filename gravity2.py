from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFilter
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import time

# ============================================================
# USER SETTINGS
# ============================================================
FRAME_FOLDER = Path(__file__).resolve().parent / "frames"
OUTPUT_DIR = Path(__file__).resolve().parent / "rendered_frames"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "bad_apple_render_checkpoint.pkl"
VIDEO_OUTPUT_PATH = Path(__file__).resolve().parent / "bad_apple_particles.mp4"

WIDTH = 640
HEIGHT = 480
FPS = 30
THRESHOLD = 128

# Rendering / output
DRAW_VIDEO_BACKGROUND = False
VIDEO_CRF = 18
EXPORT_MP4_AT_END = True
OVERWRITE_EXISTING_OUTPUT_IF_NOT_RESUMING = False
MAX_VIDEO_FRAMES: int | None = None   # None = render full video

# Particle system
NUM_PARTICLES = 4000
PARTICLE_RADIUS = 1
SPAWN_STRIDE = 2                      # respawn/spawn sampling density
INITIAL_SPEED_MIN = 80.0
INITIAL_SPEED_MAX = 230.0

# Physics: high precision defaults
BASE_PHYSICS_STEPS_PER_VIDEO_FRAME = 120
MAX_PHYSICS_STEPS_PER_VIDEO_FRAME = 480
MAX_PARTICLE_TRAVEL_PER_SUBSTEP = 0.20
PARTICLE_COLLISION_ITERATIONS = 2

# Collisions
BOUNDARY_RESTITUTION = 0.98
PARTICLE_RESTITUTION = 0.99
CELL_SIZE = max(PARTICLE_RADIUS * 4, 4)

# Moving boundary sweep / frame-to-frame mask jumps
MOVING_BOUNDARY_MAX_SWEEP_PIXELS = 120
MOVING_BOUNDARY_PUSH_EPSILON = 1.25
MOVING_BOUNDARY_MIN_INWARD_SPEED = 80.0

# Checkpoint / resume
RESUME_IF_CHECKPOINT_EXISTS = True
CHECKPOINT_EVERY_N_FRAMES = 25
FRAME_CACHE_SIZE = 4

# Diagnostics
PRINT_PROGRESS_EVERY_N_FRAMES = 1
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ============================================================
# DATA TYPES
# ============================================================
@dataclass(slots=True)
class FrameData:
    mask_bytes: bytes
    safe_mask_bytes: bytes
    white_points: tuple[tuple[int, int], ...]
    has_white: bool
    file_name: str


@dataclass(slots=True)
class Particle:
    x: float
    y: float
    vx: float
    vy: float


# ============================================================
# FRAME LOADING
# ============================================================
def natural_key(path: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def list_frame_paths() -> list[Path]:
    if not FRAME_FOLDER.exists():
        raise FileNotFoundError(f"Missing frames folder: {FRAME_FOLDER}")

    files = sorted(
        [p for p in FRAME_FOLDER.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )
    if not files:
        raise RuntimeError(f"No image frames found in: {FRAME_FOLDER}")
    return files


FRAME_PATHS = list_frame_paths()
TOTAL_SOURCE_FRAMES = len(FRAME_PATHS)
TOTAL_RENDER_FRAMES = min(TOTAL_SOURCE_FRAMES, MAX_VIDEO_FRAMES) if MAX_VIDEO_FRAMES is not None else TOTAL_SOURCE_FRAMES
SAFE_MASK_FILTER_SIZE = max(1, PARTICLE_RADIUS * 2 + 1)


@lru_cache(maxsize=FRAME_CACHE_SIZE)
def load_frame_data(frame_index: int) -> FrameData:
    path = FRAME_PATHS[frame_index]

    with Image.open(path) as img:
        gray = img.convert("L").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

    mask = gray.point(lambda p: 255 if p >= THRESHOLD else 0, mode="L")
    mask_bytes = mask.tobytes()

    if SAFE_MASK_FILTER_SIZE > 1:
        safe_mask = mask.filter(ImageFilter.MinFilter(SAFE_MASK_FILTER_SIZE))
    else:
        safe_mask = mask
    safe_mask_bytes = safe_mask.tobytes()

    white_points: list[tuple[int, int]] = []
    for y in range(0, HEIGHT, SPAWN_STRIDE):
        row = y * WIDTH
        for x in range(0, WIDTH, SPAWN_STRIDE):
            if safe_mask_bytes[row + x] > 0:
                white_points.append((x, y))

    return FrameData(
        mask_bytes=mask_bytes,
        safe_mask_bytes=safe_mask_bytes,
        white_points=tuple(white_points),
        has_white=bool(white_points),
        file_name=path.name,
    )


# ============================================================
# MASK HELPERS
# ============================================================
def is_inside(mask: bytes, x: float, y: float) -> bool:
    ix = int(x)
    iy = int(y)
    if ix < 0 or ix >= WIDTH or iy < 0 or iy >= HEIGHT:
        return False
    return mask[iy * WIDTH + ix] > 0


def mask_value(mask: bytes, ix: int, iy: int) -> float:
    if ix < 0 or ix >= WIDTH or iy < 0 or iy >= HEIGHT:
        return 0.0
    return 1.0 if mask[iy * WIDTH + ix] > 0 else 0.0


def inward_normal(mask: bytes, x: float, y: float) -> tuple[float, float]:
    ix = int(x)
    iy = int(y)
    gx = mask_value(mask, ix + 1, iy) - mask_value(mask, ix - 1, iy)
    gy = mask_value(mask, ix, iy + 1) - mask_value(mask, ix, iy - 1)
    mag = math.hypot(gx, gy)
    if mag < 1e-8:
        return 0.0, 0.0
    return gx / mag, gy / mag


def nearest_inside_point(mask: bytes, x: float, y: float, max_radius: int) -> tuple[float, float] | None:
    cx = int(round(x))
    cy = int(round(y))

    if 0 <= cx < WIDTH and 0 <= cy < HEIGHT and mask[cy * WIDTH + cx] > 0:
        return float(cx), float(cy)

    for r in range(1, max_radius + 1):
        left = cx - r
        right = cx + r
        top = cy - r
        bottom = cy + r

        # top and bottom rows
        if 0 <= top < HEIGHT:
            for xx in range(max(0, left), min(WIDTH, right + 1)):
                if mask[top * WIDTH + xx] > 0:
                    return float(xx), float(top)
        if 0 <= bottom < HEIGHT:
            for xx in range(max(0, left), min(WIDTH, right + 1)):
                if mask[bottom * WIDTH + xx] > 0:
                    return float(xx), float(bottom)

        # left and right columns (skip corners already checked)
        if 0 <= left < WIDTH:
            for yy in range(max(0, top + 1), min(HEIGHT, bottom)):
                if mask[yy * WIDTH + left] > 0:
                    return float(left), float(yy)
        if 0 <= right < WIDTH:
            for yy in range(max(0, top + 1), min(HEIGHT, bottom)):
                if mask[yy * WIDTH + right] > 0:
                    return float(right), float(yy)

    return None


# ============================================================
# PARTICLE HELPERS
# ============================================================
def pick_spawn_point(frame: FrameData) -> tuple[float, float]:
    if not frame.white_points:
        raise RuntimeError("Cannot spawn a particle: frame has no valid white points.")
    x, y = random.choice(frame.white_points)
    return float(x), float(y)


def random_velocity() -> tuple[float, float]:
    angle = random.uniform(0.0, 2.0 * math.pi)
    speed = random.uniform(INITIAL_SPEED_MIN, INITIAL_SPEED_MAX)
    return math.cos(angle) * speed, math.sin(angle) * speed


def spawn_particle(frame: FrameData) -> Particle:
    x, y = pick_spawn_point(frame)
    vx, vy = random_velocity()
    jitter = min(0.49, PARTICLE_RADIUS * 0.35 + 0.15)
    return Particle(
        x=x + random.uniform(-jitter, jitter),
        y=y + random.uniform(-jitter, jitter),
        vx=vx,
        vy=vy,
    )


def ensure_particles(frame: FrameData, particles: list[Particle]) -> None:
    if particles:
        return
    if not frame.has_white:
        return
    particles.extend(spawn_particle(frame) for _ in range(NUM_PARTICLES))


def respawn_particle(p: Particle, frame: FrameData) -> None:
    fresh = spawn_particle(frame)
    p.x = fresh.x
    p.y = fresh.y
    p.vx = fresh.vx
    p.vy = fresh.vy


# ============================================================
# PHYSICS
# ============================================================
def choose_physics_steps(particles: list[Particle]) -> int:
    if not particles:
        return BASE_PHYSICS_STEPS_PER_VIDEO_FRAME

    max_speed = 0.0
    for p in particles:
        speed = math.hypot(p.vx, p.vy)
        if speed > max_speed:
            max_speed = speed

    base = BASE_PHYSICS_STEPS_PER_VIDEO_FRAME
    required = base
    if max_speed > 1e-9:
        travel_per_video_frame = max_speed / FPS
        required = math.ceil(travel_per_video_frame / MAX_PARTICLE_TRAVEL_PER_SUBSTEP)

    steps = max(base, required)
    return min(MAX_PHYSICS_STEPS_PER_VIDEO_FRAME, steps)


def reflect_velocity(vx: float, vy: float, nx: float, ny: float, restitution: float) -> tuple[float, float]:
    dot = vx * nx + vy * ny
    return (vx - 2.0 * dot * nx) * restitution, (vy - 2.0 * dot * ny) * restitution


def move_particle_one_substep(p: Particle, frame: FrameData, dt: float) -> None:
    old_x = p.x
    old_y = p.y
    new_x = p.x + p.vx * dt
    new_y = p.y + p.vy * dt

    min_x = PARTICLE_RADIUS
    max_x = WIDTH - PARTICLE_RADIUS - 1
    min_y = PARTICLE_RADIUS
    max_y = HEIGHT - PARTICLE_RADIUS - 1

    # screen boundaries first
    if new_x < min_x:
        new_x = min_x + (min_x - new_x)
        p.vx = abs(p.vx) * BOUNDARY_RESTITUTION
    elif new_x > max_x:
        new_x = max_x - (new_x - max_x)
        p.vx = -abs(p.vx) * BOUNDARY_RESTITUTION

    if new_y < min_y:
        new_y = min_y + (min_y - new_y)
        p.vy = abs(p.vy) * BOUNDARY_RESTITUTION
    elif new_y > max_y:
        new_y = max_y - (new_y - max_y)
        p.vy = -abs(p.vy) * BOUNDARY_RESTITUTION

    # silhouette boundary using safe mask so full radius stays inside
    if frame.has_white and not is_inside(frame.safe_mask_bytes, new_x, new_y):
        nx, ny = inward_normal(frame.safe_mask_bytes, new_x, new_y)
        if nx == 0.0 and ny == 0.0:
            back_x = old_x - new_x
            back_y = old_y - new_y
            mag = math.hypot(back_x, back_y)
            if mag > 1e-9:
                nx = back_x / mag
                ny = back_y / mag
            else:
                nx, ny = 0.0, -1.0

        p.vx, p.vy = reflect_velocity(p.vx, p.vy, nx, ny, BOUNDARY_RESTITUTION)
        p.x = max(min_x, min(max_x, old_x + nx * MOVING_BOUNDARY_PUSH_EPSILON))
        p.y = max(min_y, min(max_y, old_y + ny * MOVING_BOUNDARY_PUSH_EPSILON))
    else:
        p.x = max(min_x, min(max_x, new_x))
        p.y = max(min_y, min(max_y, new_y))


def solve_particle_collisions(particles: list[Particle]) -> None:
    if len(particles) < 2:
        return

    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, p in enumerate(particles):
        cell = (int(p.x // CELL_SIZE), int(p.y // CELL_SIZE))
        grid[cell].append(idx)

    min_dist = PARTICLE_RADIUS * 2.0
    min_dist_sq = min_dist * min_dist
    neighbor_offsets = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 1)]

    for (cx, cy), here in grid.items():
        # same-cell pairs
        count_here = len(here)
        for i in range(count_here):
            idx_a = here[i]
            pa = particles[idx_a]
            for j in range(i + 1, count_here):
                idx_b = here[j]
                pb = particles[idx_b]
                resolve_pair(pa, pb, min_dist, min_dist_sq)

        # neighbor cells, half neighborhood only to avoid duplicate pairs
        for ox, oy in neighbor_offsets[1:]:
            other = grid.get((cx + ox, cy + oy))
            if not other:
                continue
            for idx_a in here:
                pa = particles[idx_a]
                for idx_b in other:
                    pb = particles[idx_b]
                    resolve_pair(pa, pb, min_dist, min_dist_sq)


def resolve_pair(pa: Particle, pb: Particle, min_dist: float, min_dist_sq: float) -> None:
    dx = pb.x - pa.x
    dy = pb.y - pa.y
    dist_sq = dx * dx + dy * dy
    if dist_sq >= min_dist_sq:
        return

    if dist_sq < 1e-12:
        angle = random.uniform(0.0, 2.0 * math.pi)
        nx = math.cos(angle)
        ny = math.sin(angle)
        dist = 1e-6
    else:
        dist = math.sqrt(dist_sq)
        nx = dx / dist
        ny = dy / dist

    overlap = min_dist - dist
    correction = overlap * 0.5
    pa.x -= nx * correction
    pa.y -= ny * correction
    pb.x += nx * correction
    pb.y += ny * correction

    # equal mass impulse
    rvx = pb.vx - pa.vx
    rvy = pb.vy - pa.vy
    rel_normal = rvx * nx + rvy * ny
    if rel_normal >= 0.0:
        return

    impulse = -(1.0 + PARTICLE_RESTITUTION) * rel_normal * 0.5
    ix = impulse * nx
    iy = impulse * ny
    pa.vx -= ix
    pa.vy -= iy
    pb.vx += ix
    pb.vy += iy


def reconcile_particles_with_frame_change(
    particles: list[Particle],
    prev_frame: FrameData | None,
    new_frame: FrameData,
) -> tuple[int, int]:
    if not particles:
        return 0, 0

    if not new_frame.has_white:
        return 0, 0

    swept_hits = 0
    respawns = 0

    for p in particles:
        if is_inside(new_frame.mask_bytes, p.x, p.y):
            continue

        was_inside_prev = prev_frame is not None and prev_frame.has_white and is_inside(prev_frame.mask_bytes, p.x, p.y)

        if was_inside_prev:
            nearby = nearest_inside_point(new_frame.safe_mask_bytes, p.x, p.y, MOVING_BOUNDARY_MAX_SWEEP_PIXELS)
            if nearby is not None:
                hit_x, hit_y = nearby
                nx, ny = inward_normal(new_frame.safe_mask_bytes, hit_x, hit_y)
                if nx == 0.0 and ny == 0.0:
                    vx = hit_x - p.x
                    vy = hit_y - p.y
                    mag = math.hypot(vx, vy)
                    if mag > 1e-9:
                        nx = vx / mag
                        ny = vy / mag
                    else:
                        nx, ny = 0.0, -1.0

                p.x = max(PARTICLE_RADIUS, min(WIDTH - PARTICLE_RADIUS - 1, hit_x + nx * MOVING_BOUNDARY_PUSH_EPSILON))
                p.y = max(PARTICLE_RADIUS, min(HEIGHT - PARTICLE_RADIUS - 1, hit_y + ny * MOVING_BOUNDARY_PUSH_EPSILON))

                inward_speed = p.vx * nx + p.vy * ny
                if inward_speed < 0.0:
                    p.vx, p.vy = reflect_velocity(p.vx, p.vy, nx, ny, BOUNDARY_RESTITUTION)
                    inward_speed = p.vx * nx + p.vy * ny

                if inward_speed < MOVING_BOUNDARY_MIN_INWARD_SPEED:
                    boost = MOVING_BOUNDARY_MIN_INWARD_SPEED - inward_speed
                    p.vx += nx * boost
                    p.vy += ny * boost

                swept_hits += 1
                continue

        # not recoverable as a swept boundary collision -> respawn
        respawn_particle(p, new_frame)
        respawns += 1

    return swept_hits, respawns


# ============================================================
# RENDERING
# ============================================================
def speed_to_color(speed: float) -> tuple[int, int, int]:
    t = max(0.0, min(speed / INITIAL_SPEED_MAX, 1.0))
    if t < 0.5:
        u = t / 0.5
        r = int(80 + 175 * u)
        g = int(170 + 70 * u)
        b = int(255 - 205 * u)
    else:
        u = (t - 0.5) / 0.5
        r = 255
        g = int(240 - 190 * u)
        b = int(50 - 35 * u)
    return r, g, b


def render_frame(frame: FrameData, particles: list[Particle], frame_index: int) -> Image.Image:
    if DRAW_VIDEO_BACKGROUND:
        # reopen only when explicitly requested
        with Image.open(FRAME_PATHS[frame_index]) as img:
            canvas = img.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    else:
        canvas = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))

    if not particles or not frame.has_white:
        return canvas

    draw = ImageDraw.Draw(canvas)
    r = PARTICLE_RADIUS
    raw_mask = frame.mask_bytes

    if r <= 1:
        for p in particles:
            if not is_inside(raw_mask, p.x, p.y):
                continue
            speed = math.hypot(p.vx, p.vy)
            color = speed_to_color(speed)
            ix = int(p.x)
            iy = int(p.y)
            draw.point((ix, iy), fill=color)
    else:
        for p in particles:
            if not is_inside(raw_mask, p.x, p.y):
                continue
            speed = math.hypot(p.vx, p.vy)
            color = speed_to_color(speed)
            draw.ellipse((p.x - r, p.y - r, p.x + r, p.y + r), fill=color)

    return canvas


# ============================================================
# CHECKPOINTING
# ============================================================
def particles_to_state(particles: list[Particle]) -> list[tuple[float, float, float, float]]:
    return [(p.x, p.y, p.vx, p.vy) for p in particles]


def particles_from_state(items: list[tuple[float, float, float, float]]) -> list[Particle]:
    return [Particle(*item) for item in items]


def save_checkpoint(next_frame_index: int, particles: list[Particle]) -> None:
    state = {
        "next_frame_index": next_frame_index,
        "particles": particles_to_state(particles),
        "random_state": random.getstate(),
        "settings": {
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "num_particles": NUM_PARTICLES,
            "particle_radius": PARTICLE_RADIUS,
        },
    }

    tmp_path = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp_path, CHECKPOINT_PATH)


def try_load_checkpoint() -> tuple[int, list[Particle]] | None:
    if not CHECKPOINT_PATH.exists() or not RESUME_IF_CHECKPOINT_EXISTS:
        return None

    with open(CHECKPOINT_PATH, "rb") as f:
        state = pickle.load(f)

    next_frame_index = int(state["next_frame_index"])
    particles = particles_from_state(state["particles"])
    random.setstate(state["random_state"])
    return next_frame_index, particles


# ============================================================
# EXPORT
# ============================================================
def export_mp4() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[info] ffmpeg not found on PATH. PNG frames were rendered successfully, but MP4 export was skipped.")
        return

    pattern = str(OUTPUT_DIR / "frame_%06d.png")
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(VIDEO_CRF),
        str(VIDEO_OUTPUT_PATH),
    ]
    print("[info] Exporting MP4 with ffmpeg...")
    subprocess.run(cmd, check=True)
    print(f"[done] MP4 saved to: {VIDEO_OUTPUT_PATH}")


# ============================================================
# MAIN LOOP
# ============================================================
def prepare_output_dir(start_index: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if start_index == 0 and OVERWRITE_EXISTING_OUTPUT_IF_NOT_RESUMING:
        for path in OUTPUT_DIR.glob("frame_*.png"):
            path.unlink()


def format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def run() -> None:
    print(f"[info] Source frames: {TOTAL_SOURCE_FRAMES}")
    print(f"[info] Frames to render: {TOTAL_RENDER_FRAMES}")
    print(f"[info] Output folder: {OUTPUT_DIR}")
    print(f"[info] Num particles: {NUM_PARTICLES}")
    print(f"[info] Physics steps/frame: base={BASE_PHYSICS_STEPS_PER_VIDEO_FRAME}, max={MAX_PHYSICS_STEPS_PER_VIDEO_FRAME}")

    checkpoint = try_load_checkpoint()
    if checkpoint is not None:
        start_frame_index, particles = checkpoint
        print(f"[resume] Loaded checkpoint. Resuming at frame index {start_frame_index}.")
    else:
        start_frame_index = 0
        particles = []
        print("[info] Starting fresh render.")

    prepare_output_dir(start_frame_index)

    start_time = time.perf_counter()
    prev_frame = load_frame_data(start_frame_index - 1) if start_frame_index > 0 else None

    for frame_index in range(start_frame_index, TOTAL_RENDER_FRAMES):
        frame = load_frame_data(frame_index)

        # lazily create particles when the first non-black frame appears
        ensure_particles(frame, particles)

        swept_hits = 0
        respawns = 0
        if particles and frame.has_white:
            swept_hits, respawns = reconcile_particles_with_frame_change(particles, prev_frame, frame)

            physics_steps = choose_physics_steps(particles)
            dt = 1.0 / FPS / physics_steps
            for _ in range(physics_steps):
                for p in particles:
                    move_particle_one_substep(p, frame, dt)
                for _ in range(PARTICLE_COLLISION_ITERATIONS):
                    solve_particle_collisions(particles)
        else:
            physics_steps = 0

        image = render_frame(frame, particles, frame_index)
        out_path = OUTPUT_DIR / f"frame_{frame_index:06d}.png"
        image.save(out_path)

        if (frame_index + 1) % CHECKPOINT_EVERY_N_FRAMES == 0 or frame_index == TOTAL_RENDER_FRAMES - 1:
            save_checkpoint(frame_index + 1, particles)

        if PRINT_PROGRESS_EVERY_N_FRAMES > 0 and (
            frame_index == start_frame_index
            or frame_index == TOTAL_RENDER_FRAMES - 1
            or (frame_index + 1) % PRINT_PROGRESS_EVERY_N_FRAMES == 0
        ):
            elapsed = time.perf_counter() - start_time
            done = frame_index - start_frame_index + 1
            avg = elapsed / max(done, 1)
            remaining = TOTAL_RENDER_FRAMES - frame_index - 1
            eta = avg * remaining
            print(
                f"[render] {frame_index + 1}/{TOTAL_RENDER_FRAMES} | "
                f"file={frame.file_name} | physics_steps={physics_steps} | swept={swept_hits} | "
                f"respawns={respawns} | elapsed={elapsed:.1f}s | eta={format_eta(eta)}"
            )

        prev_frame = frame

    print("[done] PNG sequence finished.")
    if EXPORT_MP4_AT_END:
        export_mp4()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n[stopped] Render interrupted by user. Checkpoint was preserved.")
        sys.exit(1)
