#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import sys
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _find_downloaded_video(outdir: Path) -> Path | None:
    exts = [".mp4", ".mkv", ".webm", ".mov"]
    candidates = []
    for ext in exts:
        candidates.extend(outdir.glob(f"bad_apple*{ext}"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def download_youtube_video(url: str, outdir: Path) -> Path:
    """
    Download a YouTube video-only file into outdir using yt-dlp.
    No ffmpeg is required because we avoid audio/video merging.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    existing = _find_downloaded_video(outdir)
    if existing is not None:
        return existing

    try:
        import yt_dlp

        ydl_opts = {
            "noplaylist": True,
            "quiet": False,
            "noprogress": False,
            "format": "bv*[ext=mp4]/b[ext=mp4]/bv*/b",
            "outtmpl": str(outdir / "bad_apple.%(ext)s"),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            requested = info.get("requested_downloads") or []
            for item in requested:
                fp = item.get("filepath")
                if fp and os.path.exists(fp):
                    return Path(fp)

            prepared = ydl.prepare_filename(info)
            if os.path.exists(prepared):
                return Path(prepared)

        found = _find_downloaded_video(outdir)
        if found is not None:
            return found

    except Exception as e:
        print(f"[warn] Python yt_dlp download path failed: {e}")

    yt_dlp_bin = shutil.which("yt-dlp")
    if yt_dlp_bin is None:
        raise RuntimeError(
            "Could not download from YouTube. Install yt-dlp with:\n"
            "  py -m pip install yt-dlp\n"
            "or manually download the video and pass --input <file>."
        )

    cmd = [
        yt_dlp_bin,
        "--no-playlist",
        "-f",
        "bv*[ext=mp4]/b[ext=mp4]/bv*/b",
        "-o",
        str(outdir / "bad_apple.%(ext)s"),
        url,
    ]
    subprocess.run(cmd, check=True)

    found = _find_downloaded_video(outdir)
    if found is None:
        raise RuntimeError("yt-dlp finished but no video file was created.")
    return found


def ensure_video(args, outdir: Path) -> Path:
    if args.input:
        video_path = Path(args.input)
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        return video_path

    if not args.url:
        raise ValueError("Provide either --input <video.mp4> or --url <youtube_url>")

    if not args.redownload:
        existing = _find_downloaded_video(outdir)
        if existing is not None:
            print(f"[info] Reusing existing downloaded video: {existing}")
            return existing

    print(f"[info] Downloading video from: {args.url}")
    return download_youtube_video(args.url, outdir)


def compute_binary_mask(gray: np.ndarray, blur_ksize: int, use_otsu: bool, thresh: int) -> np.ndarray:
    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if use_otsu:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    return binary


def clean_binary_mask(binary: np.ndarray, open_iters: int, close_iters: int) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    if open_iters > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=open_iters)
    if close_iters > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
    return binary


def extract_contours_from_binary(
    binary: np.ndarray,
    min_area: float,
    simplify_epsilon: float,
    keep_holes: bool,
):
    retrieval = cv2.RETR_CCOMP if keep_holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(binary, retrieval, cv2.CHAIN_APPROX_NONE)

    kept = []
    for cnt in contours:
        area = abs(cv2.contourArea(cnt))
        if area < min_area:
            continue
        if simplify_epsilon > 0:
            cnt = cv2.approxPolyDP(cnt, simplify_epsilon, True)
        if len(cnt) >= 2:
            kept.append(cnt)

    return kept, hierarchy


def contour_payload(contours, width: int, height: int, normalize: bool):
    payload = []
    denom = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    for cnt in contours:
        pts = cnt[:, 0, :].astype(np.float32)
        if normalize:
            pts = pts / denom
            pts = np.round(pts, 6)
        else:
            pts = pts.astype(np.int32)
        payload.append(pts.tolist())
    return payload


def overlay_edges_on_frame(frame: np.ndarray, edge_img: np.ndarray) -> np.ndarray:
    out = frame.copy()
    out[edge_img > 0] = (0, 0, 255)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-frame black/white boundary edges and contours from Bad Apple or any high-contrast video."
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--input", type=str, help="Path to a local input video file")
    src.add_argument("--url", type=str, help="YouTube URL to download and process")

    parser.add_argument("--outdir", type=str, default="bad_apple_edges", help="Output directory")
    parser.add_argument("--redownload", action="store_true", help="Force re-download if using --url")

    parser.add_argument("--resize-width", type=int, default=320, help="Resize frames to this width (0 keeps original)")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1, help="Inclusive end frame; -1 means process the full video")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")

    parser.add_argument("--use-otsu", action="store_true", help="Use Otsu thresholding instead of fixed threshold")
    parser.add_argument("--threshold", type=int, default=127, help="Fixed binary threshold if not using Otsu")
    parser.add_argument("--blur-ksize", type=int, default=3, help="Gaussian blur kernel size; 0/1 disables blur")
    parser.add_argument("--open-iters", type=int, default=0, help="Morphological opening iterations")
    parser.add_argument("--close-iters", type=int, default=0, help="Morphological closing iterations")

    parser.add_argument("--min-area", type=float, default=8.0, help="Discard contours with smaller pixel area")
    parser.add_argument("--simplify-epsilon", type=float, default=1.0, help="Douglas-Peucker contour simplification epsilon in pixels")
    parser.add_argument("--no-holes", action="store_true", help="Only keep external contours")
    parser.add_argument("--normalize", action="store_true", help="Save contour points normalized to [0,1]")

    parser.add_argument("--save-edge-pngs", action="store_true", help="Save a binary edge PNG for each processed frame")
    parser.add_argument("--save-overlay-pngs", action="store_true", help="Save the original frame with edges overlaid in red")
    parser.add_argument("--save-preview-video", action="store_true", help="Save an MP4 preview with red contour overlays")
    parser.add_argument("--save-binary-pngs", action="store_true", help="Save the thresholded binary frames")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    edges_dir = outdir / "edges"
    overlay_dir = outdir / "overlay"
    binary_dir = outdir / "binary"

    if args.save_edge_pngs:
        edges_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay_pngs:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_binary_pngs:
        binary_dir.mkdir(parents=True, exist_ok=True)

    video_path = ensure_video(args, outdir)
    print(f"[info] Processing video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if args.resize_width and args.resize_width > 0:
        scale = args.resize_width / src_w
        out_w = args.resize_width
        out_h = int(round(src_h * scale))
    else:
        out_w, out_h = src_w, src_h

    preview_writer = None
    if args.save_preview_video:
        preview_path = outdir / "edge_preview.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_writer = cv2.VideoWriter(
            str(preview_path),
            fourcc,
            fps / max(args.frame_stride, 1),
            (out_w, out_h),
        )

    metadata = {
        "video_path": str(video_path),
        "source_width": src_w,
        "source_height": src_h,
        "output_width": out_w,
        "output_height": out_h,
        "fps": fps,
        "total_frames": total_frames,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "frame_stride": args.frame_stride,
        "use_otsu": args.use_otsu,
        "threshold": args.threshold,
        "blur_ksize": args.blur_ksize,
        "open_iters": args.open_iters,
        "close_iters": args.close_iters,
        "min_area": args.min_area,
        "simplify_epsilon": args.simplify_epsilon,
        "keep_holes": not args.no_holes,
        "normalize": args.normalize,
    }
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    jsonl_path = outdir / "contours.jsonl.gz"
    processed = 0
    written_frames = 0

    with gzip.open(jsonl_path, "wt", encoding="utf-8") as jf:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx < args.start_frame:
                frame_idx += 1
                continue
            if args.end_frame >= 0 and frame_idx > args.end_frame:
                break
            if (frame_idx - args.start_frame) % args.frame_stride != 0:
                frame_idx += 1
                continue

            if out_w != src_w or out_h != src_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary = compute_binary_mask(gray, args.blur_ksize, args.use_otsu, args.threshold)
            binary = clean_binary_mask(binary, args.open_iters, args.close_iters)

            contours, hierarchy = extract_contours_from_binary(
                binary,
                min_area=args.min_area,
                simplify_epsilon=args.simplify_epsilon,
                keep_holes=not args.no_holes,
            )

            edge_img = np.zeros_like(binary)
            if contours:
                cv2.drawContours(edge_img, contours, contourIdx=-1, color=255, thickness=1)

            record = {
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 6) if fps > 0 else None,
                "width": out_w,
                "height": out_h,
                "num_contours": len(contours),
                "contours": contour_payload(contours, out_w, out_h, args.normalize),
            }
            jf.write(json.dumps(record, separators=(",", ":")) + "\n")
            written_frames += 1

            if args.save_binary_pngs:
                cv2.imwrite(str(binary_dir / f"frame_{frame_idx:06d}.png"), binary)
            if args.save_edge_pngs:
                cv2.imwrite(str(edges_dir / f"frame_{frame_idx:06d}.png"), edge_img)
            if args.save_overlay_pngs:
                overlay = overlay_edges_on_frame(frame, edge_img)
                cv2.imwrite(str(overlay_dir / f"frame_{frame_idx:06d}.png"), overlay)
            if preview_writer is not None:
                overlay = overlay_edges_on_frame(frame, edge_img)
                preview_writer.write(overlay)

            processed += 1
            if processed % 100 == 0:
                print(f"[info] Processed {processed} frames...")

            frame_idx += 1

    cap.release()
    if preview_writer is not None:
        preview_writer.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nkeyboard interrupt")
        sys.exit(1)
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)
