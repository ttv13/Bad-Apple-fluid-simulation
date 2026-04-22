#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def find_downloaded_video(outdir: Path):
    for ext in [".mp4", ".mkv", ".webm", ".mov"]:
        files = sorted(outdir.glob(f"bad_apple*{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return files[0]
    return None


def download_video(url: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)

    existing = find_downloaded_video(outdir)
    if existing is not None:
        return existing

    try:
        import yt_dlp

        opts = {
            "noplaylist": True,
            "format": "bv*[ext=mp4]/b[ext=mp4]/bv*/b",
            "outtmpl": str(outdir / "bad_apple.%(ext)s"),
            "quiet": False,
        }

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)

            requested = info.get("requested_downloads") or []
            for item in requested:
                fp = item.get("filepath")
                if fp and os.path.exists(fp):
                    return Path(fp)

            prepared = ydl.prepare_filename(info)
            if os.path.exists(prepared):
                return Path(prepared)

    except Exception as e:
        print(f"[warn] yt_dlp import path failed: {e}")

    yt_dlp_bin = shutil.which("yt-dlp")
    if yt_dlp_bin is None:
        raise RuntimeError(
            "yt-dlp not found. Install it with:\n"
            "  py -m pip install yt-dlp\n"
            "or download the video yourself and use --input."
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

    video = find_downloaded_video(outdir)
    if video is None:
        raise RuntimeError("Download finished, but no video file was found.")
    return video


def get_video(args, outdir: Path) -> Path:
    if args.input:
        video = Path(args.input)
        if not video.exists():
            raise FileNotFoundError(f"Input video not found: {video}")
        return video

    if args.url:
        return download_video(args.url, outdir)

    raise ValueError("Provide either --input <video> or --url <youtube_url>")


def threshold_frame(gray: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def extract_contours(binary: np.ndarray):
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    kept = []
    for cnt in contours:
        area = abs(cv2.contourArea(cnt))
        if area < 8:
            continue
        cnt = cv2.approxPolyDP(cnt, 1.0, True)
        if len(cnt) >= 2:
            kept.append(cnt)
    return kept


def contours_to_list(contours):
    output = []
    for cnt in contours:
        pts = cnt[:, 0, :].astype(int)
        output.append(pts.tolist())
    return output


def overlay_edges(frame: np.ndarray, edge_img: np.ndarray) -> np.ndarray:
    out = frame.copy()
    out[edge_img > 0] = (0, 0, 255)
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract per-frame black/white boundary contours from a video.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Path to local video file")
    src.add_argument("--url", type=str, help="YouTube URL")

    parser.add_argument("--outdir", type=str, default="bad_apple_edges", help="Output directory")
    parser.add_argument("--resize-width", type=int, default=320, help="Resize frames to this width")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    edges_dir = outdir / "edges"
    edges_dir.mkdir(parents=True, exist_ok=True)

    video_path = get_video(args, outdir)
    print(f"[info] Using video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if args.resize_width > 0 and src_w > 0:
        scale = args.resize_width / src_w
        out_w = args.resize_width
        out_h = int(round(src_h * scale))
    else:
        out_w, out_h = src_w, src_h

    preview_path = outdir / "edge_preview.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    preview = cv2.VideoWriter(str(preview_path), fourcc, fps, (out_w, out_h))

    metadata = {
        "video_path": str(video_path),
        "fps": fps,
        "source_width": src_w,
        "source_height": src_h,
        "output_width": out_w,
        "output_height": out_h,
    }
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    jsonl_path = outdir / "contours.jsonl.gz"

    frame_idx = 0
    with gzip.open(jsonl_path, "wt", encoding="utf-8") as jf:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if out_w != src_w or out_h != src_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary = threshold_frame(gray)
            contours = extract_contours(binary)

            edge_img = np.zeros_like(binary)
            if contours:
                cv2.drawContours(edge_img, contours, -1, 255, 1)

            record = {
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 6),
                "width": out_w,
                "height": out_h,
                "num_contours": len(contours),
                "contours": contours_to_list(contours),
            }
            jf.write(json.dumps(record, separators=(",", ":")) + "\n")

            cv2.imwrite(str(edges_dir / f"frame_{frame_idx:06d}.png"), edge_img)
            preview.write(overlay_edges(frame, edge_img))

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"[info] Processed {frame_idx} frames...")

    cap.release()
    preview.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] Interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)
