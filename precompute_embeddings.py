import os
import cv2
import json
import torch
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import subprocess

# === Config ===
FPS = 2
STRIDE = 30
WINDOW = 10
MODEL_NAME = "openai/clip-vit-base-patch32"

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CLIP Init ===
clip_model = CLIPModel.from_pretrained(MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model.eval()


def auto_fix_video(input_path, fixed_path):
    cmd = [
        "ffmpeg", "-y", "-err_detect", "ignore_err",
        "-i", str(input_path),
        "-c:v", "libx264", "-c:a", "copy",
        str(fixed_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return fixed_path.exists()


def extract_windows(video_path: str, stride: int = STRIDE, window: int = WINDOW, fps: int = FPS):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / video_fps

    segments = []
    time_points = np.arange(0, duration - window, stride)

    for start in time_points:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        frames = []
        timestamps = []
        read_frames = int(window * fps)
        frame_interval = int(video_fps / fps)
        read_count = 0
        while read_count < read_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_interval == 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                timestamps.append(timestamp)
                read_count += 1
        if frames:
            segments.append((start, start + window, frames, timestamps))
    cap.release()
    return segments


def get_clip_embeddings(frames):
    inputs = clip_processor(images=frames, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs / outputs.norm(p=2, dim=-1, keepdim=True)


def process_video(video_dir: str, out_dir: str):
    try:
        video_dir = Path(video_dir)
        video_id = video_dir.name
        video_path = video_dir / f"{video_id}.mp4"
        out_path = Path(out_dir) / (video_id + ".npy")
        meta_path = Path(out_dir) / (video_id + ".json")
        if out_path.exists():
            logger.info(f"[SKIP] Already exists: {out_path.name}")
            return

        segments = extract_windows(video_path)
        if segments is None:
            logger.warning(f"[WARN] Failed to open {video_path.name}, trying to repair...")
            fixed_path = video_dir / f"{video_id}.fixed.mp4"
            if auto_fix_video(video_path, fixed_path):
                segments = extract_windows(fixed_path)
                if segments is None:
                    logger.error(f"[FAIL] Repair failed: {video_path.name}")
                    return
                else:
                    logger.info(f"[OK] Repaired: {video_path.name} → {fixed_path.name}")
                    video_path = fixed_path
            else:
                logger.error(f"[FAIL] Could not repair: {video_path.name}")
                return

        all_embeddings = []
        all_meta = []

        for start, end, frames, timestamps in segments:
            if not frames:
                continue
            emb = get_clip_embeddings(frames).cpu().numpy()
            mean_emb = emb.mean(axis=0)
            all_embeddings.append(mean_emb)
            all_meta.append({"start": start, "end": end, "timestamps": timestamps})

        np.save(out_path, np.stack(all_embeddings))

        with open(meta_path, "w") as f:
            json.dump({"video": video_id, "segments": all_meta}, f)

        logger.info(f"[OK] {video_path.name} → {out_path.name} ({len(all_embeddings)} segments)")
    except Exception as e:
        logger.error(f"[FAIL] {video_dir} — {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to folder with video folders")
    parser.add_argument("--output", type=str, required=True, help="Path to save .npy/.json")
    parser.add_argument("--workers", type=int, default=cpu_count() // 2)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted([d for d in input_dir.iterdir() if (d / f"{d.name}.mp4").exists()])
    logger.info(f"Found {len(video_dirs)} video(s) in {input_dir}")

    with Pool(args.workers) as pool:
        list(tqdm(pool.imap(partial(process_video, out_dir=output_dir), video_dirs), total=len(video_dirs)))


if __name__ == "__main__":
    main()