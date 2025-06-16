import os
import subprocess
from pathlib import Path

INPUT_ROOT = Path("/Users/idg0d/Downloads/data_test_short")
OUTPUT_ROOT = Path("data_test_fixed")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def fix_video(input_path: Path, output_path: Path):
    print(f"Fixing: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-err_detect", "ignore_err",
        "-i", str(input_path),
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-c:a", "copy",
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f" Сохранено: {output_path}")
    except subprocess.CalledProcessError:
        print(f" Ошибка в исправлении: {input_path}")


for folder in INPUT_ROOT.iterdir():
    if folder.is_dir():
        mp4_path = folder / f"{folder.name}.mp4"
        if mp4_path.exists():
            out_path = OUTPUT_ROOT / folder.name / f"{folder.name}.mp4"
            fix_video(mp4_path, out_path)
