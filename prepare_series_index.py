import json
from pathlib import Path
from collections import defaultdict


def extract_series_name(name_str):
    return name_str.split(".")[0].strip()


def prepare_series_index(annotation_path, out_path):
    with open(annotation_path) as f:
        data = json.load(f)

    series_index = defaultdict(list)

    for key, value in data.items():
        video_id = key.split("_")[-1]
        series_name = extract_series_name(value["name"])
        series_index[series_name].append(video_id)

    with open(out_path, "w") as f:
        json.dump(series_index, f, indent=2, ensure_ascii=False)

    print(f" Сохранены индексы серий {out_path} ({len(series_index)} групп сериалов)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", type=str, required=True, help="Path to annotation json (train/test)")
    parser.add_argument("--out", type=str, required=True, help="Output path for series_index.json")
    args = parser.parse_args()

    prepare_series_index(args.ann, args.out)
