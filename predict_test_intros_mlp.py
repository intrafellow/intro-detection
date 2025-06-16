import json
import numpy as np
from pathlib import Path
import joblib

# === Configuration ===
MODEL_PATH = Path("/Users/idg0d/PycharmProjects/vk/mlp_model.pkl")
TEST_EMBEDDING_DIR = Path("/Users/idg0d/PycharmProjects/vk/embeddings_test")
OUTPUT_PREDICTIONS_PATH = Path("/Users/idg0d/PycharmProjects/vk/test_predicted_intro_segments.json")


mlp = joblib.load(MODEL_PATH)


results = {}
threshold = 0.5
# === Finding intros ===

for emb_path in TEST_EMBEDDING_DIR.glob("*.npy"):
    video_id = emb_path.stem
    meta_path = TEST_EMBEDDING_DIR / f"{video_id}.json"
    if not meta_path.exists():
        continue

    emb = np.load(emb_path)
    with open(meta_path) as f:
        meta = json.load(f)["segments"]

    probs = mlp.predict_proba(emb)[:, 1]
    segments = [(s["start"], s["end"]) for s, p in zip(meta, probs) if p >= threshold]

    if not segments:
        continue

    merged = []
    current_start, current_end = segments[0]
    for start, end in segments[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))

    results[video_id] = [{"start": round(s, 2), "end": round(e, 2)} for s, e in merged]

with open(OUTPUT_PREDICTIONS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n Готово! Предсказания для тестов сохранены в {OUTPUT_PREDICTIONS_PATH}")
