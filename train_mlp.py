import os
import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# # === Configuration ===
EMBEDDING_DIR = Path("/Users/idg0d/PycharmProjects/vk/embeddings")  # Папка с .npy/.json
LABELS_JSON_PATH = Path("/Users/idg0d/Downloads/data_train_short/labels.json")  # JSON с разметкой интро
MODEL_PATH = Path("/Users/idg0d/PycharmProjects/vk/mlp_model.pkl")  # Сохраняем модель сюда
PREDICTIONS_PATH = Path("/Users/idg0d/PycharmProjects/vk/train_predicted_intro_segments.json")  # JSON с результатами

def timestamp_to_seconds(ts: str) -> float:
    parts = [int(p) for p in ts.strip().split(":")]
    return parts[0]*3600 + parts[1]*60 + parts[2] if len(parts) == 3 else parts[0]*60 + parts[1]


with open(LABELS_JSON_PATH) as f:
    intro_times = json.load(f)

X, y = [], []
video_segments = {}


for video_id, info in intro_times.items():
    emb_path = EMBEDDING_DIR / f"{video_id}.npy"
    meta_path = EMBEDDING_DIR / f"{video_id}.json"
    if not emb_path.exists() or not meta_path.exists():
        continue

    emb = np.load(emb_path)
    with open(meta_path) as f:
        meta = json.load(f)["segments"]
    video_segments[video_id] = meta

    intro_start = timestamp_to_seconds(info["start"])
    intro_end = timestamp_to_seconds(info["end"])

    for segment, emb_vec in zip(meta, emb):
        seg_start = segment["start"]
        seg_end = segment["end"]

        inter_start = max(seg_start, intro_start)
        inter_end = min(seg_end, intro_end)
        intersection = max(0, inter_end - inter_start)
        union = max(seg_end, intro_end) - min(seg_start, intro_start)
        iou = intersection / union if union > 0 else 0

        label = 1 if iou > 0.3 else 0
        X.append(emb_vec)
        y.append(label)

X = np.array(X)
y = np.array(y)

counter = Counter(y)
total = len(y)
class_weights = {cls: total / count for cls, count in counter.items()}
sample_weights = np.array([class_weights[label] for label in y])

# === Model ===
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
mlp.fit(X, y, sample_weight=sample_weights)
joblib.dump(mlp, MODEL_PATH)

y_pred = mlp.predict(X)
print("\n=== MLP Classifier ===")
print(classification_report(y, y_pred, digits=3))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d',
            cmap='Greens', xticklabels=["not_intro", "intro"],
            yticklabels=["not_intro", "intro"])
ax.set_title("MLP Classifier (train)")
plt.tight_layout()
plt.show()

results = {}
threshold = 0.5

for video_id, meta in video_segments.items():
    emb = np.load(EMBEDDING_DIR / f"{video_id}.npy")
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

# --- Сохраняем в JSON
with open(PREDICTIONS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nГотово! Предсказания сохранены в {PREDICTIONS_PATH}")
