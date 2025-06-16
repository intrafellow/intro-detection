import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

# === Configuration ===
EMBEDDING_DIR = Path("/Users/idg0d/PycharmProjects/vk/embeddings_test")  # Path to .npy and .json embeddings
MODEL_PATH = Path("/Users/idg0d/PycharmProjects/vk/intro_1dcnn.pt")      # Trained model checkpoint
OUTPUT_PATH = Path("/Users/idg0d/PycharmProjects/vk/test_predicted_segments_1dcnn.json")  # Output predictions
WINDOW_SIZE = 5
BATCH_SIZE = 32

# === Model ===


class Intro1DCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(512, 128, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveMaxPool1d(1)
        self.fc = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 512, T)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

# === Load model ===


model = Intro1DCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# === Finding intros ===
results = {}

for emb_path in tqdm(sorted(EMBEDDING_DIR.glob("*.npy"))):
    video_id = emb_path.stem
    meta_path = EMBEDDING_DIR / f"{video_id}.json"
    if not meta_path.exists():
        continue

    emb = np.load(emb_path)
    with open(meta_path) as f:
        meta = json.load(f)["segments"]

    if len(emb) < WINDOW_SIZE:
        continue

    X_seq = np.stack([emb[i:i + WINDOW_SIZE] for i in range(len(emb) - WINDOW_SIZE + 1)])
    timestamps = meta[WINDOW_SIZE // 2: -(WINDOW_SIZE // 2) or None]

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=BATCH_SIZE)

    probs = []
    with torch.no_grad():
        for xb in loader:
            logits = model(xb[0])
            prob = torch.sigmoid(logits)
            probs.extend(prob.tolist())

    if not probs:
        continue

    top_idx = int(np.argmax(probs))
    top_seg = timestamps[top_idx]
    results[video_id] = [{
        "start": round(top_seg["start"], 2),
        "end": round(top_seg["end"], 2)
    }]

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n Готово! Предсказания для тестов сохранены в {OUTPUT_PATH}")
