import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from pathlib import Path

# --- Пути
EMBEDDING_DIR = Path("/Users/idg0d/PycharmProjects/vk/embeddings")
LABELS_JSON_PATH = Path("/Users/idg0d/Downloads/data_train_short/labels.json")
MODEL_PATH = Path("intro_1dcnn.pt")
PREDICTION_JSON_PATH = Path("intro_1dcnn_predictions.json")

# --- Параметры
WINDOW_SIZE = 5
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
THRESHOLD = 0.3

def timestamp_to_seconds(ts: str) -> float:
    parts = [int(p) for p in ts.strip().split(":")]
    return parts[0]*3600 + parts[1]*60 + parts[2] if len(parts) == 3 else parts[0]*60 + parts[1]

with open(LABELS_JSON_PATH) as f:
    intro_times = json.load(f)

X_raw, y_raw = [], []

for video_id, info in intro_times.items():
    emb_path = EMBEDDING_DIR / f"{video_id}.npy"
    meta_path = EMBEDDING_DIR / f"{video_id}.json"
    if not emb_path.exists() or not meta_path.exists():
        continue

    emb = np.load(emb_path)
    with open(meta_path) as f:
        meta = json.load(f)["segments"]

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
        X_raw.append(emb_vec)
        y_raw.append(label)

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)

X_seq = np.stack([X_raw[i:i + WINDOW_SIZE] for i in range(len(X_raw) - WINDOW_SIZE + 1)])
y_seq = y_raw[WINDOW_SIZE // 2: -(WINDOW_SIZE // 2) or None]

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)

pos_weight = torch.tensor([len(y_raw) / max(sum(y_raw), 1)])

train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

class Intro1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(512, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

model = Intro1DCNN()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(xb), yb).item() for xb, yb in val_loader)
        val_losses.append(val_loss / len(val_loader))

torch.save(model.state_dict(), MODEL_PATH)

plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.title("Loss")
plt.legend()
plt.show()

# --- Предсказания
model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in DataLoader(dataset, batch_size=1):
        logits = model(xb[0].unsqueeze(0))
        prob = torch.sigmoid(logits).item()
        all_preds.append(prob)

print("🔍 Пример предсказаний:", all_preds[:20])
print("🔍 Максимум:", max(all_preds))

segments = []
active = False
start = None

for i, p in enumerate(all_preds):
    if p >= THRESHOLD and not active:
        start = i
        active = True
    elif p < THRESHOLD and active:
        segments.append({"start": round(start, 2), "end": round(i, 2)})
        active = False

if active:
    segments.append({"start": round(start, 2), "end": round(len(all_preds), 2)})

with open(PREDICTION_JSON_PATH, "w") as f:
    json.dump({"segments": segments}, f, indent=2)

print(f"\nГотово!  Модель сохранена в {MODEL_PATH}")
print(f"Готово! Предсказания сохранены в {PREDICTION_JSON_PATH}")
