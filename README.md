#  Short Intro Detection in TV Series

Этот проект предназначен для **автоматического поиска коротких заставок (интро)** в эпизодах сериалов.  
Интро определяется как **повторяющийся сегмент**, чаще всего демонстрирующий **название сериала** в начале серии.

---
### Имплементация

- Видео → кадры → визуальные эмбеддинги с помощью CLIP.
- По аннотациям создаются метки `intro/not_intro` для обучения.
- Используются два подхода:
  - **MLP** (`scikit-learn`) как бейзлайн.
  - **1D-CNN** (`PyTorch`) для обработки временной зависимости между эмбеддингами.
- Предсказания агрегируются в сегменты и сохраняются в `.json`.

---

## Подход

Решение задачи построено по классическому ML/Deep Learning pipeline:

---

### 0.  Исправление "сломанных видео"
- Используется `ffmpeg` с `-err_detect ignore_err` и перекодировкой в `libx264`.
- Это позволяет обработать видео с повреждёнными NAL-блоками.
- Сохраняются в `data_train_fixed/` или `data_test_fixed/`.

Файл: `fix_videos.py`

---

### 0.1 Построение индекса сериалов

* Группирует видеофайлы по названиям сериалов, извлекая `name` из аннотаций.
* Удобно для анализа повторяемости интро.

Файл: `prepare_series_index.py`

```bash
python prepare_series_index.py --ann labels.json --out series_index.json
```

---

### 1. Извлечение эмбеддингов из видео

* Используем модель \[`openai/clip-vit-base-patch32`] для преобразования фреймов в **визуальные эмбеддинги**.
* Видео разбивается на сегменты по 10 секунд, с шагом 30 сек.
* Для каждого сегмента извлекаются N кадров (по FPS), и считается **средний эмбеддинг**.

Файл: `precompute_embeddings.py`

```bash
python precompute_embeddings.py --input data_train_fixed --output embeddings
```

---

### 2. Обучение модели

Для каждого размеченного видео создаётся обучающая выборка `(эмбеддинг, метка 0/1)`:

* Метка `1`, если сегмент пересекается с интро (`IoU > 0.3`);
* Метка `0` — если нет;
* Также считаются `sample weights` для борьбы с дисбалансом классов.

####  MLP (baseline)
Файл: `train_mlp.py`

####  1D-CNN (усиленная модель)
Файл: `train_1dcnn.py`

---

### 3. Предсказания на тестовом наборе

После обучения модель можно применить к эмбеддингам из `test`:

Результаты сохраняются в файл `predicted_intro_segments_test.json` в формате:

```json
{
  "-220020068_456241758": [
    { "start": 60.0, "end": 70.0 }
  ],
  "-220020068_456249732": [
    { "start": 0.0, "end": 10.0 }
  ]
}
```
