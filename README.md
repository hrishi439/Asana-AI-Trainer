

# Asana AI Trainer

A lightweight, production-ready toolkit to **train, evaluate, and deploy models that recognize and coach yoga asanas** from images or video. Comes with CLI tools, reproducible configs, and optional real-time inference using a webcam.

---

## 🚀 Quick Start

###  Clone the repo

```bash
git clone https://github.com/hrishi439/Asana-AI-Trainer/tree/main
cd <https://github.com/hrishi439/Asana-AI-Trainer/edit/main/README.md>
```


## 📦 Requirements

* Python ≥ 3.9
* pip ≥ 21
* CUDA 11+ (optional, for GPU)
* Virtual environment recommended

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -U pip
pip install -r requirements.txt
```

---

## 🧰 Project Structure

```
asana-ai-trainer/
├─ asana_ai/        # package code
│  ├─ data/         # dataset loaders
│  ├─ models/       # model architectures
│  ├─ training/     # train/val loops
│  ├─ inference/    # realtime/video inference
│  └─ utils/        # helpers, logging
├─ configs/         # YAML configs
├─ scripts/         # CLI entrypoints
├─ notebooks/       # experiments
├─ tests/           # unit tests
├─ requirements.txt
└─ README.md
```

---

## 📚 Datasets

* Place raw data under `data/` (customizable).
* Use a manifest CSV/JSON with paths + labels.
* Pose-keypoint datasets (COCO, MediaPipe, etc.) are supported.

---

## 🏋️ Training

```bash
python scripts/train.py --config configs/base.yaml
```

Common flags:

* `--epochs 50`
* `--batch-size 32`
* `--device cuda`
* `--output runs/exp1`

---

## ✅ Evaluation

```bash
python scripts/evaluate.py --checkpoint runs/exp1/best.ckpt --split val
```

Outputs accuracy, F1, confusion matrix, and per-class metrics.

---

## 🎥 Inference

* **Webcam (real-time):**

  ```bash
  python scripts/infer.py --source webcam --checkpoint runs/exp1/best.ckpt
  ```

* **Video file:**

  ```bash
  python scripts/infer.py --source path/to/video.mp4 --checkpoint runs/exp1/best.ckpt
  ```

Options:

* `--pose-estimator mediapipe|openpose`
* `--overlay true`
* `--fps 30`

---

## ⚙️ Configuration

Example `configs/base.yaml`:

```yaml
seed: 42
device: auto
data:
  root: data/
  train_manifest: data/train.csv
  val_manifest: data/val.csv
  num_classes: 20
model:
  name: resnet18
  pretrained: true
train:
  epochs: 50
  batch_size: 32
  lr: 3e-4
log:
  dir: runs/exp1
  save_best: true
```

---

## 🧪 Testing

```bash
pytest -q
```

Covers data loaders, models, and training logic.

---

## 📦 Packaging

* Build wheel:

  ```bash
  python -m build
  ```
* Publish:

  ```bash
  python -m twine upload dist/*
  ```

Attach ZIP or wheel to GitHub Releases for easy download.

---

## 🗺️ Roadmap

* [ ] Add more asana classes
* [ ] Export to ONNX/TFLite
* [ ] Web demo
* [ ] Pose-quality scoring

---

## 🤝 Contributing

1. Fork
2. Create branch
3. Commit + push
4. Open PR

---



## 🔗 Direct Download Links

Main branch:
  `https://github.com/hrishi439/Asana-AI-Trainer/edit/main`

