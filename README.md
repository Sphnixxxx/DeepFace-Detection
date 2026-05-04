# 🎭 DeepFace Detection — Real vs Fake Faces (StyleGAN3)

Binary image classifier to detect AI-generated faces using the **10,000 Real vs Fake Faces** dataset powered by NVIDIA StyleGAN3.

---

## 📁 Project Structure

```
FACE/
├── data/
│   ├── Real faces/          # 10,000 FFHQ real face images
│   ├── Fake faces/          # 10,000 StyleGAN3 generated images
│   └── metadata.csv         # Labels and file paths
└── notebooks/
    ├── deepface_detection.ipynb   # TensorFlow / Keras implementation
    └── deepface_pytorch.ipynb      # PyTorch implementation
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Total Images | 20,000 (balanced) |
| Real Faces | 10,000 — sourced from [FFHQ](https://github.com/NVlabs/ffhq-dataset) |
| Fake Faces | 10,000 — generated with NVIDIA StyleGAN3 (truncation ψ = 0.8) |
| Image Size | 256 × 256 px, RGB |
| License | CC BY-NC-SA 4.0 (non-commercial research only) |

---

## 🧠 Implementations

### TensorFlow (`deepface_detection.ipynb`)
- Custom CNN: 3× Conv-BN-Pool → GAP → Dense → Dropout
- `ImageDataGenerator` with augmentation
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- Optional transfer learning: **MobileNetV2**

### PyTorch (`deepface_pytorch.ipynb`)
- Same CNN architecture via `nn.Module`
- Stratified train/val split with `torchvision.transforms`
- Manual training loop with early stopping & LR scheduler
- Optional transfer learning: **EfficientNet-B0**

Both notebooks include: training curves, confusion matrix, classification report, misclassified examples viewer, and single-image inference.

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install tensorflow torch torchvision opencv-python scikit-learn matplotlib seaborn pandas

# Launch a notebook
jupyter notebook notebooks/deepface_detection.ipynb
```

---

## 📈 Results

| Framework | Model | Val Accuracy |
|---|---|---|
| TensorFlow | Custom CNN | 0.64 |
| PyTorch | Custom CNN | 0.66 |

> results after training.

---

## 📜 License

Dataset is provided under **CC BY-NC-SA 4.0** — for educational and non-commercial research purposes only.

- Real images: [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) by NVIDIA
- Fake images: Generated using [StyleGAN3](https://github.com/NVlabs/stylegan3) by NVIDIA
