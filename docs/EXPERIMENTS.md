# Reproducing the experiments

Everything happens in **one notebook**:
[`notebook/AlexNet_LeNet5_CIFAR10.ipynb`](../notebook/AlexNet_LeNet5_CIFAR10.ipynb).
The notebook already contains the executed outputs of a real 20-epoch
run on an Apple M4 Pro (MPS), so you can read the full experiment
without running anything. The instructions below cover re-running it.

## 0. Setup (only needed for re-running)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter
```

GPU is **strongly recommended**. On a Colab T4 the full run takes about 30
minutes; on an Apple M4 Pro with MPS it takes about 53 minutes (LeNet-5
~19 min + AlexNet ~34 min); CPU is much slower.

## 1. Run the notebook

### Option A — Google Colab

Open `notebook/AlexNet_LeNet5_CIFAR10.ipynb` in Colab, set the runtime to
**T4 GPU** under *Runtime → Change runtime type*, then *Runtime → Run all*.
The notebook downloads CIFAR-10, builds both networks, trains each for
20 epochs, and produces all the comparison plots and tables.

### Option B — Local

```bash
jupyter notebook notebook/AlexNet_LeNet5_CIFAR10.ipynb
# then Cell -> Run All in the Jupyter UI
```

The notebook auto-detects the best available device:
CUDA → Apple Silicon MPS → CPU.

## 2. Measured results (Apple M4 Pro, MPS)

These numbers are from the run whose outputs are already embedded in the
notebook (seed 42, PyTorch 2.12.0).

| Metric                              | LeNet-5             | AlexNet             |
| ----------------------------------- | ------------------- | ------------------- |
| Trainable parameters                | 62,006              | 36,051,786          |
| Learnable layers                    | 5                   | 8                   |
| Final test accuracy (20 epochs)     | 59.57%              | 85.67%              |
| Best test accuracy (20 epochs)      | 59.57%              | 85.67%              |
| Mean epoch time (MPS)               | 57.0 s              | 102.1 s             |
| Total wall time (20 epochs)         | 19.0 min            | 34.0 min            |

Accuracy gap (AlexNet - LeNet-5): **+26.10 percentage points**. AlexNet
wins on every CIFAR-10 class; the full per-class confusion matrix is
embedded in the notebook.

## 3. Hyperparameters

| Hyperparameter   | Value            |
| ---------------- | ---------------- |
| Optimiser        | SGD              |
| Momentum         | 0.9              |
| Weight decay     | 5e-4             |
| LR schedule      | Cosine annealing |
| Initial LR       | 0.01             |
| Batch size       | 128              |
| Epochs           | 20               |
| Loss             | Cross-entropy    |
| Augmentation     | RandomCrop(32, pad=4) + HorizontalFlip |
| Normalisation    | Per-channel (CIFAR-10 train stats)     |
| Seed             | 42               |

All of these live inside the notebook itself — change the constants in
the `Setup` / `Training utilities` cells if you want to override them.

## 4. Troubleshooting

| Symptom                                                       | Fix |
| ------------------------------------------------------------- | --- |
| `ModuleNotFoundError: No module named 'torch'`                | `pip install -r requirements.txt` |
| `CUDA out of memory` on AlexNet                               | Lower the `BATCH` constant to 64 or 32 |
| Training stalls in a Jupyter notebook on Windows              | Set `num_workers=0` in the DataLoader cell |
| First epoch much slower than later epochs                     | Expected — CIFAR-10 is downloading and the cuDNN auto-tuner is warming up. |
