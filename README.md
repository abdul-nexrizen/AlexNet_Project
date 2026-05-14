# AlexNet vs LeNet-5 on CIFAR-10

**Deep Learning for Computer Vision**

## What's in here

A single self-contained Jupyter notebook that reimplements **LeNet-5**
(LeCun et al., 1998) and **AlexNet** (Krizhevsky, Sutskever, and Hinton,
2012) in PyTorch, trains both on CIFAR-10 under identical conditions, and
analyses the results. The notebook **already contains all the outputs from
a real 20-epoch run on an Apple M4 Pro** (training logs, learning curves,
confusion matrix, learned first-layer filters, summary table) so you can
read the full experiment without having to run anything.

```
AlexNet_Project/
├── README.md
├── requirements.txt
├── notebook/
│   └── AlexNet_LeNet5_CIFAR10.ipynb   <- THE notebook (read or re-run)
├── docs/
│   ├── ARCHITECTURE.md                 <- architecture notes
│   └── EXPERIMENTS.md                  <- short reproduction guide
├── figures/                            <- diagrams + measured plots
├── presentation/                       <- slide deck
└── report/                             <- written report (.docx)
```

There are no separate training scripts or `src/` package: the notebook is
the only entry point and the only place the model code lives.

## Measured results (20 epochs on CIFAR-10, Apple M4 Pro / MPS)

These numbers come from the executed run that is **already embedded as
outputs in [`notebook/AlexNet_LeNet5_CIFAR10.ipynb`](notebook/AlexNet_LeNet5_CIFAR10.ipynb)**.
Open the notebook to see the per-epoch logs, the side-by-side learning
curves, the AlexNet confusion matrix, and the learned first-layer filters
inline.

| Metric                              | LeNet-5             | AlexNet             |
| ----------------------------------- | ------------------- | ------------------- |
| Trainable parameters                | 62,006              | 36,051,786          |
| Learnable layers                    | 5                   | 8                   |
| Final test accuracy (20 epochs)     | 59.57%              | 85.67%              |
| Best test accuracy (20 epochs)      | 59.57%              | 85.67%              |
| Mean epoch time (MPS)               | 57.0 s              | 102.1 s             |
| Total wall time (20 epochs)         | 19.0 min            | 34.0 min            |

AlexNet beats LeNet-5 by **+26.10 percentage points** on the test set,
and improves over LeNet-5 on every one of the 10 CIFAR-10 classes
(largest gains: deer +40.4pp, cat +32.5pp, bird +32.4pp).

## Re-running the notebook

If you want to reproduce the results instead of just reading them:

### Option 1 — Google Colab (recommended)

1. Open `notebook/AlexNet_LeNet5_CIFAR10.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**.

Expected total runtime on a Colab T4 GPU: ~30 minutes.

### Option 2 — Local Python

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter
jupyter notebook notebook/AlexNet_LeNet5_CIFAR10.ipynb
```

The notebook auto-detects the best available device (CUDA → Apple Silicon
MPS → CPU) so the same cells work unchanged on Colab, an NVIDIA GPU, an
Apple Silicon Mac, or plain CPU.

## Key innovations of AlexNet (vs LeNet-5)

1. **ReLU** activations - ~6x faster training than tanh, no vanishing gradients.
2. **Dropout** in the FC layers - regularizes the huge classifier head.
3. **Aggressive data augmentation** - random crops, horizontal flips.
4. **GPU training** - two GTX 580s in parallel (the historical first).
5. **Overlapping max pooling + LRN** - improvements over LeNet's average pooling.

## References

1. Krizhevsky, A., Sutskever, I., Hinton, G. E. (2012). *ImageNet
   Classification with Deep Convolutional Neural Networks.* NeurIPS.
2. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). *Gradient-Based
   Learning Applied to Document Recognition.* Proc. IEEE.
3. Hinton, G. E. et al. (2012). *Improving neural networks by preventing
   co-adaptation of feature detectors.* arXiv:1207.0580.
4. PyTorch documentation - <https://pytorch.org/docs/>
5. Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning.* MIT Press. Available at: [MIT Press Deep Learning Book](https://mitpress.mit.edu/9780262035613/deep-learning/)
