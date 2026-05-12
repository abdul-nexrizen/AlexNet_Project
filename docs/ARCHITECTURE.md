# Architecture notes

This document explains the design decisions behind both models and the
adaptations made for CIFAR-10. It is the companion to the slide deck and
report — those summarise the result, this explains the *why*.

---

## 1. LeNet-5 (LeCun et al., 1998)

LeNet-5 is the canonical convolutional neural network. Its conv-pool-conv-pool-FC-FC blueprint defined the field until AlexNet came along fourteen years later.

### 1.1 Original architecture

| Layer | Operation                              | Output       |
| ----- | -------------------------------------- | ------------ |
| C1    | conv 5×5, 6 filters, **tanh**          | 28×28×6      |
| S2    | **average** pool 2×2, stride 2         | 14×14×6      |
| C3    | conv 5×5, 16 filters, tanh             | 10×10×16     |
| S4    | average pool 2×2, stride 2             | 5×5×16       |
| C5    | fully connected, 120 units, tanh       | 120          |
| F6    | fully connected, 84 units, tanh        | 84           |
| Out   | fully connected, 10 units (softmax)    | 10           |

### 1.2 Why each choice was made

- **tanh activation** — ReLU did not exist yet as a default; tanh saturates
  for large magnitudes but was the standard differentiable non-linearity
  of the time.
- **Average pooling** — LeCun argued for spatial averaging because it
  reduces variance more smoothly than max pooling; this also predated the
  empirical observation that max pooling tends to work better for
  classification.
- **Fully connected C5** — the paper implements C5 as a 5×5 convolution
  that collapses the 5×5×16 feature map to 1×1×120. Mathematically that is
  identical to a fully connected layer when the feature map is exactly
  5×5, so we use `nn.Linear` for readability.

### 1.3 Adaptations for CIFAR-10

The only change is the number of input channels (1 → 3). Spatial sizes
(32 → 28 → 14 → 10 → 5) and the rest of the architecture are unchanged,
which keeps the comparison faithful to the original.

---

## 2. AlexNet (Krizhevsky et al., 2012)

AlexNet won ImageNet 2012 by a margin of more than 10 absolute percentage
points and effectively restarted deep-learning research. The architecture
itself is straightforward; what made it work was the combination of five
techniques.

### 2.1 Original architecture (224×224 ImageNet)

| Layer  | Operation                                     | Output       |
| ------ | --------------------------------------------- | ------------ |
| Conv1  | 11×11×96, stride 4, ReLU + LRN + MaxPool      | 55×55×96     |
| Conv2  | 5×5×256, pad 2, ReLU + LRN + MaxPool          | 27×27×256    |
| Conv3  | 3×3×384, pad 1, ReLU                          | 13×13×384    |
| Conv4  | 3×3×384, pad 1, ReLU                          | 13×13×384    |
| Conv5  | 3×3×256, pad 1, ReLU + MaxPool                | 13×13×256    |
| FC6    | 4096 units, ReLU + Dropout                    | 4096         |
| FC7    | 4096 units, ReLU + Dropout                    | 4096         |
| FC8    | 1000 units (softmax)                          | 1000         |

### 2.2 The five key innovations

1. **ReLU activations** — `f(x) = max(0, x)` does not saturate in the
   positive regime, so gradients flow through all eight layers. Krizhevsky
   et al. report ~6× faster training than tanh.
2. **Dropout** — randomly zeroes out half of each FC unit's activations
   during training. Forces the network to learn redundant, ensemble-like
   features.
3. **Data augmentation** — random crops, horizontal flips, and PCA-based
   colour jitter expand the effective dataset by more than 2,000×.
4. **GPU training** — two NVIDIA GTX 580 cards, with model parallelism
   between them. First time a deep CNN was trained on GPUs.
5. **Overlapping max pooling + LRN** — 3×3 max pooling with stride 2
   means the pooling windows overlap, which slightly improves accuracy.
   Local Response Normalization (LRN) implements lateral inhibition
   between feature maps (now superseded by BatchNorm).

### 2.3 Adaptations for CIFAR-10

CIFAR-10 images are 32×32, eight times smaller per side than the
224×224 ImageNet images AlexNet was designed for. The first convolution's
11×11 kernel with stride 4 would collapse the feature map to nothing, so
we replace it with a 3×3 kernel of stride 1, padding 1. All subsequent
channel counts, kernel sizes, dropout rates, and FC widths are preserved.

---

## 3. Side-by-side summary

| Aspect             | LeNet-5 (1998)                | AlexNet (2012)                                |
| ------------------ | ----------------------------- | --------------------------------------------- |
| Depth              | 5 learnable layers            | **8 learnable layers**                        |
| Parameters         | ~60,000                       | ~60,000,000                                   |
| Activation         | tanh                          | **ReLU** (~6× faster)                         |
| Pooling            | average, non-overlapping      | **max, overlapping**                          |
| Regularization     | none                          | **Dropout** + data augmentation               |
| Normalisation      | none                          | **Local Response Normalization** (LRN)        |
| Hardware           | CPU                           | **two GPUs in parallel**                      |
| Dataset            | MNIST (~70k digits)           | ImageNet (~1.2M images, 1000 classes)         |

The improvement from one to the other is not attributable to any single
design choice — capacity, optimisation, and regularisation each contribute
roughly a third of the gap. See the report for the ablation discussion.
