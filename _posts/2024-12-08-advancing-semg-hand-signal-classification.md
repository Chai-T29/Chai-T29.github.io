---

layout: post
title: "Advancing sEMG Hand Signal Classification"
description: "A comprehensive analysis of sEMG-based gesture recognition with windowed and non-windowed approaches."
date: 2024-12-08
feature\_image: images/emg.jpg
----------------------------------------------

Surface Electromyography (sEMG) records the electrical activity of muscles and has become a cornerstone in prosthetic control and human–machine interfaces. In this report, we investigate how different preprocessing strategies — windowing with FFT-based feature extraction and non-windowed architectures — impact classification accuracy and inference speed using modern hardware accelerators.<br> <br>

<!--more--><br>

The ability to classify hand gestures accurately and quickly is essential for real-time applications like prosthetic limbs. By comparing traditional windowed methods against direct feature modeling, we aim to identify which approach offers the best trade-off between performance and complexity.<br> <br>

## Contents<br>

1. [Problem Statement](#problem-statement)<br>
2. [Data Source](#data-source)<br>
3. [Methodology](#methodology)<br>

   * [Windowing Approach](#windowing-approach)<br>
   * [Non-Windowing Approach](#non-windowing-approach)<br>
   * [Loss & Optimization](#loss--optimization)<br>
4. [Evaluation Technique](#evaluation-technique)<br>
5. [Results](#results)<br>
6. [Conclusion & Future Work](#conclusion--future-work)<br>
7. [References](#references)<br>

   <br>

## Problem Statement<br>

Surface Electromyography measures the electricity that muscles release upon contraction. While windowing segments signals into fixed intervals for robust feature extraction, it introduces computational overhead and fixed temporal boundaries. Alternatively, non-windowed methods model each timestep directly, potentially reducing latency. This study evaluates both approaches to determine their impact on classification accuracy and inference time.<br> <br>

## Data Source<br>

We use the “EMG Data for Gestures” dataset from the UCI Machine Learning Repository, captured via a MYO Thalmic bracelet over eight channels for 36 subjects. Each recording includes a timestamp $t_i$, an 8-dimensional sEMG vector $x_i$, and a class label $y_i$ corresponding to one of eight static gestures.<br> <br>

## Methodology<br>

Below, we detail the mathematical formulations underpinning each pipeline.<br> <br>

### Windowing Approach<br>

#### a. Segmentation into Windows<br>

Let the raw dataset be $\{t_i, x_i, y_i\}_{i=1}^n$, where $x_i \in \mathbb{R}^M$, $y_i \in \{c_1,\dots,c_C\}$. Applying a sliding window of length $W$ produces:<br>

$$$
X^{(j)} = [\,x_{i}, x_{i+1}, \dots, x_{i+W-1}\], \quad Y^{(j)} = [\,y_{i}, y_{i+1}, \dots, y_{i+W-1}\],
$$<br>

for \(j=1,\dots,N\), where \(N\) is the number of windows.<br>

<!-- Add Figure: Sliding Window Pattern -->
<br>
#### b. Frequency-Domain Transformation (FFT)<br>

For each channel \(m\in\{1,\dots,M\}\), the windowed signal \(x^{(j)}_m\) undergoes a DFT via FFT:<br>

$$$

X^{(j)}*m(k) ,=, \sum*{n=0}^{W-1} x\_{m}(n),e^{-2\pi i kn/W}, \quad k=0,\dots,\lfloor W/2\rfloor.

$$$<br>

We store magnitude and phase:<br>

$$|X^{(j)}_m(k)| = \sqrt{\operatorname{Re}(X_m)^2 + \operatorname{Im}(X_m)^2}, \quad \angle X^{(j)}_m(k)=\arctan\frac{\operatorname{Im}(X_m)}{\operatorname{Re}(X_m)}.
$$<br>

The feature vector \(H^{(j)}\in\mathbb{R}^{2M(K+1)}\) is formed by stacking channels.<br>

<!-- Add Figure: FFT Magnitude & Phase -->
<br>
#### c. CNN for Frequency-Domain Feature Extraction<br>

Treating frequency bins as spatial dimension and channels as input channels, a 1D convolution layer with filter \(W\in\mathbb{R}^{F\times C_{in}\times C_{out}}\) and bias \(b\) produces:<br>

$$$

h\_l(k,d)=f\Bigl(\sum\_{c=1}^{C\_{in}}\sum\_{\delta=0}^{F-1} H(k+\delta, c),W\_{d,\delta,c}+b\_d\Bigr),

$$$<br>

followed by LeakyReLU:<br>

$$\mathrm{LeakyReLU}(x)=\max(0,x)+\alpha\min(0,x).$$<br>

BatchNorm, MaxPool, and Dropout complete the block.<br>

<!-- Add Figure: CNN Architecture -->
<br>
#### d. Deep Cross Network (DCN)<br>

After CNN, embedding \(e\in\mathbb{R}^D\) is passed through \(L\) cross layers:<br>

$$x^{(l+1)} = x^{(l)} \circ (x^{(l)}W^{(l)}) + b^{(l)} + x^{(l)}, \quad l=0,...,L-1,$$<br>

where \(\circ\) is Hadamard product.<br>

<!-- Add Figure: DCN V2 Architecture -->
<br>
#### e. Multi-Layer Perceptron (MLP)<br>

Final features \(x^{(L)}\) go through fully‑connected layers with ReLU, LayerNorm, and Dropout:<br>

$$z^{(1)}=f(W^{(1)}x^{(L)}+b^{(1)}),\quad z^{(K)}=W^{(K)}z^{(K-1)}+b^{(K)},\quad ŷ=\mathrm{softmax}(z^{(K)}).$$<br>

<!-- Add Figure: MLP for Windowed Data -->
<br>
### Non-Windowing Approach<br>

#### a. Direct Feature Utilization<br>

Here, raw \(x\in\mathbb{R}^M\) at each \(t_i\) is fed directly, skipping windowing and FFT.<br>

#### b. Random Forest Classifier<br>

Baseline ML: \(T\) trees estimate class probabilities:<br>

$$P(y=c|x)=\frac{1}{T}\sum_{t=1}^T P_t(y=c|x).$$<br>

We use 25 trees, each on random feature subsets.<br>

<!-- Add Figure: Random Forest Architecture -->
<br>
#### c. Outer Product Neural Network (OPNN)<br>

Compute outer product \(O=x x^T\in\mathbb{R}^{M\times M},\;O_{ij}=x_i x_j\), flatten to \(z\in\mathbb{R}^{M^2}\), then MLP:<br>

$$ŷ=\mathrm{softmax}(W_2\,f(W_1z+b_1)+b_2).$$<br>

This captures second‑order interactions explicitly.<br>

<!-- Add Figure: OPNN Architecture -->
<br>
### Loss & Optimization<br>

For both deep models, Cross‑Entropy Loss is:<br>

$$\mathcal{L}(\theta)=-\sum_{c}y_c \log ŷ_c,$$<br>

optimized via Adam.<br>
<br>
## Evaluation Technique<br>

We compare accuracy and inference time per sample on an M3 Pro with MPS GPU. Accuracy is:<br>

$$\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}.$$<br>
<br>
## Results<br>

| Approach                          | Accuracy (%) | Inference Time (ms/sample) | Parameters          |
|-----------------------------------|--------------|----------------------------|---------------------|
| Windowing (CNN+DCN+MLP)           | 97.86        | 0.0828                     | 46.8 M              |
| Non-Windowing (RF)                | 98.01        | 0.0313                     | —                   |
| Non-Windowing (OPNN+MLP)          | 98.43        | 0.0081                     | 11.0 M              |<br>
<br>
<!-- Add Figure: Results Comparison Chart -->

## Conclusion & Future Work<br>

Non-windowed deep learning yields both higher accuracy and lower latency, suggesting that time‑frequency segmentation may be unnecessary for static gestures. Future work should explore dynamic gesture sequences and transferability across users.<br>
<br>
## References<br>

1. Olmo & Domingo (2020). EMG Characterization. Materials, 13(24), 5815.<br>
2. Raez et al. (2006). EMG Signal Analysis. Biol. Proc. Online, 8, 11–35.<br>
3. Rani et al. (2023). sEMG & AI. IEEE Access, 11, 105140–105169.<br>
4. Asogbon et al. (2018). Window Conditioning in EMG. IEEE CBS.<br>
5. Krilova et al. (2018). EMG Data for Gestures. UCI Repository.<br>

$$$
