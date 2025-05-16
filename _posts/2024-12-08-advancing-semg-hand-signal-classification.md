---
layout: post
title: "Advancing sEMG Hand Signal Classification"
description: "A comprehensive analysis of sEMG-based gesture recognition with windowed and non-windowed approaches."
date: 2024-12-08
feature_image: images/emg.jpg
---

Surface Electromyography (sEMG) records the electrical activity of muscles and has become a cornerstone in prosthetic control and human–machine interfaces. In this report, we compare windowed FFT-based feature extraction with non-windowed models to see how they affect classification accuracy and inference speed.

<!--more-->

The ability to classify hand gestures accurately and quickly is essential for real-time applications such as prosthetic limbs. By comparing windowed methods against direct feature modeling, we identify which approach offers the best trade-off between performance and complexity.

## Contents

1. [Problem Statement](#problem-statement)
2. [Data Source](#data-source)
3. [Methodology](#methodology)

   * [Windowing Approach](#windowing-approach)
   * [Non-Windowing Approach](#non-windowing-approach)
   * [Loss & Optimization](#loss--optimization)
4. [Evaluation Technique](#evaluation-technique)
5. [Results](#results)
6. [Conclusion & Future Work](#conclusion--future-work)
7. [References](#references)

## Problem Statement

Surface Electromyography measures the electrical activity produced by muscle contractions. Windowing segments signals into fixed intervals for robust feature extraction but introduces computational overhead and fixed temporal boundaries. Non-windowed methods model each timestep directly, potentially reducing latency. This study evaluates both approaches to determine their impact on classification accuracy and inference time.

## Data Source

We use the “EMG Data for Gestures” dataset from the UCI Machine Learning Repository, captured via a MYO Thalmic bracelet over eight channels for 36 subjects. Each recording includes a timestamp \$t\_i\$, an 8-dimensional sEMG vector \$x\_i\$, and a class label \$y\_i\$ corresponding to one of eight static gestures.

## Methodology

Below are the mathematical foundations for each pipeline.

### Windowing Approach

#### a. Segmentation into Windows

Let the raw dataset be \${t\_i, x\_i, y\_i}\_{i=1}^n\$, where \$x\_i\in\mathbb{R}^M\$ and \$y\_i\in{c\_1,\dots,c\_C}\$. Applying a sliding window of length \$W\$ produces:

$$
X^{(j)} = [\,x_i, x_{i+1}, \dots, x_{i+W-1}\,], 
\quad
Y^{(j)} = [\,y_i, y_{i+1}, \dots, y_{i+W-1}\,],
$$

for \$j=1,\dots,N\$, where \$N\$ is the number of windows.

<!-- Add Figure: Sliding Window Pattern -->

#### b. Frequency-Domain Transformation (FFT)

For each channel \$m=1,\dots,M\$, the windowed signal \$x^{(j)}\_m\$ undergoes a Discrete Fourier Transform (DFT):

$$
X^{(j)}_m(k) = \sum_{n=0}^{W-1} x_m(n)\,e^{-2\pi i k n / W},
\quad
k=0,\dots,\lfloor W/2
floor.
$$

We record both magnitude and phase:

$$
|X^{(j)}_m(k)| = \sqrt{\Re(X^{(j)}_m(k))^2 + \Im(X^{(j)}_m(k))^2},
\quad
\angle X^{(j)}_m(k)=\arctanrac{\Im(X^{(j)}_m(k))}{\Re(X^{(j)}_m(k))}.
$$

The feature vector \$H^{(j)}\in\mathbb{R}^{2M(K+1)}\$ is formed by stacking all channels.

<!-- Add Figure: FFT Magnitude & Phase -->

#### c. CNN for Frequency-Domain Feature Extraction

Treating frequency bins as a spatial dimension and channels as input planes, a 1D convolution with filters \$W\in\mathbb{R}^{F	imes C\_{\mathrm{in}}	imes C\_{\mathrm{out}}}\$ and bias \$b\$ computes:

$$
h_\ell(k,d) = f\Bigl(\sum_{c=1}^{C_{\mathrm{in}}}\sum_{\delta=0}^{F-1}H(k+\delta,c)\,W_{d,\delta,c} + b_d\Bigr),
$$

where \$f\$ is LeakyReLU: \$f(x)=\max(0,x)+\alpha\min(0,x)\$. Each block includes BatchNorm, MaxPool, and Dropout.

<!-- Add Figure: CNN Architecture -->

#### d. Deep Cross Network (DCN)

Given an embedding \$e\in\mathbb{R}^D\$, each cross layer updates:

$$
x^{(\ell+1)} = x^{(\ell)}\circigl(x^{(\ell)}W^{(\ell)}igr) + b^{(\ell)} + x^{(\ell)},\quad \ell=0,\dots,L-1,
$$

where \$\circ\$ is the Hadamard product.

<!-- Add Figure: DCN Architecture -->

#### e. Multi-Layer Perceptron (MLP)

The final vector \$x^{(L)}\$ passes through fully-connected layers:

$$
z^{(1)} = figl(W^{(1)}x^{(L)} + b^{(1)}igr),\quad
z^{(K)} = W^{(K)}z^{(K-1)} + b^{(K)},\quad
\hat y = 	ext{softmax}(z^{(K)}).
$$

<!-- Add Figure: MLP for Windowed Data -->

### Non-Windowing Approach

#### a. Raw Feature Input

Raw vectors \$x\in\mathbb{R}^M\$ at each timestamp \$t\_i\$ are fed directly into the model.

#### b. Random Forest Classifier

With \$T\$ trees, class probabilities are averaged:

$$
P(y=c\mid x) = rac{1}{T}\sum_{t=1}^T P_t(y=c\mid x).
$$

<!-- Add Figure: Random Forest Architecture -->

#### c. Outer-Product Neural Network (OPNN)

Compute the outer product \$O = x,x^	op\in\mathbb{R}^{M	imes M}\$, flatten to \$z\in\mathbb{R}^{M^2}\$, then compute:

$$
\hat y = 	ext{softmax}igl(W_2\,f(W_1 z + b_1) + b_2igr).
$$

<!-- Add Figure: OPNN Architecture -->

### Loss & Optimization

Both deep models use cross-entropy loss:

$$
\mathcal{L}(	heta) = -\sum_{c} y_c \log \hat y_c,
$$

optimized with Adam.

## Evaluation Technique

Accuracy and inference time (ms/sample) are measured on an M3 Pro with Metal Performance Shaders. Accuracy is defined as:

$$
ext{Accuracy} = rac{TP + TN}{TP + TN + FP + FN}.
$$

## Results

| Approach                     | Accuracy (%) | Inference Time (ms/sample) | Parameters |
| ---------------------------- | ------------ | -------------------------- | ---------- |
| Windowing (CNN+DCN+MLP)      | 97.86        | 0.0828                     | 46.8 M     |
| Non-Windowing (RandomForest) | 98.01        | 0.0313                     | —          |
| Non-Windowing (OPNN+MLP)     | 98.43        | 0.0081                     | 11.0 M     |

<!-- Add Figure: Results Comparison Chart -->

## Conclusion & Future Work

Non-windowed deep learning offers higher accuracy and lower latency, suggesting that time-frequency segmentation may not be necessary for static gesture recognition. Future work should explore dynamic gestures and cross-user generalization.

## References

1. Olmo & Domingo (2020). EMG Characterization. *Materials*, 13(24), 5815.
2. Raez et al. (2006). EMG Signal Analysis. *Biological Procedures Online*, 8, 11–35.
3. Rani et al. (2023). sEMG & AI. *IEEE Access*, 11, 105140–105169.
4. Asogbon et al. (2018). Window Conditioning in EMG. *IEEE CBS*.
5. Krilova et al. (2018). EMG Data for Gestures. UCI Repository.
