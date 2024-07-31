---
layout: post
title: "Facial Recognition"
description: "This project leverages the power of Higher-Order Discriminant Analysis (HODA) to classify celebrity photos."
date: 2019-01-03
feature_image: images/mountain.jpg
tags: [tips, work]
---

This project leverages the power of Higher-Order Discriminant Analysis (HODA) and Support Vector Machines (SVM) to classify celebrity photos. And yes, this project does not use any neural networks! Facial recognition technology has become a crucial tool in modern society due to its wide-ranging applications in security, healthcare, social media, and entertainment. By enabling machines to identify and verify individuals based on their facial features, it enhances security systems, simplifies user authentication processes, and offers personalized user experiences.

<!--more-->

The growing importance of facial recognition technology underscores the need for advanced algorithms that can accurately and efficiently process high-dimensional image data. HODA is a supervised feature extraction algorithm, and this notebook implements the algorithm as outlined in “Tensor Decompositions for Feature Extraction and Classification of High Dimensional Datasets” by Anh Huy Phan and Andrzej Cichocki [[1]](#references).

## Contents

Here are the main sections of this article:

1. [Loading the Data](#loading-the-data)
2. [Feature Extraction using HODA](#feature-extraction-using-hoda)
3. [Fitting the Model](#fitting-the-model)
4. [Conclusion](#conclusion)
5. [References](#references)

<br>

## Loading the Data
The dataset was found on Kaggle and is called the [Hollywood Celebrity Facial Recognition Dataset](https://www.kaggle.com/datasets/bhaveshmittal/celebrity-face-recognition-dataset). There are 17 actors and 100 photos per actor that are relevant to us. Scarlett Johanson has 200 photos because she's just that awesome! However, we will only be using 100 to standardize the model training.

Before we start, we need to set up some libraries for our analysis.

```python
import numpy as np
import pandas as pd
import zipfile
import io
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import seaborn as sns
%matplotlib inline
import tensorly as tl
from tensorly.tenalg import multi_mode_dot
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict
from tqdm.notebook import tqdm
```

Once we have our libraries set up, we can run the code below to create our training and testing tensors with dimensions as height x width x samples, and labels for each sample. The data processing ensures that all of the photos are the same size and then converts them to grayscale. We could leave RGB on, but it doesn't add much information for our analysis and it takes three times longer to compute.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
target_shape = (240, 240)

X = []
y = []

zip_path = 'archive.zip'
i = 0
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    try:
        for file in zip_ref.infolist():
            folder_name = file.filename.split('/')[0]
            y.append(folder_name)
            with zip_ref.open(file.filename) as image_file:
                img = plt.imread(io.BytesIO(image_file.read()), format='jpeg')
                
                if img.shape[:2] != target_shape:
                    factors = (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1], 1)
                    img = zoom(img, factors, order=3)
                    
                gray_img = (img@np.array([0.2989, 0.5870, 0.1140])).astype(np.float32)
                X.append(gray_img)
    
            i += 1
    except Exception as e:
        print(f'Error on iteration {i}')

X = np.array(X).transpose(1, 2, 0)
y = np.array(y)

label_indices = defaultdict(list)
for index, label in enumerate(y):
    label_indices[label].append(index)

train_indices = []
test_indices = []

for label, indices in label_indices.items():
    np.random.shuffle(indices)
    train_indices.extend(indices[:97])
    test_indices.extend(indices[97:100])

X_train, X_test = X[..., train_indices], X[..., test_indices]
y_train, y_test = y[train_indices], y[test_indices]

print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)
```
</Details>

Now that we have our data set up, let's look at some examples of the photos.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
indices = np.random.choice(X_train.shape[2], 25, replace=False)

fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[..., indices[i]], cmap='gray')
    ax.set_title(y_train[indices[i]])

fig.suptitle('25 Randomly Sampled Images from the Dataset')
plt.tight_layout()
plt.show()
```
</Details>

![download](https://github.com/user-attachments/assets/bea1c1cf-5f81-4f95-96d8-c8a421f510fa)

We can see that not all photos are standardized, and there are many inconsistencies in age, facial accessories, angles, and more. Let's find out how much this will affect our analysis!

<br>

## Feature Extraction using HODA

As mentioned earlier, this feature extraction method is based off of the paper, “Tensor Decompositions for Feature Extraction and Classification of High Dimensional Datasets” by Anh Huy Phan and Andrzej Cichocki [[1]](#references), and it is an iterative process. If you are not the biggest fan of math, you can skip over to the [next section](#implementing-the-hoda-algorithm)!

The HODA algorithm is formulated as such:

$$
\underline{\tilde{\mathbf{X}}}^{(k)} = \underline{\mathbf{X}}^{(k)} - \underline{\tilde{\mathbf{X}}}^{(c_k)}
$$

$$
\underline{\tilde{\mathbf{X}}}^{(c)} = \sqrt{K_c} \left( \underline{\tilde{\mathbf{X}}}^{(c)} - \underline{\tilde{\mathbf{X}}} \right)
$$

**input** : $\underline{\mathbf{X}}$: Concatenated tensor of $K$ training samples $I_1 \times I_2 \times \cdots \times I_N \times K$  
**output** : $\mathbf{U}^{(n)}$: $N$ orthogonal basis factors $I_n \times J_n \ (n = 1, 2, \ldots, N)$  
**output** : $\mathbf{G}$: Training feature tensors $J_1 \times J_2 \times \cdots \times J_N \times K$  

**begin**

- Initialize $\mathbf{U}^{(n)}$
    
- Calculate $\underline{\tilde{\mathbf{X}}}$ and $\underline{\tilde{\mathbf{X}}}^{(c)}$ as outlined above.

    **repeat**
        **for** $n = 1$ to $N$ **do**
  
$$\tilde{\mathbf{Z}}_n = \underline{\tilde{\mathbf{X}}} \times_{\{(n,N+1)\}} \{ \mathbf{U}^{(T)} \}$$

$$\mathbf{S}_w^n = \tilde{\mathbf{Z}}_n \tilde{\mathbf{Z}}_n^{(T)}$$

$$\tilde{\mathbf{Z}}_n^b = \underline{\tilde{\mathbf{X}}}^{(c)} \times_{\{(n,N+1)\}} \{ \mathbf{U}^{(T)} \}$$

$$\mathbf{S}_b^n = \tilde{\mathbf{Z}}_n^b \tilde{\mathbf{Z}}_n^{b(T)}$$

$$
\varphi = \frac{\text{trace}(\mathbf{U}^{(n)T} \mathbf{S}_b^n \mathbf{U}^{(n)})}{\text{trace}(\mathbf{U}^{(n)T} \mathbf{S}_w^n \mathbf{U}^{(n)})}
$$

$$
\left[ \mathbf{U}^{(n)}, \Lambda \right] = \text{eigs}(\mathbf{S}_b^n - \varphi \mathbf{S}_w^n, J_n, 'LM')
$$

$$
or /> [\mathbf{U}^{(n)}, \Lambda] = \text{eigs}(\mathbf{S}_b^n, \mathbf{S}_w^n, J_n, 'LM')
$$

$$
\left[ \mathbf{U}^{(n)}, \Lambda \right] = \text{eigs}(\mathbf{U}^{(n)} \mathbf{U}^{(n)T} \mathbf{X}_{\underline{-n}} \mathbf{X}_{\underline{-n}}^T \mathbf{U}^{(n)} \mathbf{U}^{(n)T}, J_n, 'LM')
$$

- **until** a criterion is met

- $\mathbf{G} = \mathbf{X}_{\underline{-}(N+1)} \left[ \mathbf{U} \right]^T$

**end**

The output of the model is similar to Tucker Decomposition, but it is a supervised approach to it. So, understanding Tucker Decomposition can help get an idea of what's going on under the hood. The diagram below explains it more clearly.

![Third-order-Tucker-decomposition](https://github.com/user-attachments/assets/235fa466-0aaf-49df-95a8-876c40171dd5)

Source: https://www.researchgate.net/figure/Third-order-Tucker-decomposition_fig1_257482079

In our case, the factor matrices, when multiplied by the original tensor across each axis, will generate a lower dimensional representation of the images. And so, our algorithm must return these factor matrices.

#### Implementing the HODA Algorithm

With all the math out of the way, we can implement our algorithm! The function in the code below returns the factor matrices that need to be multiplied across specific modes, as mentioned earlier. We compute the HODA algorithm on our training data, get the factor matrices, and multiply them across each axis for both training and testing tensors.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
def HODA(X_train, y_train, new_dims, alpha=1.0):
    I, K = X_train.shape[:-1], X_train.shape[-1]
    training_samples = K // len(np.unique(y_train))

    X_c = np.zeros_like(X_train).astype(np.float32)
    for lab in np.unique(y_train)[1:]:
        X_c[..., y_train == lab] = np.mean(X_train[..., y_train == lab].astype(np.float32), axis=-1, keepdims=True)

    X_mean = np.mean(X_train.astype(np.float32), axis=-1, keepdims=True)

    U_n = [np.random.rand(I[i], new_dims[i]).astype(np.complex64) for i in range(len(new_dims))]

    X_v = np.sqrt(training_samples).astype(np.float32) * (X_c - X_mean)

    X_tilde = X_train.astype(np.float32) - X_c

    for _ in tqdm(range(10), desc='Total Iterations'):
        for n in range(len(new_dims)):
            Z_tilde = tl.unfold(multi_mode_dot(X_tilde.astype(np.complex64), [U_n[i].T for i in range(len(U_n)) if i != n], modes=[i for i in range(len(U_n)) if i != n]), mode=n)

            S_w = (Z_tilde @ Z_tilde.T).astype(np.complex64)

            Z_v = tl.unfold(multi_mode_dot(X_v.astype(np.complex64), [U_n[i].T for i in range(len(U_n)) if i != n], modes=[i for i in range(len(U_n)) if i != n]), mode=n)

            S_b = (Z_v @ Z_v.T).astype(np.complex64)

            phi = np.trace(U_n[n].T @ S_b @ U_n[n]) / np.trace(alpha * (U_n[n].T @ S_w @ U_n[n]) + (1-alpha) * np.eye(new_dims[n]).astype(np.complex64))
            print(phi)

            U_n[n] = np.linalg.eig(S_b - phi * S_w)[1][:, :new_dims[n]]
            
            X_train_unfolded = tl.unfold(X_train.astype(np.complex64), mode=n)

            U_n[n] = np.linalg.eig(U_n[n] @ U_n[n].T @ X_train_unfolded @ X_train_unfolded.T @ U_n[n] @ U_n[n].T)[1][:, :new_dims[n]]

    return U_n, phi

new_dims = (10, 10)
U_n, phi = HODA(X_train, y_train, new_dims, alpha=1e7)
print('Phi at Convergence:', phi)

G_train = multi_mode_dot(X_train, [np.real(U.T) for U in U_n], modes=[0, 1])
G_test = multi_mode_dot(X_test, [np.real(U.T) for U in U_n], modes=[0, 1])
```
</Details>

With 'new_dims' set at (10, 10), the dimensionality of both 'G_train' and 'G_test' are reduced to 10 x 10 x samples. Now, you might be wondering: what do the example images from earlier look like now? Well, let's find out!

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
G_train = multi_mode_dot(X_train, [np.real(U.T) for U in U_n], modes=[0, 1])
G_test = multi_mode_dot(X_test, [np.real(U.T) for U in U_n], modes=[0, 1])

fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(G_train[..., indices[i]], cmap='gray')
    ax.set_title(y_train[indices[i]])

fig.suptitle('25 Randomly Sampled Images from the Dataset')
plt.tight_layout()
plt.show()
```
</Details>

![download-1](https://github.com/user-attachments/assets/31af00fc-3c0b-4c53-b6c3-01111141758d)

The images look nothing like they did before, and they appear to be uninformative! But, let's put it through a model and see how well it performs.

## Fitting the Model

We'll be using a Support Vector Classifier from sklearn's library for classification. The first step is to flatten the columns of the data, and we can then fit the model with our new features.

```python
G_train = tl.unfold(G_train, mode=-1)
G_test = tl.unfold(G_test, mode=-1)

svc_model = SVC(probability=True, C=100, kernel='rbf', gamma='scale')
svc_model.fit(G_train, y_train)
```

The model performed fairly well, with an accuracy of 70.59% with minimal tuning! Here's a detailed overview of the performance:

![download-2](https://github.com/user-attachments/assets/3c73c806-23a1-4d90-95d0-f05886bf17c1)

Most of the predictions are falling across the diagonal, which shows us that the algorithm can pick up relevant features. To improve the model, we could use neural networks, but I wanted to focus on the ability to classify high-dimensional data with simpler models.

## Conclusion

In this project, we’ve demonstrated the power of Higher-Order Discriminant Analysis (HODA) combined with Support Vector Machines (SVM) for classifying celebrity photos. Despite not using any neural networks, the model achieved impressive results, showcasing the effectiveness of advanced tensor decompositions for feature extraction. Facial recognition technology, as explored here, has vast applications in various fields including security, healthcare, social media, and entertainment, making it an indispensable tool in modern society. Looking ahead, there are several ways to enhance this project. Incorporating larger and more diverse datasets could further improve model robustness. Additionally, optimizing the HODA algorithm and exploring its integration with other machine-learning models could yield even better performance.

Thank you for following along on this journey into facial recognition technology. I hope you found it informative and fun!

## References

[1] Anh Huy Phan, Andrzej Cichocki, Tensor decompositions for feature extraction and classification of high dimensional datasets, Nonlinear Theory and Its Applications, IEICE, 2010, Volume 1, Issue 1, Pages 37-68, Released on J-STAGE October 01, 2010, Online ISSN 2185-4106, https://doi.org/10.1587/nolta.1.37, https://www.jstage.jst.go.jp/article/nolta/1/1/1_1_37/_article/-char/en
