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

As mentioned earlier, this feature extraction method is based off of the paper, “Tensor Decompositions for Feature Extraction and Classification of High Dimensional Datasets” by Anh Huy Phan and Andrzej Cichocki [[1]](#references), and it is an iterative process. If you are not the biggest fan of math, you can skip over this section!

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

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
```
</Details>

## Fitting the Model
<Details markdown="block">
<summary>Click here to view the code</summary>

```python
```
</Details>

## Conclusion

## References

[1] Anh Huy Phan, Andrzej Cichocki, Tensor decompositions for feature extraction and classification of high dimensional datasets, Nonlinear Theory and Its Applications, IEICE, 2010, Volume 1, Issue 1, Pages 37-68, Released on J-STAGE October 01, 2010, Online ISSN 2185-4106, https://doi.org/10.1587/nolta.1.37, https://www.jstage.jst.go.jp/article/nolta/1/1/1_1_37/_article/-char/en
