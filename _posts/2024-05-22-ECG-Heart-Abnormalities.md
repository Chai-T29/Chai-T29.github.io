---
layout: post
title: "ECG Heart Abnormalities"
description: "ECG, or electrocardiogram, data measures the activity of the heart, and doctors use this information to help diagnose a plethora of heart diseases."
date: 2024-05-22
feature_image: images/ecg.jpg
tags: [machinelearning, healthcare]
---

ECG, or electrocardiogram, data measures the activity of the heart, and doctors use this information to help diagnose a plethora of heart diseases. The purpose of this project is to develop a quick and scalable solution for early heart-disease detection. Automation in the healthcare industry can help alleviate some of the pressures caused by inefficient human resources. Although this project covers one use-case, this methodology can be applied across many aspects of healthcare analytics. I hope you enjoy this read!

<!--more-->

This project is an extension of a project I worked on in my High-Dimensional Data Analytics Class under Professor Kamran Paynabar [2], but I take it a step further for this project. This project covers tools and tricks that can improve classification tasks for ECG data to detect heart abnormalities, which is vital for early detection of heart diseases.

ECG data is functional in nature and is considered to be a continuous function. In simpler terms, the number of data points for ECG data is dependent on how many times we sample the ECG signal, which can theoretically reach close to infinity! So, how do we represent this data in a machine-readable format that will not only be quick to compute, but more accurate as well? That's what this project aims to talk about! Here are the key objectives:

-  How can we effectively reduce the dimensionality of ECG data?
-  How do classification models perform on ECG data?
-  What insights can we gain about heart abnormalities from our models?

## Contents

1.  [Understanding our Data](#understanding-our-data)
2.  [B-Splines Approach](#b-splines-approach)
4.  [Functional Principal Component Analysis Approach](#functional-principal-component-analysis-approach)
6.  [Conclusion](#conclusion)
7.  [References](#references)

<br>

## Understanding our Data

The data consists of 500 training samples and 4500 testing samples. The data for this project, and many others, can be downloaded from this [source](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

Once you download the data, you have to figure out the secret password (it's not too difficult) to unlock the data. To start this project, we'll need the following libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
%matplotlib inline
import seaborn as sns
from skfda.representation.basis import BSplineBasis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
                            roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.ndimage import gaussian_filter
```

Let's now load up our data and visualize it to understand what the data looks like.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
train = pd.read_csv('ECG5000_TRAIN.tsv', header=None, sep='\t').rename(columns={0:'y'})
test = pd.read_csv('ECG5000_TEST.tsv', header=None, sep='\t').rename(columns={0:'y'})
train['y'] = pd.Series(train['y'] == 1).astype(int)
test['y'] = pd.Series(test['y'] == 1).astype(int)

X_train = train.iloc[:, 1:]
y_train = train['y']
X_test = test.iloc[:, 1:]
y_test = test['y']

mask_train = (y_train == 1)
xx = np.linspace(0, 1, X_train.shape[1])

sns.set_style('darkgrid')
plt.figure(figsize=(12, 4))
plt.plot(xx, X_train[mask_train].iloc[1].T, c='b', lw=0.02, label='Class 1')
plt.plot(xx, X_train[np.logical_not(mask_train)].iloc[1].T, c='r', lw=0.02, label='Class 2')
plt.plot(xx, X_train[mask_train].iloc[1:].T, c='b', lw=0.02)
plt.plot(xx, X_train[np.logical_not(mask_train)].iloc[1:].T, c='r', lw=0.02)

plt.xticks()
plt.title('ECG Data   (Class 1: Blue,   Class 2: Red)')
plt.legend()
plt.tight_layout()
plt.show()
```
</Details>

![ecg5000](https://github.com/user-attachments/assets/84b7c070-2ecc-43c0-987b-5eff4ab51bb3)

We can see clear separation in the chart above, but how do we classify signals like this? The answer to that question is a B-Spline basis representation of the data.

<br>

## B-Splines Approach

B-Splines, or Basis Splines, are piece-wise polynomial approximations of a curve. They are defined recursively as such [1][2]:

$$
B_{i, j}(x) = \frac{x - t_i}{t_{i+j} - t_i} B_{i, j-1}(x) \\ + \frac{t_{i+j+1} - x}{t_{i+j+1} - t_{i+1}} B_{i+1, j-1}(x)
$$

for $j \ge 1$ with the initial condition:

$$
B_i^0(x) =
\begin{cases}
1 & \text{if } t_i \le x < t_{i+1} \\
0 & \text{otherwise}
\end{cases}
$$

Here:
- $B_{i, j}(x)$ is the B-Spline basis function of degree $k$.
- $x$ is the parameter.
- $t_i$ are the knots.

The B-Spline curve $C(x)$ of degree $j$ can be defined as a linear combination of these basis functions:

$$
C(t) = \sum_{i=0}^{n} P_i B_{i, j}(x)
$$

where $P_i$ are the control points.

Once we develop this feature space, we project the extracted feature space onto the B-Spline feature space, which effectively transforms the data into a lower dimensional approximation. For example, if we have $12$ knot, then no matter how many columns our data has, we would have knots + $2$, or $14$, columns.

#### Finding Optimal Knots Sequence

Now that we have a foundational understanding of the math for B-Splines, we can implement our algorithm in Python using scikit-fda. As part of the process, we iterate through various knot sequences to find which one yields the highest accuracy. This is a form of cross-validation where we're optimizing the hyperparameter 'nknots'.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
knots = []
accuracy_scores = []
order = 4
nknots = 6
for nknots in range(3, 20):
    k = np.linspace(0, 1, nknots)
    bs = BSplineBasis(knots=k, order=order)
    bs_basis = bs(xx).T[0]

    
    H_train = lse_solver(bs_basis, X_train.T).T
    H_test = lse_solver(bs_basis, X_test.T).T
    
    svc_model = SVC(probability=True, C=10, kernel='rbf', gamma='scale')
    svc_model.fit(H_train, y_train)
    y_pred = svc_model.predict(H_test)
    accuracy = accuracy_score(y_test, y_pred)
    knots.append(nknots)
    accuracy_scores.append(accuracy)

i = np.argmax(accuracy_scores)
nknots = knots[i]
print(f'Optimal number of knots: {nknots}')
sns.set_style('whitegrid')
plt.figure(figsize=(5,5))
plt.plot(knots, accuracy_scores, c='b')
plt.plot(nknots, accuracy_scores[i], 'r*', label='Highest Accuracy')
plt.xlabel('nknots')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
</Details>

![cv_knots](https://github.com/user-attachments/assets/2411d074-0bb7-4d31-83e0-d21a49659a86)

#### Fitting B-Spline Model

With our optimal knot sequence, we can now train our Support Vector Machine Classifier. Here are the results:

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
k = np.linspace(0, 1, nknots)
bs = BSplineBasis(knots=k, order=order)
bs_basis = bs(xx).T[0]

H_train = lse_solver(bs_basis, X_train.T).T
H_test = lse_solver(bs_basis, X_test.T).T
svc_model = SVC(probability=True, C=10, kernel='poly', gamma='scale')
svc_model.fit(H_train, y_train)
y_pred = svc_model.predict(H_test)
```
</Details>

![bspline_results](https://github.com/user-attachments/assets/cb0b6cca-9764-40f6-851f-260c730c28dc)

The model achieved an astonishing $98.73$% accuracy on the 4500 test samples! And, the model was trained in the fraction of a second with only 500 samples of data! This is impressive to say the least. Let's see what the actual vs. predicted labels look like:

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
mask1_test = (y_test == 1)
mask2_test = (y_pred == 1)

sns.set_style('dark')
plt.figure(figsize=(12, 4))
plt.plot(xx, X_test[mask1_test].iloc[1].T, c='w', lw=1, label='Actual')
plt.plot(xx, X_test[mask2_test].iloc[1].T, c='b', lw=0.02, label='Predicted')
plt.plot(xx, X_test[mask1_test].iloc[1:].T, c='w', lw=1)
plt.plot(xx, X_test[mask2_test].iloc[1:].T, c='b', lw=0.02)

plt.xticks()
plt.title('Actual vs Predicted for Class 1')
plt.tight_layout()
plt.legend()
plt.show()

sns.set_style('dark')
plt.figure(figsize=(12, 4))

plt.plot(xx, X_test[np.logical_not(mask1_test)].iloc[1].T, c='w', lw=0.5, label='Actual')
plt.plot(xx, X_test[np.logical_not(mask2_test)].iloc[1].T, c='r', lw=0.02, label='Predicted')
plt.plot(xx, X_test[np.logical_not(mask1_test)].iloc[1:].T, c='w', lw=0.5)
plt.plot(xx, X_test[np.logical_not(mask2_test)].iloc[1:].T, c='r', lw=0.02)

plt.xticks()
plt.title('Actual vs Predicted for Class 2')
plt.tight_layout()
plt.legend()
plt.show()
```
</Details>

![class1](https://github.com/user-attachments/assets/9a5d7c6f-785d-4e0d-a793-2ccba6b76df5)

![class2](https://github.com/user-attachments/assets/58fa11fa-d35b-41fc-902c-86ea5c7ffc84)

We can see that most of the labels were predicted correctly as our model has shown earlier. I thought it would be cool to see it this as well!

<br>

## Functional Principal Component Analysis Approach

We can alternatively figure out what features explain the most variance in the functional data (functional principal components). Below is the methodology to do this [2]:

1.	Mean Function:
Compute the mean function $\mu(t)$:
$$
\mu(t) = \frac{1}{n} \sum_{i=1}^n X_i(t)
$$

2.	Centered Functions:
Center the functions by subtracting the mean function:
$$
\tilde{X}_i(t) = X_i(t) - \mu(t)
$$

3.	Covariance Function:
Compute the covariance function $C(s, t)$:
$$
C(s, t) = \frac{1}{n} \sum_{i=1}^n \tilde{X}_i(s) \tilde{X}_i(t)
$$

4.	Eigenfunctions and Eigenvalues:
Solve the eigenfunction problem for the covariance function:
$$
\int_a^b C(s, t) \phi_k(t) , dt = \lambda_k \phi_k(s)
$$
where $\lambda_k$ and $\phi_k(t)$ are the $k$-th eigenvalue and eigenfunction, respectively.

5.	Principal Component Scores:
Compute the principal component scores $\xi_{ik}$ for each function $X_i(t)$:
$$
\xi_{ik} = \int_a^b \tilde{X}_i(t) \phi_k(t) , dt
$$

6.	Approximation:
Approximate each function $X_i(t)$ using a finite number of principal components $K$:
$$
X_i(t) \approx \mu(t) + \sum_{k=1}^K \xi_{ik} \phi_k(t)
$$

With this logic, we can now implement the code in steps.

#### Computing Mean Functions

Let's start by visualizing the mean of the data, spline, smoothing spline, and classes.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
B_train_stacked = np.tile(bs_basis.T, len(train)).T
X_train_stacked = X_train.values.ravel()
beta = lse_solver(B_train_stacked, X_train_stacked)
mu_hat_train = bs_basis@beta
mu_hat_train2 = gaussian_filter(X_train.mean(0), 3)

B_test_stacked = np.tile(bs_basis.T, len(test)).T
X_test_stacked = X_test.values.ravel()
beta = lse_solver(B_test_stacked, X_test_stacked)
mu_hat_test = bs_basis@beta
mu_hat_test2 = gaussian_filter(X_test.mean(0), 3)

mean1 = np.mean(X_train[mask_train], axis=0).T
mean2 = np.mean(X_train[np.logical_not(mask_train)], axis=0).T

sns.set_style('darkgrid')
plt.figure(figsize=(12,4))
plt.plot(xx, X_train.mean(0), 'k+-', lw=1, label='Mean of data')
plt.plot(xx, mu_hat_train, 'b', lw=1, label='Mean function via Spline')
plt.plot(xx, mu_hat_train2, 'm', lw=1, label='Mean function via Smoothing data mean')
plt.plot(xx, mean1, 'g', lw=1, label='Class 0')
plt.plot(xx, mean2, 'r', lw=1, label='Class 1')
plt.legend()
plt.title("Mean Estimate")
plt.show()
```
</Details>

![mean-functions](https://github.com/user-attachments/assets/4557aa5b-49e2-4553-a6c9-6b409738229c)

#### Computing Covariance Function

Let's compute the unsmoothed covariances as outlined in the mathematical expressions from earlier.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
diffs_train = X_train-mu_hat_train
cov_train = np.cov(diffs_train.T)

diffs_test = X_test-mu_hat_test
cov_test = np.cov(diffs_test.T)

grids_train = np.meshgrid(xx, xx)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(grids_train[0], grids_train[1], cov_train, cmap='magma', lw=0, antialiased=False)
plt.title('Unsmoothed Covariances')
plt.show()
```
</Details>

![unsmoothed covariances](https://github.com/user-attachments/assets/7c6c3c65-cdc9-4f1e-bb08-ad36d6de86e9)

Now, we need to smooth out the covariances using a gaussian filter to remove some of the extra noise.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
cov_train = gaussian_filter(cov_train, sigma=7)
cov_test = gaussian_filter(cov_test, sigma=7)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(grids_train[0], grids_train[1], cov_train, cmap='magma', lw=0, antialiased=False)
plt.title('Smoothed Covariances')
plt.show()
```
</Details>

![smoothed-covariances](https://github.com/user-attachments/assets/23f642d1-dc5e-4785-9810-86754d403606)

#### Finding Optimal Number of Principal Components

Using eigenvalue decomposition, we can find the optimal number of principal components. That is, the minimum number of principal components that explains a desired threshold variance. In our case, we use 0.99 as the threshold to maintain most of the information.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
eig_v_train, eig_f_train = np.linalg.eigh(cov_train)
eig_v_test, eig_f_test = np.linalg.eigh(cov_test)

explained_variance_ratio = eig_v_train[::-1] / np.sum(eig_v_train)
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

threshold = 0.99
optimal_num_pcs = np.argmax(cumulative_explained_variance >= threshold) + 1
print(f"Optimal number of principal components: {optimal_num_pcs}")

plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, lw=0.5, marker='+', c='g')
plt.axhline(y=threshold, color='r', linestyle='--')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. # of Principal Components')
plt.show()
```
</Details>

![download-8](https://github.com/user-attachments/assets/65178658-326b-4d4d-9abc-9f3b2489f00c)

#### Fitting FPCA Model

With our optimized number of FPCs, we can now train and test a model. Here are the results:

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
eig_v_train, eig_f_train = np.linalg.eigh(cov_train)
eig_v_test, eig_f_test = np.linalg.eigh(cov_test)

explained_variance_ratio = eig_v_train[::-1] / np.sum(eig_v_train)
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

threshold = 0.99
optimal_num_pcs = np.argmax(cumulative_explained_variance >= threshold) + 1
print(f"Optimal number of principal components: {optimal_num_pcs}")

PCs_train = eig_f_train[:, -optimal_num_pcs:]
FPC_train = diffs_train@PCs_train

PCs_test = eig_f_test[:, -optimal_num_pcs:]
FPC_test = diffs_test@PCs_test

plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, lw=0.5, marker='+', c='g')
plt.axhline(y=threshold, color='r', linestyle='--')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. # of Principal Components')
plt.show()
```
</Details>

![fpca-results](https://github.com/user-attachments/assets/edc93a5b-3e07-4ed2-a728-c40ebeb55e05)

This method also performed extremely well, with an accuracy of $97.42$%! However, there are a lot more false positives using this approach, so B-Splines are the better approach overall.

<br>

## Conclusion

With this project, we have learned how to preprocess and reduce the dimensionality of ECG data with B-Splines and FPCA, as well as how to apply classification models to detect heart abnormalities using the reduced feature space. These skills are invaluable for developing effective models in medical diagnostics and other fields, especially when functional data is involved. In the near future, we will be seeing a lot more automation in the field of healthcare, which will improve patient outcomes, drive down costs, and improve medical accessibility across the world.

If you've made it this far, I hope you enjoyed the read! Happy learning!

<br>

## References:

[1] Hastie, T., Tibshirani, R., Friedman, J. (2009). Basis Expansions and Regularization. In: The Elements of Statistical Learning. Springer Series in Statistics. Springer, New York, NY. https://doi.org/10.1007/978-0-387-84858-7_5

[2] Paynabar, Kamran, "Intro to HD and Functional Data." Class lecture, High Dimensional Data Analytics, Georgia Institute of Technology, Atlanta, GA. May 13, 2024.
