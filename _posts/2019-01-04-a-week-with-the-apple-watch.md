---
layout: post
title: "Understanding Customer Churn"
description: "Customer churn is a core component of marketing analytics and marketing-focused data science."
date: 2024-04-24
feature_image: images/churn.jpg
tags: [marketing, analytics, machinelearning]
---
Customer churn is a core component of marketing analytics and marketing-focused data science. Analyzing customer churn is crucial for shaping business strategy as it provides insights into why customers leave, allowing businesses to identify and address underlying issues, improve customer retention, and enhance overall customer satisfaction. By understanding the factors that contribute to churn, companies can implement targeted interventions, optimize their products or services, and tailor marketing efforts to retain existing customers and attract new ones. Did this pique your interest? Then let's dive right in!

<!--more-->

This project showcases the efficacy of various binary classification models in predicting customer churn (1 for churned, and 0 for retained). The project initially covers Principal Component Analysis (PCA) for dimensionality reduction and Logistic Regression (using gradient ascent and Newton's method). Additionally, we compare the efficacy of Decision Trees, Support Vector Machines, K-nearest neighbors, Random Forest, and Gradient Boosting models from the sklearn library based on training time and performance metrics.

Here are the main questions this project aims to answer:

1.  How effective are the Logistic Regression models at predicting customer churn?
2.  What other models can we use and how do they perform compared to a simple Logistic Regression model?
3.  How well can we predict churn and what does that tell us about shaping business strategy?

## Contents

Here are the contents of this project:

1.  [Understanding the Data](#understanding-the-data)
2.  [PCA for Dimensionality Reduction](#pca-for-dimensionality-reduction)
3.  [Logistic Regression](#logistic-regression)
4.  [Other Models](#other-models)
5.  [Observations and Results](#observations-and-results)
6.  [References](#references)

<br>

## Understanding the Data

The dataset was found on Kaggle at [this source](https://www.kaggle.com/datasets/anandshaw2001/customer-churn-dataset/data) and includes 14 columns:

-  RowNumber: A unique identifier for each row in the dataset.
-  CustomerId: Unique customer identification number.
-  Surname: The last name of the customer (for privacy reasons, consider anonymizing this data if not already done).
-  CreditScore: The customer's credit score at the time of data collection.
-  Geography: The customer's country or region, providing insights into location-based trends in churn.
-  Gender: The customer's gender.
-  Age: The customer's age, valuable for demographic analysis.
-  Tenure: The number of years the customer has been with the bank.
-  Balance: The customer's account balance.
-  NumOfProducts: The number of products the customer has purchased or subscribed to.
-  HasCrCard: Indicates whether the customer has a credit card (1) or not (0).
-  IsActiveMember: Indicates whether the customer is an active member (1) or not (0).
-  EstimatedSalary: The customer's estimated salary.
-  Exited: The target variable, indicating whether the customer has churned (1) or not (0).

#### Setting up our Analysis

Let's start our analysis by importing the necessary libraries for this project:

```python
import pandas as pd
import numpy as np
from numpy.linalg import svd
from numpy.linalg import solve
from scipy.special import expit as sigmoid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
                            roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from math import e
import time
```

Now, let's import the data and try to understand what the data looks like under the hood. Here's a quick description

```python
# Reading our data into a pandas dataframe
data = pd.read_csv('Churn_Modelling.csv', index_col=False).rename(columns={'Exited': 'Churned'})

# Creating another column based on 'NumOfProducts' that converts it to categorical labels
data['NumOfProducts_Categorical'] = data['NumOfProducts'].astype(str)

# Checking number of unique customers
n_customers = len(data['CustomerId'].unique())
print(f'\nNumber of unique customers: {n_customers}')

display(data.describe())
```

There are 10000 unique customers and 10000 data points, which means that each row of the dataset corresponds to exactly one customer. Let's dig into the data a little bit more!

#### Exploratory Data Visualization

Since our dataset contains both categorical and continuous variables, we need to identify which ones are what for creating appropriate visualizations.

```python
categorical_cols = ['Geography', 'Gender', 'NumOfProducts_Categorical', 'HasCrCard', 'IsActiveMember']
continuous_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
```

Now let's create some cool visualizations, starting with a pairplot!

```python
sns.pairplot(data[continuous_cols + categorical_cols + ['Churned']], hue='Churned', palette='Greens')
plt.show()
```

![pairplot](https://github.com/user-attachments/assets/e63aa944-be74-46f1-b3e0-75be88c0f4f5)

It is immediately apparent that there is a significant correlation between many of the independent variables. It should be noted that since the variables are plotted against each other, the matrix of plots has a diagonal of ones in a sense (because variables are perfectly correlated to themselves). If the plot was viewed as a triangular matrix, all of the relevant information would still be intact.

Here are some histograms for the continuous data:

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(continuous_cols):
    ax = axs.flatten()[i]
    sns.histplot(data[col], kde=True, ax=ax, color='g')
    ax.set_title(f'Distribution of {col}')
plt.tight_layout()
```
</Details>

![histograms](https://github.com/user-attachments/assets/8a282d72-1f40-40fa-959a-aaa7c38ec092)

The distributions of the CreditScore, Balance, and Age columns are relatively normal with right skew, outliers at 0, and left skewed, respectively. The distribution of NumOfProducts drops significantly in the two product ranges. The distributions of Tenure and EstimatedSalary are relatively constant throughout.

Here are some countplots for the categorical data:

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(categorical_cols):
    ax = axs.flatten()[i]
    sns.countplot(x=data[col], ax=ax, palette='Greens')
    ax.set_title(f'Frequency of {col}')
plt.tight_layout()
```
</Details>

![countplots](https://github.com/user-attachments/assets/a43f91b8-c08d-4d3d-a9ae-5b27987e836f)

The distributions of the categorical variables immediately show the differences between individual groups within the data. This is especially noticeable in the Geography, HasCrCard, and NumOfProduct_Categorical (same as the continuous version of this data) columns.

We can also gain better insights into the continuous columns when we split the data by churn rate. I created boxplots below to visualize this:

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(continuous_cols):
    ax = axs.flatten()[i]
    sns.boxplot(x='Churned', y=col, data=data, ax=ax, palette='Greens')
    ax.set_title(f'Distribution of {col} by Churn Rate')
plt.tight_layout()
```
</Details>

![boxplots](https://github.com/user-attachments/assets/c68a2d41-06b1-4ebd-b10d-aa64857067d5)

These plots showcase the distributions of the continuous variables across the Churned columns. Although some of the boxplots are nearly identical, there are slight variations in the distributions that the models should be able to learn from.

#### Preprocessing Data

Now that we have a better understanding of our dataset, we can start building our model, but the first step is to preprocess the data. This step includes creating dummy variables for the categorical columns and scaling the continuous columns to have a mean of 0 and a standard deviation of 1. This is because principal component analysis requires that the data is centered.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
# Splitting the data between X and y (independent and dependent variables)
X = data.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Churned'], axis=1, inplace=False)
y = data[['Churned']]

print('\033[1mOriginal X data\033[0m:')
display(X.head())
print('\n\033[1my data\033[0m:')
display(y.head())

# Creating dummy columns for all of the non-binary categorical columns
X = pd.get_dummies(X, columns=['Geography', 'Gender', 'NumOfProducts_Categorical'])

# Scaling the data using sklearn's StandardScaler function (normalizes the data with mean of 0 and var of 1)
scaler = StandardScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

print('\033[1mNormalized X data\033[0m:')
display(X.head())
```
</Details>

## PCA for Dimensionality Reduction

Now that our data is fully ready, let's go over the math to reduce the dimensionality of the data [[1]](#references)! If you're not a fan of math, feel free to skip to the [next section](#implementing-the-algorithm).

**1.** Decompose matrix  $\mathbf{X}$ :

$$ \mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T $$

**2.** Truncate SVD by keeping the top $k$ dimensions:

$$ \mathbf{X} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T $$

**3.** Compute compression error:

$$ \text{compression_error}(\mathbf{\Sigma}, k) = \frac{\sqrt{\sum_{i=k+1}^r \sigma_i^2}}{\sqrt{\sum_{i=1}^r \sigma_i^2}} $$

**4.** Find $k$ such that the error is minimized:

$$ \text{compression_error}(\mathbf{\Sigma}, k) \leq \text{error_rate} $$

<br>
Here's a visualization of what principal components look like:

![PCA](https://github.com/user-attachments/assets/e090105e-b6e4-436a-9bb3-b8a6cdad3e00)

Source: https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d

#### Implementing the Algorithm

Now that the math is out of the way, we can create an algorithm to implement this. I'll be splitting the algorithm into three main functions: finding the rank, decomposing/truncating the matrices, and computing the compression error [[1]](#references). Upon some trial and error, setting the error_rate to 0.1 was sufficient for the algorithm to find an optimal solution.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
def svd_decompose (X, k):
    """Decomposes matrix with numpy.linalg.svd() and obtains top k dimensions for U, Sigma, and VT."""
    
    assert k > 0
    Uk, Sk, VTk = svd(X, full_matrices=False)
    
    return (Uk[:,:k], Sk[:k], VTk[:k,:])

def compression_error (Sigma, k):
    """Computes the compression error of the svd_decompose() function."""
    
    return Sigma[k:].dot(Sigma[k:])**0.5 / Sigma.dot(Sigma)**0.5

def find_rank (Sigma, error_rate=0.1):
    """Finds rank based on Sigma and desired error rate."""
    
    for i in range(len(Sigma) - 1):
        k = i + 1
        error = compression_error(Sigma, k)
        if error <= error_rate:
            break
            
    print(f'\n\033[1mCompression Error\033[0m: {error}')
    
    return k

# Getting the full SVD decomposition matrices without reduced dimensionality
U, Sigma, VT = svd_decompose (X, X.shape[1])
print('\n\033[1mShapes of SVD Matrices\033[0m:')
print(U.shape, Sigma.shape, VT.shape)

# Finding optimal rank (k) to use for our analysis
error_rate = 0.1
k = find_rank(Sigma, error_rate=error_rate)
print(f'There are \033[1m{k}\033[0m relevant features at an error rate of \033[1m{error_rate}\033[0m.')
```
</Details>

The compression error is ~3.6% at an allowed error rate of 0.1. The optimal rank is 13, which means that we reduced the number of features by about 23%. This shows us that most of the data is independent and that it is most likely not heavily correlated. However, we were still able to reduce the dimensionality with minimal loss.

Now that we have our optimal rank, we can simply decompose the matrix and project the reduced feature space onto our data.

```python
Uk, Sk, VTk = svd_decompose (X, k)
X_compressed = X.dot(VTk.T)
X_compressed.head()
```

## Logistic Regression

With our reduced dataset, we can now start building models. Starting with logistic regression, I will be covering a gradient ascent approach with a learning decay function, and Newton's Method [[2]](#references).

#### Gradient Ascent Approach

Let's outline the algorithm we will be using for this method.

For each iteration \( t \) from 0 to \( T-1 \):

1. **Adjust Learning Rate**:
   Update the learning rate \(\alpha_t\) using the decay rate:

   $$ \alpha_t = \alpha \times \beta^t $$

2. **Compute Gradient of Log-Likelihood**:
   Calculate the gradient of the log-likelihood with respect to \(\theta\):

   $$ \nabla \log L(\theta^{(t)}) = \mathbf{X}^T (\mathbf{y} - \sigma(\mathbf{X} \theta^{(t)})) $$

3. **Update Theta**:
   Perform a gradient ascent step to update \(\theta\):

   $$ \theta^{(t+1)} = \theta^{(t)} + \alpha_t \nabla \log L(\theta^{(t)}) $$

4. **Compute Log Loss**:
   Calculate the logistic loss (negative log-likelihood):

   $$ \text{log loss} = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] $$

where \( p_i = \sigma(\mathbf{X}_i \theta) \) and \( \sigma(z) \) is the sigmoid function:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

The optimal parameter vector \(\theta_{\text{optimal}}\) is obtained within \(T\) iterations:

$$ \theta_{\text{optimal}} = \theta^{(T)} $$

Here's the code for this algorithm

## Other Models

## Observations and Results

## References

[1] Vuduc, Richard, “Compression via the PCA and the SVD.” Class lecture, Computing for Data Analysis, Georgia Institute of Technology, Atlanta, GA. April 19, 2024.

[2] Vuduc, Richard, “Logistic Regression.” Class lecture, Computing for Data Analysis, Georgia Institute of Technology, Atlanta, GA. April 12, 2024.
