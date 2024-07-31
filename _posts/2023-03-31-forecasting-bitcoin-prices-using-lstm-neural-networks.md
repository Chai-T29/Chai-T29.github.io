---
layout: post
title: "Forecasting Bitcoin Prices using LSTM Neural Networks"
description: "Customer churn is a core component of marketing analytics and marketing-focused data science."
date: 2023-03-31
feature_image: images/bitcoin.jpg
---
Customer churn is a core component of marketing analytics and marketing-focused data science. Analyzing customer churn is crucial for shaping business strategy as it provides insights into why customers leave, allowing businesses to identify and address underlying issues, improve customer retention, and enhance overall customer satisfaction. By understanding the factors that contribute to churn, companies can implement targeted interventions, optimize their products or services, and tailor marketing efforts to retain existing customers and attract new ones. Did this pique your interest? Then let's dive right in!

<!--more-->

This project showcases the efficacy of logistic regression models in predicting customer churn (1 for churned, and 0 for retained). The project initially covers Principal Component Analysis (PCA) for dimensionality reduction and logistic regression (using gradient ascent and Newton's method).

Here are the main questions this project aims to answer:

1.  How effective are the logistic regression models at predicting customer churn?
2.  How can different approaches to logistic regression change the outcome of the results?
3.  How well can we predict churn and what does that tell us about shaping business strategy?

## Contents

Here are the contents of this project:

1.  [Understanding the Data](#understanding-the-data)
2.  [PCA for Dimensionality Reduction](#pca-for-dimensionality-reduction)
3.  [Logistic Regression](#logistic-regression)
4.  [Observations and Results](#observations-and-results)
5.  [References](#references)

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

### Gradient Ascent Approach

The gradient ascent algorithm is based on the first derivative, which generally takes a while to compute with a constant learning rate. However, with a decay function, the model converges at an optimum much faster.

Let's outline the algorithm we will be using for this method.

For each iteration $t$ from 0 to $T-1$:

**1.** Update the learning rate $\alpha_t$ using the decay rate:

   $$ \alpha_t = \alpha \times \beta^t $$

**2.** Calculate the gradient of the log-likelihood with respect to \(\theta\):

   $$ \nabla \log L(\theta^{(t)}) = \mathbf{X}^T (\mathbf{y} - \sigma(\mathbf{X} \theta^{(t)})) $$

**3.** Perform a gradient ascent step to update $\theta$:

   $$ \theta^{(t+1)} = \theta^{(t)} + \alpha_t \nabla \log L(\theta^{(t)}) $$

**4.** Calculate the logistic loss (negative log-likelihood):

   $$ \text{log loss} = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] $$

where $ p_i = \sigma(\mathbf{X}_i \theta) $ and $ \sigma(z) $ is the sigmoid function:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

The optimal parameter vector $\theta_{\text{optimal}}$ is obtained within $T$ iterations:

$$ \theta_{\text{optimal}} = \theta^{(T)} $$

#### Training the Model

With the math out of the way, we can train the model.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
start = time.time()

# Setting up our hyperparameters
ALPHA = 0.05        # learning rate
DECAY_RATE = 0.95   # incrementally reduces learning rate
MAX_STEP = 100      # total no. of iterations

# Get the data coordinate matrix, X, and label vector, y
X_train = np.array(X_train)
y_train = np.array(y_train)

# Store guesses of theta, for subsequent analysis
thetas = np.zeros((k, MAX_STEP+1))

# Incrementally going up the gradient function to converge at an optimal theta
losses_ga = []
for t in range(MAX_STEP):
    alpha_t = adjust_learning_rate(ALPHA, t, DECAY_RATE)
    gradient = grad_log_likelihood(thetas[:,t:t+1], y_train, X_train)
    s = alpha_t * gradient
    thetas[:,t+1:t+2] = thetas[:,t:t+1] + s
    losses_ga.append(compute_log_loss(thetas[:,t:t+1], X_train, y_train))

# Storing optimal theta
theta_ga = thetas[:, MAX_STEP:]

end = time.time()
training_time = end - start

# Visualizing training loss
plt.figure(figsize=(10, 6))
plt.plot(losses_ga, label='Log-Loss', color='g')
plt.xlabel('Iteration')
plt.ylabel('Negative Log-Likelihood Loss')
plt.title('Loss Over Iterations')
plt.legend()
plt.show()

print(f"Training Time: {training_time:.3f} seconds")
```
</Details>

Here's the training loss for this model:

![ga_loss](https://github.com/user-attachments/assets/747e7c95-06f0-4a1d-8871-6b334d64f395)

As you can see, the model converges at an optimal answer and fast! This model only took 0.344 seconds to train, with an accuracy of 84.6% on the training data. Here's the detailed performance of the model:

![ga_results](https://github.com/user-attachments/assets/34a85e5c-1f6d-4b5a-b791-f2bbc2558500)

<br>

### Newton's Method

Unlike the gradient ascent approach, which is based on the first derivative, Newton's Method involves the use of the Hessian matrix, which is the second derivative. In theory, this should be faster than a standard gradient ascent approach without a decay function, but will it be faster than the model from earlier with a decay? Let's find out!

First, let's break down the math of what's going on.

For each iteration $t$ from 0 to $T-1$:

**1.** Calculate the Hessian matrix $ H(\theta^{(t)}) $:

   $$ H(\theta^{(t)}) = -\mathbf{X}^T \text{diag}(\sigma(\mathbf{X} \theta^{(t)}) (1 - \sigma(\mathbf{X} \theta^{(t)}))) \mathbf{X} $$

**2.** Calculate the gradient of the log-likelihood with respect to $\theta$:

   $$ \nabla \log L(\theta^{(t)}) = \mathbf{X}^T (\mathbf{y} - \sigma(\mathbf{X} \theta^{(t)})) $$

**3.** Solve for the Newton step $s$ by solving the linear system:

   $$ H(\theta^{(t)}) s = -\nabla \log L(\theta^{(t)}) $$

**4.** Perform a Newton step to update $\theta$:

   $$ \theta^{(t+1)} = \theta^{(t)} + s $$

**5.** Calculate the logistic loss (negative log-likelihood):

   $$ \text{log loss} = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] $$

where $ p_i = \sigma(\mathbf{X}_i \theta) $ and $ \sigma(z) $ is the sigmoid function:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

The optimal parameter vector $\theta_{\text{optimal}}$ is obtained after $T$ iterations:

$$ \theta_{\text{optimal}} = \theta^{(T)} $$

#### Training the Model

Let's assess how this new algorithm performs!

<Details markdown="block">
<summary>Click here to view the code</summary>
  
```python
start = time.time()

# Setting up our hyperparameters
MAX_STEP = 55      # total no. of iterations

k = X_train.shape[1]
thetas_newt = np.zeros((k, MAX_STEP+1))
losses_newt = []
for t in range(MAX_STEP):
    hessian = hess_log_likelihood(thetas_newt[:,t:t+1], X_train)
    gradient = grad_log_likelihood(thetas_newt[:,t:t+1], y_train, X_train)
    s = solve(hessian, -gradient)
    thetas_newt[:,t+1:t+2] = thetas_newt[:,t:t+1] + s
    losses_newt.append(compute_log_loss(thetas[:,t:t+1], X_train, y_train))
theta_newt = thetas_newt[:, MAX_STEP:]

end = time.time()
training_time = end - start

plt.figure(figsize=(10, 6))
plt.plot(losses_newt, label='Log-Loss', color='g')
plt.xlabel('Iteration')
plt.ylabel('Negative Log-Likelihood Loss')
plt.title('Loss Over Iterations')
plt.legend()
plt.show()

print(f"Training Time: {training_time:.3f} seconds")
```
</Details>

Here's the training loss for this model:

![nm_loss](https://github.com/user-attachments/assets/7f4746d7-f929-45bb-b3ea-62dbde4d208f)

The model trained in much fewer steps than the gradient ascent approach, but each step took significantly longer with the Hessian computation. Because of this, the model took 0.713 seconds to converge, which is nearly half the speed of the other model. The accuracy is also slightly lower at 84.16%, which is not a significant difference. Here's a more detailed overview of the performance:

![nm_results](https://github.com/user-attachments/assets/40f56b32-dcd1-4c11-b7b0-8e55f1fefa27)

<br>

## Observations and Results

In conclusion, comparing the logistic regression models trained with gradient ascent and Newton’s method provides useful insights into predicting customer churn. The gradient ascent model converges faster and performs slightly better, though both models have a solid ROC AUC of 0.83. There’s room to reduce false negatives, which is important for keeping customers. Using better features or combining these models with other techniques could help. For the business, these results show that improving predictive models can help identify customers at risk of churning, leading to better retention strategies.

<br>

## References

[1] Vuduc, Richard, “Compression via the PCA and the SVD.” Class lecture, Computing for Data Analysis, Georgia Institute of Technology, Atlanta, GA. April 19, 2024.

[2] Vuduc, Richard, “Logistic Regression.” Class lecture, Computing for Data Analysis, Georgia Institute of Technology, Atlanta, GA. April 12, 2024.
