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

Let's start our analysis by importing the necessary libraries for this project:

```python
import pandas as pd
import numpy as np
from numpy.linalg import svd
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

Now, let's import the data and try to understand what the data looks like under the hood.

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
![dataframe_head](https://github.com/user-attachments/assets/179f9bcb-dccf-4d5a-910d-88b0fdb22509)

There are 10000 unique customers and 10000 data points, which means that each row of the dataset corresponds to exactly one customer. Although

## PCA for Dimensionality Reduction

## Logistic Regression

## Other Models

## Observations and Results

## References

[1] Vuduc, Richard, “Compression via the PCA and the SVD.” Class lecture, Computing for Data Analysis, Georgia Institute of Technology, Atlanta, GA. April 19, 2024.

[2] Vuduc, Richard, “Logistic Regression.” Class lecture, Computing for Data Analysis, Georgia Institute of Technology, Atlanta, GA. April 12, 2024.
