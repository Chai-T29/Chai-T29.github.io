---
layout: post
title: "Clustering Subreddit Communities"
description: "This project models Reddit user interactions to understand behavior and patterns."
date: 2024-08-31
feature_image: images/subreddit.jpg
---

This project models Reddit user interactions to understand behavior and patterns. This project utilizes Spectral Clustering from the Graph Laplacian (using normalized cuts) to uncover local communities where users have more concentrated interactions. We discuss the importance of understanding user behavior, the results of this project, and areas of improvement for further research. If you're interested, then let's dive right in!

<!--more-->

Reddit is a vast platform with countless communities, and understanding how these communities connect can provide valuable insights. For instance, knowing which Subreddits are often visited by the same users can help in creating better content recommendations, improving user engagement, or even in moderating content more effectively.

## Contents

Here are the main sections of this article:

1. [Loading the Data](#loading-the-data)
2. [Constructing the Graph Laplacian](#constructing-the-graph-laplacian)
3. [K-Means Algorithm](#k-means-algorithm)
4. [Results and Observations](#results-and-observations)
5. [References](#references)

<br>

## Loading the Data

The dataset was found on Kaggle and is called [Subreddit Interactions for 25,000 Users](https://www.kaggle.com/datasets/colemaclean/subreddit-interactions/data). The dataset contains three columns: "user", "subreddit", and "utc-stamp". For our analysis, we are only interested in the first two columns.

Before we start, we need to set up some libraries for our analysis and load in our data (Note: throughout this article, I will not be showing the subreddit names too often as far too many of them are NSFW!).

```python
import numpy as np
import pandas as pd
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
import os

interactions = pd.read_csv('reddit_data.csv').iloc[:, :-1]  # remove utc times
interactions.tail()
```

### Preprocessing the Data

Now that we have our data loaded in, we need to somehow standardize user interactions so that one user does not have greater influence than another user. The approach outlined in the code below groups the interactions by username and finds the proportion of visits made by a user on specific subreddits. Here's the mathematical formulation of this problem:

Given a dataset of interactions where each interaction is defined by a 'username' and a 'subreddit', we want to compute the probability of a user interacting with a specific subreddit. 

$$
p(u, s) = \frac{\sum_{i=1}^{n} r(u_i = u, s_i = s)}{\sum_{i=1}^{n} r(u_i = u)}
$$

Here, $n$ is the total number of interactions, $u_i$ is the $i$-th username, $s_i$ is the $i$-th subreddit in the dataset, and $r$ is the indicator function. Here is how we can implement this in Python efficiently:

```python
subreddit_counts = interactions.groupby(['username', 'subreddit']).size().reset_index(name='subreddit_count')
total_counts = interactions.groupby('username').size().reset_index(name='total_count')
merged_df = pd.merge(subreddit_counts, total_counts, on='username')
merged_df['probabilities'] = merged_df['subreddit_count'] / merged_df['total_count']
interactions = pd.merge(interactions, merged_df[['username', 'subreddit', 'probabilities']], on=['username', 'subreddit'], how='left').values

del subreddit_counts, total_counts, merged_df

interactions = np.vstack(tuple(set(map(tuple,interactions))))  # removing duplicate interactions
display(interactions[-5:])
```

Now we have converted interactions into a numpy array and added a third columns with the interaction proportions by user. The data now only contains unique rows because we do not need to model redundant edges in our matrix. With this, we can create some maps that will help construct our adjacency matrix and reduce the computational complexity for algorithms later in this project.

```python
nodes = np.unique(interactions[:, 1])
nodes_map = {node: i for i, node in enumerate(nodes)}
reverse_map = {i: node for i, node in enumerate(nodes)}
print("Number of Subreddit pages:", nodes.shape[0])
```

<br>

## Constructing the Graph Laplacian

With our data set up and ready to go, we can begin constructing a weighted Adjacency Matrix. In our case, let $A$ be an adjacency matrix where $A_{i, j}$ represents the weighted connection between node $i$ and node $j$. Nodes are mapped from pages using a mapping function $\text{nodes\_map}$ such that $i = \text{nodes\_map}(\text{page1})$ and $j = \text{nodes\_map}(\text{page2})$. If you wish to skip all the math, you can click [here](#implementing-the-algorithm).

For each user $u$, let  $P_u$  be the set of pages visited by $u$, and let $p_1$ and $p_2$ be any two pages in $P_u$. Define the class probabilities associated with $p_1$ and $p_2$ as $c(p_1)$ and $c(p_2)$, respectively.

***The Adjacency Matrix $A$ is defined as follows [2]:***
$$
A_{i, j} = \sum_{u \in U} \sum_{\substack{p_1, p_2 \in P_u \ p_1 \neq p_2}} c(p_1) \cdot c(p_2)
$$

Where:
- $U$ is the set of all unique users.
- $P_u$ is the set of pages visited by user $u$.
- $c(p_1)$ and $c(p_2)$ are the class probabilities for pages $p_1$ and $p_2$, respectively.
- $i = \text{nodes\_map}(p_1)$ and $j = \text{nodes\_map}(p_2)$.
- $A_{i, j}$  is incremented by the product of the class probabilities $c(p_1) \cdot c(p_2)$.

With $A$ defined, we must then remove any disconnected edges.

Define a mask vector $\mathbf{m}$ of length $n$ such that:

$$
\mathbf{m}_i =
\begin{cases}
1 & \text{if } \sum_{j=1}^{n} A_{i, j} \neq 0 \
0 & \text{if } \sum_{j=1}^{n} A_{i, j} = 0
\end{cases}
$$

Here, $\mathbf{m}_i$ is $1$ if node $i$ has at least one connection, and $0$ if it has no connections.

The reduced adjacency matrix $A{\prime}$ is obtained by selecting only the rows and columns where $\mathbf{m}[i] = 1$:

$$
A’ = A[\mathbf{m} = 1, \mathbf{m} = 1]
$$

***The Degree Matrix $D$ is defined as follows [2]:***

Given a reduced adjacency matrix $A{\prime}$ (obtained from the original adjacency matrix $A$ after removing isolated nodes), the normalized diagonal degree matrix $D$ is defined as:

$$
D = \text{diag}\left(\frac{1}{\sqrt{\sum_{j=1}^{m} A’_{i,j}}}\right)
$$

***The Graph Laplacian Matrix $L$ is defined as follows [1]:***

Via the normalized cuts method [1], the Graph Laplacian can be defined as:

$$
L = D \times A \times D
$$

To make the matrix symmetrical we can simply do:

$$
L = L + L^T
$$

***Computing top $k$ eigenvectors***

Given the eigenvalue decomposition for $L$:

$$
Lx_i = \lambda_i x_i
$$

Where:

- $L$ is symmetric.
- $\lambda_i$ are the eigenvalues, ordered such that $ \lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_n $.
- $x_i$ are the corresponding eigenvectors.

The top $k$ eigenvectors then become:

$$
X_k = [x_{(1)}, x_{(2)}, \dots, x_{(k)}]
$$

<br>

### Implementing the Algorithm

With all the math aside, we can now create and run our algorithm!

```python
A = np.zeros(shape=(nodes.shape[0], nodes.shape[0]), dtype=np.float32)

for u in tqdm(np.unique(interactions[:, 0])):
    users = interactions[interactions[:, 0] == u, 1:]
    
    for ind, page1 in enumerate(users[:-1]):
        i = nodes_map[page1[0]]
        for page2 in users[ind:]:
            j = nodes_map[page2[0]]
            if page1[0] != page2[0]:
                A[i, j] += page1[1].astype(np.float32) * page2[1].astype(np.float32)

A = A + A.T

mask = np.sum(A, axis=1) != 0
A = A[mask][:, mask]

D = np.diag(1/np.sqrt(np.sum(A, axis=1)))

L = D @ A @ D  # Graph Laplacian via Normalized Cuts method [1]
v, x = np.linalg.eigh(L)

print(f"A.shape: {A.shape}")
print(f"D.shape: {D.shape}")
print(f"L.shape: {L.shape}")
print(f"x.shape: {x.shape}")
```

Since the computation of the eigenvalue decomposition is extremely time-consuming, we can save the eigenvalue decomposition matrix with the code below.

```python
eigs = os.path.join(os.getcwd(), f'GL_Eigenvectors_Weighted')
os.makedirs(eigs, exist_ok=True)
np.save(os.path.join(eigs, 'eigenvectors_x.npy'), x)
```

<br>

## K-Means Algorithm

<Details markdown="block">
<summary>Click here to view the code</summary>

```python

```
</Details>


<Details markdown="block">
<summary>Click here to view the code</summary>

```python

```
</Details>

