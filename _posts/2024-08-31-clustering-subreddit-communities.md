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

Here, $ n $ is the total number of interactions, $ u_i $ is the $ i $-th username, $ s_i $ is the $ i $-th subreddit in the dataset, and $r$ is the indicator function.



