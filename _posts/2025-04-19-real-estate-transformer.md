---
layout: post
title: "Navigating the Rent-versus-Buy Decision with Deep Learning"
description: "A Transformer-based framework for forecasting ZIP-level housing and rental cost dynamics."
date: 2025-04-19
feature_image: images/rent_vs_buy.jpg
---

This project presents a deep learning framework for forecasting monthly home prices and rental costs at the ZIP-code level across the United States. We integrate Redfin property sales data with American Community Survey rental figures to produce localized, forward-looking insights on the rent-versus-buy decision. If you’d like to explore the full code and dataset, you can find everything [here]([<YOUR_GOOGLE_DRIVE_LINK>](https://drive.google.com/drive/folders/1wBVWORZl7w8UccK1tqoNAn0_LC3hptcO?usp=sharing)).

<!--more-->

Deciding whether to rent or buy a home is one of the most consequential financial choices individuals face, yet readily available tools often rely on broad averages, neglecting local variation and temporal trends. By forecasting both markets simultaneously at the ZIP-code scale, our approach empowers users with precise, neighborhood-level break-even estimates tailored to their financial assumptions.

## Contents

1. [Data Sources](#data-sources)  
2. [Data Processing](#data-processing)  
3. [Model Architecture](#model-architecture)  
4. [Visualization Dashboards](#visualization-dashboards)  
5. [Evaluation Metrics and Results](#evaluation-metrics-and-results)  
6. [Conclusion and Future Work](#conclusion-and-future-work)  
7. [References](#references)

<br>

## Data Sources

We draw upon two primary datasets:

- **Redfin Housing Data:** Quarterly ZIP-code-level home sale prices and market metrics from 2013 to
- **American Community Survey (ACS) Rental Data:** Annual median rent by ZIP code over the same period.

<br>

## Data Processing

1. **Alignment and Imputation:** ZIP codes are aligned across both datasets, reshaped into full tensors via cartesian products, and missing values are filled by linear interpolation with endpoint extrapolation.  
2. **Zero-Aware Normalization:** We mask zeros when computing means and standard deviations to ensure stable gradient flow and robustness to sparse entries.  
   
For housing data $X_{\mathrm{housing}}\in\mathbb{R}^{B\times T\times F\times P}$, define an indicator $1_{x\neq0}$. For each feature $f$:

<div style="overflow-x: auto;">
$$
\mu_f = \frac{1}{N_f + \epsilon}\sum_{b=1}^B\sum_{t=1}^T\sum_{p=1}^P 1_{x_{b,t,f,p}\neq0}\,x_{b,t,f,p},\qquad
N_f = \sum_{b,t,p}1_{x_{b,t,f,p}\neq0}+\epsilon
$$
</div>

$$
\sigma_f = \sqrt{\frac{1}{N_f + \epsilon}\sum_{b,t,p}1_{x_{b,t,f,p}\neq0}\,x_{b,t,f,p}^2 - \mu_f^2}
$$

<div style="overflow-x: auto;">
$$
\mathrm{ZA\_Norm}(x_{f}) = \frac{1_{x_f\neq0}\,x_f - \mu_f}{\sigma_f + \epsilon},\quad
x = \mathrm{ZA\_Norm}(x_{\mathrm{norm}})\,(\sigma_f + \epsilon) + \mu_f
$$
</div>

<br>

## Model Architecture

Our Transformer-based model jointly forecasts housing and rental prices 12 months ahead from a 36-month lookback. It consists of:

1. **ZIP Code Embeddings:** Compress a one-hot vector of length 24,488 into a dense embedding $Z_e\in\mathbb{R}^F$:

$$
 \mathbb{R}^{24{,}488}\;\xrightarrow{\mathrm{Embedding}}\;\mathbb{R}^F
$$

2. **MLP Projection Layers:** Transform normalized housing tensors:

$$
 X'_{\mathrm{housing}}
 = \mathrm{ReLU}\bigl(W_2\,\mathrm{ReLU}(W_1\,X_{\mathrm{housing}}+b_1)+b_2\bigr)
$$

3. **Transformer Encoder–Decoder:**  
   - Encoder input $X_{\mathrm{enc}}\in\mathbb{R}^{B\times T\times d_{\mathrm{embed}}}$  
   - Decoder input $X_{\mathrm{dec}}\in\mathbb{R}^{B\times T\times(P+1)}$

   We follow “Attention Is All You Need” with two modifications: we project encoder outputs down before feeding the decoder, and we initialize the decoder with the last observed ZA-normalized values.  

<img width="591" alt="Screenshot 2025-06-08 at 12 51 07 PM" src="https://github.com/user-attachments/assets/5829c5a1-f86a-47cc-b0db-616b60ea51eb" />

<br>

## Visualization Dashboards

We deployed two complementary interfaces:

- **Flask Dashboard:** Interactive map and ZIP selector for side-by-side forecast charts, allowing users to input mortgage rate and down payment. This application can be run locally by following the steps in our [Colab Notebook](https://colab.research.google.com/drive/1eK-8Jtmvc8v7x1bd74if_X0ct84L21BG?usp=drive_link).

- **Tableau Dashboard:** Choropleth map illustrating rent vs. buy cost-effectiveness, with adjustable parameters to explore regional patterns. You can view the full Tableau visualization [here](https://public.tableau.com/shared/GZY7ZH5DJ?:display_count=n&:origin=viz_share_link)

{% raw %}
<style>
  /* ensure container never overflows */
  .tableauPlaceholder {
    max-width: 100% !important;
  }
  .tableauPlaceholder object {
    width: 100% !important;
    display: block;
  }
</style>

<div class="tableauPlaceholder" id="viz1749402530869" style="position: relative; width: 100%;">
  <noscript>
    <a href="#">
      <img alt="Dashboard 1"
           src="https://public.tableau.com/static/images/GZ/GZY7ZH5DJ/1_rss.png"
           style="border: none; width: 100%;" />
    </a>
  </noscript>
  <object class="tableauViz" style="display:none;">
    <param name="host_url"        value="https%3A%2F%2Fpublic.tableau.com%2F" />
    <param name="embed_code_version" value="3" />
    <param name="path"            value="shared/GZY7ZH5DJ" />
    <param name="toolbar"         value="yes" />
    <param name="static_image"    value="https://public.tableau.com/static/images/GZ/GZY7ZH5DJ/1.png" />
    <param name="animate_transition"    value="yes" />
    <param name="display_static_image"  value="yes" />
    <param name="display_spinner"       value="yes" />
    <param name="display_overlay"       value="yes" />
    <param name="display_count"         value="yes" />
    <param name="language"              value="en-US" />
    <param name="filter"                value="publish=yes" />
  </object>
</div>

<script type="text/javascript">
  (function() {
    var divElement = document.getElementById('viz1749402530869');
    var vizElement = divElement.getElementsByTagName('object')[0];
    var aspectRatio = 827 / 1000; // original height ÷ original width

    function resizeViz() {
      var w = divElement.offsetWidth;
      vizElement.style.width  = w + 'px';
      vizElement.style.height = Math.round(w * aspectRatio) + 'px';
    }
    // initial resize
    resizeViz();
    // resize on window changes
    window.addEventListener('resize', resizeViz);

    var scriptElement = document.createElement('script');
    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
    vizElement.parentNode.insertBefore(scriptElement, vizElement);
  })();
</script>
{% endraw %}

These tools translate complex model outputs into intuitive visuals, enabling non-technical users to compare localized forecasts easily.

<br>

## Evaluation Metrics and Results

We trained with Huber loss and monitored four metrics defined as:

$$
\mathrm{RMSE} = \frac{1}{n}\sum_{i=1}^n(y_i - \hat y_i)^2,\quad
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^n|y_i - \hat y_i|,
$$

$$
R^2 = 1 - \frac{\sum_i(y_i - \hat y_i)^2}{\sum_i(y_i - \bar y)^2},\quad
\mathrm{MAPE} = \frac{100}{n}\sum_{i=1}^n\left|\frac{y_i - \hat y_i}{y_i}\right|
$$

Under our final hyperparameters (36-month window, embedding = 32, feed-forward = 128, dropout = 0.3), we achieved:

- **Housing Forecasts:** RMSE = \$155,560.5, MAE = \$84,031.0, MAPE = 42.4 %, R² = 0.73  
- **Rental Forecasts:** RMSE = \$97.1, MAE = \$57.7, MAPE = 5.98 %, R² = 0.95  

<br>

## Conclusion and Future Work

By integrating spatio-temporal data with a zero-aware Transformer framework, we deliver precise, neighborhood-level rent-versus-buy forecasts. Limitations include sparsely populated ZIP codes and sensitivity to market shocks. Future enhancements could incorporate macroeconomic indicators, refine spatial granularity to census tracts, and add confidence intervals for uncertainty quantification. 

Despite these challenges, our system marks a significant step toward democratizing housing market forecasts, empowering individuals and investors with localized, forward-looking evidence.

<br>

## References

1. Fan et al., “Determinants of House Price: A Decision Tree Approach,” *Urban Studies* (2006).  
2. Vaswani et al., “Attention Is All You Need,” arXiv (2017).  
3. Redfin Data Center: https://www.redfin.com/news/data-center  
4. ACS Median Gross Rent: https://data.census.gov/table/ACSDT5Y2022.B25064  
