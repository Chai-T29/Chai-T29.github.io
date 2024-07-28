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

ECG data is functional in nature and is considered to be a continuous function. In simpler terms, the number of data points for ECG data is dependent on how many times we sample the ECG signal, which can theoritically reach close to infinity! So, how do we represent this data in a machine-readable format that will not only be quick to compute, but more accurate as well? That's what this project aims to talk about! Here are the key objectives:

-  How can we effectively reduce the dimensionality of ECG data?
-  How do different classification models perform on ECG data?
-  What insights can we gain about heart abnormalities from our models?

The data consists of 500 training samples and 4500 testing samples. The data for this project, and many others, can be downloaded from the following source: https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/

Once you download the data, you have to figure out the secret password (it's not too difficult) to unlock the data. Let's dive right in!

<br>

## Contents

1.  [Understanding our Data](#understanding-our-data)
2.  [B-Splines for Dimensionality Reduction](#b-splines-for-dimensionality-reduction)
3.  [Model Fitting](#model-fitting)
4.  [Functional Principal Component Analysis](#functional-principal-component-analysis)
5.  [Conclusion](#conclusion)
6.  [References](#references)

## Understanding our Data

## B-Splines for Dimensionality Reduction

## References:

[1] Hastie, T., Tibshirani, R., Friedman, J. (2009). Basis Expansions and Regularization. In: The Elements of Statistical Learning. Springer Series in Statistics. Springer, New York, NY. https://doi.org/10.1007/978-0-387-84858-7_5

[2] Paynabar, Kamran, "Intro to HD and Functional Data." Class lecture, High Dimensional Data Analytics, Georgia Institute of Technology, Atlanta, GA. May 13, 2024.
