# Deep-Learning Methods Benchmarking with Multi-Class Plankton Image Data

**[IM20500231] Information Technology Fundamental Part I - Introduction to Materials Data Science**

> C4IM2508 Jaronchai Dilokkalayakul  
> C4IM2501 Daffa Akbar Aprilio  
> C5IM2015 Ganchimeg Namuunbayar  

## Introduction

**Objectives**: To benchmark and compare the performance of various deep learning models, including Convolutional Neural Networks (CNNs) and Graph Convolutional Networks (GCNs), on plankton classification tasks.

**Goals**: Identify the most effective architecture for accurately classifying plankton species.

## EDA, Data Cleaning and Data Preparation

### Obtaining data for benchmarking

Strategy:

- Choosing 4 classes from the raw data folder:
- Represents a genus (no mixing, no non-plankton image, etc.)
- Every class contains similar amount of files

### Plankton classes to use in benchmarking

![Alt text](assets/figures/species.png)

- Dinobryon
- Pseudonitzschia
- Dactyliosolen
- Corethron

### Exploratory data analysis

![Alt text](assets/figures/eda.png)


### Data preparation for modeling

![Alt text](assets/figures/batch.png)

| Techniques | Description |
| :-------- | :-------- |
| Resize | Ensure fixed input size for model (e.g., 224×224) |
| Normalize | Scale pixel values to [0, 1] or [-1, 1] |
| Augmentation | Improve generalization and prevent overfitting |
| Train-test split | Evaluate properly on unseen data |


## Deep Learning Modelings

1. Convolutional Neural Network
2. Unsupervised Learning
3. Graph Network

## Convolutional Neural Network

### Data Preprocessing for CNN

### Model Training for CNN

### Model Evaluation for CNN

## Unsupervised Learning

### Data Preprocessing for Unsupervised

### Model Training for Unsupervised

### Clustering Visualization

![Alt text](assets/figures/unsup_tnse.png)

### Adjusted Rand Index (ARI): 0.7307

![Alt text](assets/figures/unsup_conf.png)

## Graph Network

### Data Preprocessing for Graph

### Model Training for Graph

### Model Evaluation for Graph

## Benchmarkings

## Summary
