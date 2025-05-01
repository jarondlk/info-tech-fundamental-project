---
title: "Deep-Learning Models Benchmarking with Multi-Class Plankton Image"
subtitle: "Benchmarking CNN and GCN for Plankton Classification"
author: |
  Group: G1  
  C4IM2508 Jaronchai Dilokkalayakul  
  C4IM2501 Daffa Akbar Aprilio  
  C5IM2015 Ganchimeg Namuunbayar  
date: "01 May, 2025"
output:
  pdf_document:
    toc: true
    toc_depth: 4
    fig_caption: true
fontsize: 12pt
geometry: "left=1cm,right=1cm,top=1.5cm,bottom=1.5cm"
header-includes:
  - \usepackage[section]{placeins}
  - \usepackage{fixltx2e}
  - \usepackage{longtable}
  - \usepackage{pdflscape}
  - \usepackage{graphicx}
  - \usepackage{caption}
  - \usepackage{gensymb}
  - \usepackage{subcaption}
  - \DeclareUnicodeCharacter{2264}{$\pm$}
  - \DeclareUnicodeCharacter{2265}{$\geq$}
  - \usepackage{fancyhdr}
  - \usepackage{lipsum}
---


# Update 2

- Conducted experiments using several deep learning models (ResNet50, EfficientNet, DenseNet, MobileNet, GCN)
- Currently benchmarking model performance using precision, recall, F1-score
- Working on visualizing confusion matrices and ROC curves
- Next: Finalize results, document methodology, and prepare final report
- Question: GCN training shows unstable results — is there a recommended normalization technique for graph-structured images?

# Update 1

- Selected and prepared plankton dataset from IFCB source
- Chose 4 plankton classes with balanced data samples
- Preprocessed data: resized, normalized, augmented
- Split data into training and testing sets
- Performed EDA to visualize class distribution

# Executive Summary

- **Dataset**: IFCB plankton images, 4 balanced classes  
- **Cleaning**: Resizing, normalization, augmentation  
- **EDA**: Verified class balance and image uniformity  
- **Modeling**: CNNs (ResNet, DenseNet, etc.) + GCN  
- **Results**: Accuracy, precision, F1-score evaluated  
- **Conclusion**: EfficientNet yielded best performance; GCN promising but unstable

# Abstract

This project benchmarks several deep learning models—including CNNs and GCNs—for multi-class plankton image classification. Using a balanced subset of the IFCB dataset, we preprocess and augment the images to train and evaluate ResNet, EfficientNet, DenseNet, MobileNet, and GCN. Our analysis compares performance using precision, recall, and F1-score, revealing strengths and weaknesses of each approach for plankton identification.

# Introduction

Plankton classification is essential for ecological monitoring. Traditional CNNs perform well on natural images, but their effectiveness on microscopic plankton imagery remains underexplored. We aim to benchmark various models, including GCNs which may capture spatial relationships better.

- **Dataset**: Images from the Imaging FlowCytobot (IFCB)
- **Variables**: Class labels (species), image files
- **Improvement Need**: Label refinement and additional data for underrepresented species

# Data Science Methods

- Image preprocessing: resizing, normalization, augmentation
- Modeling methods: CNNs (ResNet50, EfficientNet, DenseNet, MobileNet) and GCNs
- Tools: TensorFlow, PyTorch Geometric, seaborn, matplotlib

# Exploratory Data Analysis

## Explanation of Your Dataset

| Plankton Class    | Count |
|-------------------|-------|
| Dinobryon         | 588   |
| Pseudonitzschia   | 578   |
| Dactyliosolen     | 532   |
| Corethron         | 447   |

- Data classes: Image files (JPEG/PNG)
- Labels: Plankton genus (multi-class)
- Balanced enough for supervised learning

## Data Cleaning

- Resized images to 224x224 pixels
- Normalized pixel values to [0, 1]
- Performed data augmentation (flipping, rotation)
- Split into training and test datasets

## Data Visualizations

![Class Distribution](images/class_distribution.png "Distribution of Plankton Classes")

## Variable Correlations

N/A — image data does not have traditional variable correlations, but spatial patterns are considered in modeling.

# Statistical Learning: Modeling & Prediction

We experimented with the following models:

- [x] **ResNet50**  
- [x] **EfficientNet**  
- [x] **DenseNet**  
- [x] **MobileNet**  
- [x] **Graph Convolutional Network (GCN)**

**Evaluation Metrics**:
- Accuracy
- Precision
- Recall
- F1-score

**Validation**:
- 80/20 train-test split
- Cross-validation for CNNs
- Early stopping and learning rate scheduling

**Model Selection**:
- EfficientNet had the best overall F1-score.
- GCN was computationally intensive but showed potential.

# Discussion

EfficientNet provided the most consistent classification results across all classes. While GCNs introduced a novel graph-based learning approach, their instability and sensitivity to hyperparameters limited practical application. CNNs like DenseNet also performed well but required more training time.

# Conclusions

- CNN architectures remain highly effective for image-based plankton classification.
- GCNs are promising but need more tuning and potentially larger data.
- Data preprocessing and augmentation were key to model generalization.

# Acknowledgments

We thank the IFCB project for providing open-access plankton image datasets and our professor for guidance and support throughout the project.

# References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
3. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
