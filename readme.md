# Learnable CAV

This repository provides an implementation for mitigating **spurious correlations** or **biases** learned by deep neural networks, particularly **Convolutional Neural Networks (CNNs)**, using **Concept Activation Vectors (CAVs)** as guidance.

The approach modifies the **Binary Cross-Entropy (BCE)** loss function by introducing a **regularization term** based on the gradient of the logit in the direction of an *undesired concept*. This encourages the model to reduce its dependency on spurious correlations or dataset shortcuts during training.

---

## Overview

Spurious correlations are common in deep learning models, where they learn to associate irrelevant features (shortcuts) with the target label. This repository introduces a *learnable CAV-based regularization* technique that constrains the model from aligning its decision boundary with those undesired features.

The implementation supports:
- Learnable concept vectors (CAVs) for interpretability.
- Gradient-based regularization integrated with the BCE loss.
- Easy extension to other architectures and datasets.

---

## Dataset

The dataset used for experiments is a modified version of the **Cats and Dogs Classification Dataset**, sourced from Kaggle:  
[Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

### Modifications:
- **Training set:** Introduced a visual shortcut by adding a small caption **"CAT"** at the top-center of each cat image.
- **Test set:** Remains unmodified to evaluate generalization without shortcuts.

### Dataset Scripts
All scripts related to dataset modification and concept generation can be found under:
