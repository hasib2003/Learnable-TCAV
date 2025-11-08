# Learnable CAV

This repository provides an implementation for mitigating **spurious correlations** or **biases** learned by deep neural networks, particularly **Convolutional Neural Networks (CNNs)**, using **Concept Activation Vectors (CAVs)** as guidance.

The approach modifies the **Binary Cross-Entropy (BCE)** loss function by introducing a **regularization term** based on the gradient of the logit in the direction of an *undesired concept*. This encourages the model to reduce its dependency on spurious correlations or dataset shortcuts during training.

---

## Dataset

The dataset used for experiments is a modified version of the **Cats and Dogs Classification Dataset**, sourced from Kaggle:  
[Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

### Modifications:
- **Training set:** Introduced a visual shortcut by adding a small caption **"CAT"** at the top-center of each cat image.
- **Test set:** Remains unmodified to evaluate generalization without shortcuts.

### Dataset Scripts
All scripts related to dataset modification and concept generation can be found under:
src/misc/

---

## Results
The table below summarizes the impact of the CAV-based correction on model performance.

#### Without Correction
- **Test Accuracy:** 48.6%
- **Confusion Matrix:**

| Predicted Cat | Predicted Dog |
|---------------|---------------|
| 171           | 2329          |
| 241           | 2259          |

>Observation: The model heavily relies on spurious cues, misclassifying a large portion of cat images as dogs.

#### With CAV-Based Correction
- **Best Test Accuracy:** 73.26%
- **Confusion Matrix:**

| Predicted Cat | Predicted Dog |
|---------------|---------------|
| 1249          | 1251          |
| 86            | 2414          |
> Observation: The correction significantly improves accuracy, reducing misclassifications caused by shortcuts and spurious correlations, while balancing predictions across classes."