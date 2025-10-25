#!/usr/bin/env python
# coding: utf-8

# # Show cases Testing with Concept Activation Vectors (TCAV) on Imagenet Dataset and GoogleNet model

# This tutorial shows how to apply TCAV, a concept-based model interpretability algorithm, on a classification task using GoogleNet model and imagenet dataset.
# 
# More details about the approach can be found here:
# https://arxiv.org/pdf/1711.11279.pdf
# 
# In order to use TCAV, we need to predefine a list of concepts that we want our predictions to be test against.
# 
# **Concepts** are human-understandable, high-level abstractions such as visually represented "stripes" in case of images or tokens such as "female" in case of text. Concepts are formatted and represented as input tensors and do not need to be part of the training or test datasets.
# 
# Concepts are incorporated into the importance score computations using Concept Activation Vectors (CAVs). Traditionally, CAVs train linear classifiers and learn decision boundaries between different concepts using the activations of predefined concepts in a NN layer as inputs to the classifier that we train. The vector that is orthogonal to learnt decision boundary and is pointing towards the direction of a concept is the CAV of that concept.
# 
# TCAV measures the importance of a concept for a prediction based on the directional sensitivity (derivatives) of a concept in Neural Network (NN) layers. For a given concept and layer it is obtained by **aggregating** the dot product between CAV for a given concept in a given layer and the gradients of model predictions w.r.t. given layer output. The aggregation can be performed based on either signs or magnitudes of the directional sensitivities of concepts across multiple examples belonging to a certain class. More details about the technique can be found in above mentioned papers.
# 
# **Note:** Before running this tutorial, please install the torchvision, numpy, scipy, sklearn, PIL, and matplotlib packages.
# 
# 
# 

# In[1]:


import numpy as np
import os, glob

import matplotlib.pyplot as plt

from PIL import Image

from scipy.stats import ttest_ind

# ..........torch imports............
import torch
import torchvision

from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

#.... Captum imports..................
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients

from captum.concept import TCAV

# from captum.concept._core.lcav import TCAV

from captum.concept import Concept

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str


# ## Defining image related transformations and functions

# Let's define image transformation function.

# In[2]:

import sys

this_file = os.path.abspath(__file__)

parent_parent = os.path.dirname(os.path.dirname(this_file))

if parent_parent not in sys.path:
    sys.path.insert(0, parent_parent)

from utils.models import load_from_checkpoint , get_model

CONCEPTS_DIR = "/netscratch/aslam/TCAV/PetImages/Concepts/"
CHECKPOINT_PATH="/netscratch/aslam/TCAV/text-inflation/EXP1/concept_correction/20251006_193214/best_model.pth"

DUMP_PATH= os.path.join(os.path.dirname(CHECKPOINT_PATH),"cav-testing")
os.makedirs(DUMP_PATH,exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Method to normalize an image to Imagenet mean and standard deviation
def transform(img):

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )(img)


# Now let's define a few helper functions.

# In[3]:


def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return transform(img)


def load_image_tensors(class_name, root_path=CONCEPTS_DIR, transform=True):
    path = os.path.join(root_path, class_name)
    filenames = glob.glob(path + '/*.jpg')

    tensors = []
    for filename in filenames:
        img = Image.open(filename).convert('RGB')
        tensors.append(transform(img) if transform else img)
    
    return tensors


# Defining a helper function to load predefined concepts. `assemble_concept` function reads the concepts using a directory path where the concepts are residing and constructs concept object.

# In[4]:


def assemble_concept(name, id, concepts_path=CONCEPTS_DIR):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)


# Let's assemble concepts into Concept instances using Concept class and concept images stored in `concepts_path`. We will use these concepts later in our experiments. 
# Below we define five concepts. Three out of five are related to image texture and patterns such as `striped`, `zigzagged` and `dotted`. The other two represent random concepts. Random concepts contain elements / images that are associated with various possible concepts. Distinct from those defined for `striped`, `zigzagged` and `dotted`.
# 
# Note that concepts should be created and stored under `data/tcav/image/concepts/` folder in advance. 
# 
# `striped`, `zigzagged` and `dotted` concepts can be found in broden dataset:  https://netdissect.csail.mit.edu/broden1_224, under `images/dtd` folder as also described here: https://github.com/tensorflow/tcav/tree/master/tcav/tcav_examples/image_models/imagenet. There are in total 120 images for each of the `striped`, `zigzagged` and `dotted` concepts.
# 
# Please, download those concept images and place under `striped`, `zigzagged` and `dotted` folders under `data/tcav/image/concepts/` folder accordingly.
# 
# Random type of concepts are uniformly sampled from imagenet dataset. More details on how to download and setup imagenet dataset can he found here:
# https://github.com/tensorflow/tcav/tree/master/tcav/tcav_examples/image_models/imagenet
# 
# We will randomly sample 4 diffent sets of 120 random images from imagenet dataset. Note that these images should be distinct from concept and zebra images since the testing will be performed on zebra images.
# Place random images under `random_0`, `random_1`, `random_2` and `random_3` folders under `data/tcav/image/concepts/` folder similar to `striped` concept.
# 

# In[5]:


cat_img_concept = assemble_concept("IMG-CAT", 0, concepts_path=CONCEPTS_DIR)
cat_text_concept = assemble_concept("TEXT-CAT", 1, concepts_path=CONCEPTS_DIR)


# random_0_concept = assemble_concept("random_0", 2, concepts_path=CONCEPTS_DIR)
# random_1_concept = assemble_concept("random_1", 3, concepts_path=CONCEPTS_DIR)

# Let\'s visualize some samples from those concepts

# In[6]:


# n_figs = 5
# n_concepts = 5

# fig, axs = plt.subplots(n_concepts, n_figs + 1, figsize = (25, 4 * n_concepts))

# for c, concept in enumerate([cat_img_concept, cat_text_concept, random_0_concept, random_1_concept]):
#     concept_path = os.path.join(CONCEPTS_DIR, concept.name) + "/"
#     img_files = glob.glob(concept_path + '*')
#     for i, img_file in enumerate(img_files[:n_figs + 1]):
#         if os.path.isfile(img_file):
#             if i == 0:
#                 axs[c, i].text(1.0, 0.5, str(concept.name), ha='right', va='center', family='sans-serif', size=24)
#             else:
#                 img = plt.imread(img_file)
#                 axs[c, i].imshow(img)

#             axs[c, i].axis('off')


# ## Defining GoogleNet Model

# For this tutorial, we will load GoogleNet model and set in eval mode.

# In[7]:


model = get_model(name="resnet18",num_classes=2)
model = load_from_checkpoint(model,path=CHECKPOINT_PATH,device=DEVICE)
model = model.train()

# # Computing TCAV Scores

# Next, let's create TCAV class by passing the instance of GoogleNet model, a custom classifier and the list of layers where we would like to test the importance of concepts.

# The custom classifier will be trained to learn classification boundaries between concepts. We offer a default implementation of Custom Classifier in captum library so that the users do not need to define it. Captum users, hoowever, are welcome to define their own classifers with any custom logic. CustomClassifier class extends abstract Classifier class and provides implementations for training workflow for the classifier and means to access trained weights and classes. Typically, this can be, but is not limited to a classier, from sklearn library. In this case CustomClassifier wraps `linear_model.SGDClassifier` algorithm from sklearn library.

# In[8]:


layers = [
    "layer2.1.relu",   # deeper conv in layer2
    "layer3.1.relu",   # deeper conv in layer3
    "layer4.1.relu"    # deeper conv in layer4
]

mytcav = TCAV(model=model,
              layers=layers,
              layer_attr_method = LayerIntegratedGradients(
                model, None, multiply_by_inputs=False))


# ##### Defining experimantal sets for CAV

# Then, lets create 2 experimental sets: ["striped", "random_0"] and ["striped", "random_1"].
# 
# We will train a classifier for each experimal set that learns hyperplanes which separate the concepts in each experimental set from one another. This is especially interesting when later we want to perform two-sided statistical significance testing in order to confirm or reject the significance of a concept in a specific layer for predictions.
# 
# Now, let's load sample images from imagenet. The goal is to test how sensitive model predictions are to predefined concepts such as 'striped' when predicting 'zebra' class.
# 
# 

# In[9]:


# experimental_set_rand = [[cat_img_concept, random_0_concept], [cat_img_concept, random_1_concept]]


# Now, let's load sample images from imagenet. The goal is to test how sensitive model predictions are to predefined concepts such as 'striped' when predicting 'zebra' class. 
# 
# Please, download zebra images and place them under `data/tcav/image/imagenet/zebra` folder in advance before running the cell below. For our experiments we used 50 different zebra images from imagenet dataset.
# 

# In[10]:


# Load sample images from folder
cat_imgs = load_image_tensors('IMG-CAT', transform=False)
text_imgs = load_image_tensors('TEXT-CAT', transform=False)

# Visualizing some of the images that we will use for making predictions and explaining those predictions by the means of concepts defined above.

# In[11]:


# fig, axs = plt.subplots(1, 5, figsize = (25, 5))
# axs[0].imshow(cat_imgs[40])
# axs[1].imshow(cat_imgs[41])
# axs[2].imshow(cat_imgs[34])
# axs[3].imshow(cat_imgs[31])
# axs[4].imshow(cat_imgs[30])

# axs[0].axis('off')
# axs[1].axis('off')
# axs[2].axis('off')
# axs[3].axis('off')
# axs[4].axis('off')

# plt.savefig(os.path.join(DUMP_PATH,"IMG-CAT_VIS.png"))
# plt.clf()
# Here we perform a transformation and convert the images into tensors, so that we can use them as inputs to NN model.

# In[12]:


# Load sample images from folder
cat_img_tensors = torch.stack([transform(img) for img in cat_imgs])
cat_text_tensors = torch.stack([transform(img) for img in text_imgs])
# experimental_set_rand

# In[13]:


# zebra class index
cat_idx = 0


# tcav_scores_w_random = mytcav.interpret(inputs=cat_img_tensors,
#                                         experimental_sets=experimental_set_rand,
#                                         target=cat_idx,
#                                         n_steps=5,
#                                        )
# # tcav_scores_w_random

# Auxiliary functions for visualizing of TCAV scores.

# In[14]:


def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def plot_tcav_scores(experimental_sets, tcav_scores,save_path:str):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):

        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)

        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i-1]])
        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [format_float(scores['sign_count'][i]) for layer, scores in tcav_scores[concepts_key].items()]
            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

        # Add xticks on the middle of the group bars
        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16)

        # Create legend & Show graphic
        _ax.legend(fontsize=16)

    # plt.savefig(
    plt.savefig(save_path)
    plt.clf()

# Let's use above defined auxilary functions and visualize tcav scores below.

# In[15]:



# plot_tcav_scores(experimental_set_rand, tcav_scores_w_random,os.path.join(DUMP_PATH,"tcav_scores_cat_img_2_rand.png"))


# From the plot above we observe that the images that are predicted as `zebra` by the model are very sensitive  to `striped` concept as opposed to any random concept. Note that random concepts contain a set of images that are uniformly sampled from imagenet dataset - they do not represent any specific concept.

# Now, let's compute TCAV scores for a different experimental set that contains three different specific concepts such as `striped`, `zigzagged` and `dotted`.

# In[16]:


# experimental_set_zig_dot = [[cat_img_concept, cat_text_concept]]


# In[17]:



# tcav_scores_w_zig_dot = mytcav.interpret(inputs=cat_img_tensors,
#                                          experimental_sets=experimental_set_zig_dot,
#                                          target=cat_idx,
#                                          n_steps=5)


# In[18]:


# plot_tcav_scores(experimental_set_zig_dot, tcav_scores_w_zig_dot,os.path.join(DUMP_PATH,"tcav_scores_cat_img_2_cat_text.png"))    

# Similar to the previous case in this experiment as well we observe that `striped` concept has very high TCAV scores across all  three layers compared to `zigzagged` and `dotted`. This means that the `striped` concept is an essential concept in predicting `zebra`.  

# # Statistical significance testing of concepts

# In order to convince ourselves that our concepts truly explain our predictions, we conduct statistical significance tests on TCAV scores by constructing a number of experimental sets. In this case we look into the `striped` concept and a number of random concepts.
# If `striped` concept is truly important in predicting `zebra` in the images that contain `zebra`, then we will see consistent high TCAV scores for `striped` concept across all experimental sets as apposed to any other concept.
# 
# Each experimental set contains a random concept consisting of a number of random subsamples. In our case this allows us to estimate the robustness of TCAV scores by the means of numerous random concepts. 
# 
# Let's define those experimental sets in the cell below.
# 
# 
# We will look into `striped`, and a number of random concepts drawn uniformly random from imagenet dataset.
# ```
# concept_list = ['stripped', 'ceo', 'random_0', 'random_1', 'random2', ... , 'random_n'], n is the number of concepts
# ```
# 
# Then, we will construct pairs of experimental sets that contain pairs of two concepts. In this case:
# ```
# experimental_sets = [['stripped', 'random_0'], ['stripped', 'random_1'], ... , ['striped', 'random_n'], 
#                     ['random_0', 'random_1'], ['random_0', 'random_2'], ... , ['random_0', 'random_n']]
# ```

# In[19]:


n = 10

random_concepts = [assemble_concept('random_' + str(i+0), (i+2)) for i in range(0, n)]

print(random_concepts)

# experimental_sets = [[cat_img_concept, random_0_concept], [cat_img_concept, random_1_concept]]
experimental_sets = []
experimental_sets.extend([[cat_text_concept, random_concept] for random_concept in random_concepts])
# experimental_sets.extend([[cat_img_concept, random_concept] for random_concept in random_concepts])

# experimental_sets.append([random_0_concept, random_1_concept])
# experimental_sets.extend([[random_0_concept, random_concept] for random_concept in random_concepts])

print("experimental_sets ",len(experimental_sets) )

# Now, let's define a convenience function for assembling the experiments together as lists of Concept objects, creating and running the TCAV:

# In[20]:


def assemble_scores(scores, experimental_sets, idx, score_layer, score_type):
    score_list = []
    for concepts in experimental_sets:
        score_list.append(scores["-".join([str(c.id) for c in concepts])][score_layer][score_type][idx])
        
    return score_list

# In addition, it is interesting to look into the p-values of statistical significance tests for each concept. We say, that we reject null hypothesis, if the p-value for concept's TCAV scores is smaller than 0.05. This indicates that the concept is important for model prediction.
# 
# We label concept populations as overlapping if p-value > 0.05 otherwise disjoint.

# In[21]:


def get_pval(scores, experimental_sets, score_layer, score_type, alpha=0.05, print_ret=True):
    
    P1 = assemble_scores(scores, experimental_sets, 0, score_layer, score_type)
    P2 = assemble_scores(scores, experimental_sets, 1, score_layer, score_type)
    
    if print_ret:
        print('P1[mean, std]: ', format_float(np.mean(P1)), format_float(np.std(P1)))
        print('P2[mean, std]: ', format_float(np.mean(P2)), format_float(np.std(P2)))

    _, pval = ttest_ind(P1, P2)

    if print_ret:
        print("p-values:", format_float(pval))

    if pval < alpha:    # alpha value is 0.05 or 5%
        relation = "Disjoint"
        if print_ret:
            print("Disjoint")
    else:
        relation = "Overlap"
        if print_ret:
            print("Overlap")
        
    return P1, P2, format_float(pval), relation


# We now run the TCAV and obtain the scores:

# In[22]:


# Run TCAV
# import torch.optim as optim
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# scores = mytcav.interpret(cat_text_tensors, experimental_sets, cat_idx, n_steps=5)
scores = mytcav.interpret(cat_text_tensors, experimental_sets, cat_idx, n_steps=5,grad_kwargs={"create_graph": True, "retain_graph": True})
print("scores ",scores["1-2"]["layer4.1.relu"])
# print("scores ",scores["1-2"]["layer2.1.relu"]["magnitude"][0].backward())
# optimizer.step()
# exit()
# scores = mytcav.interpret(cat_img_tensors, experimental_sets, cat_idx, n_steps=5)

# We can present the distribution of tcav scores using boxplots and the p-values indicating whether TCAV scores of those concepts are overlapping or disjoint.

# In[23]:


n = 5
def show_boxplots(layer, metric='sign_count',save_path:str=None):

    def format_label_text(experimental_sets):
        concept_id_list = [exp.name if i == 0 else \
                             exp.name.split('_')[0] for i, exp in enumerate(experimental_sets[0])]
        return concept_id_list

    n_plots = 1

    fig, ax = plt.subplots(1, n_plots, figsize = (25, 7 * 1))
    fs = 18
    for i in range(n_plots):
        esl = experimental_sets[i * n : (i+1) * n]
        P1, P2, pval, relation = get_pval(scores, esl, layer, metric)

        ax[i].set_ylim([-2, 2])
        ax[i].set_title(layer + "-" + metric + " (pval=" + str(pval) + " - " + relation + ")", fontsize=fs)
        ax[i].boxplot([P1, P2], showfliers=True)

        ax[i].set_xticklabels(format_label_text(esl), fontsize=fs)

    plt.savefig(save_path)
    plt.clf()
    print("saved at ",save_path)

# Below box plots visualize the distribution of TCAV scores for two pairs of concepts in three different layers. Each layer is visualized in a separate jupyter cell. 
# Below diagrams show that `striped` concept has TCAV scores that are consistently high across all layers and experimental sets as apposed to `random` concept. It also shows that `striped` and `random` are disjoint populations.
# 
# When we compare the populations of two random concepts we see that the p-value is much higher, meaning that those populations overlap and we cannot reject null hypothesis anymore. 

# In[24]:


# show_boxplots ("layer2.1.relu",save_path=os.path.join(DUMP_PATH,"cav_scores_text_2_rand_l2.png"))

# # In[25]:


# show_boxplots ("layer3.1.relu",save_path=os.path.join(DUMP_PATH,"cav_scores_text_2_rand_l3.png"))

# In[26]:


show_boxplots ("layer4.1.relu",save_path=os.path.join(DUMP_PATH,"cav_scores_text_2_rand_l4.png"))

# In[ ]:


