import os
from PIL import Image

# ..........torch imports............
from torchvision import transforms

#.... Captum imports..................
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset



import numpy as np
from scipy.stats import ttest_ind

CONCEPTS_DIR = "/netscratch/aslam/TCAV/PetImages/Concepts/"
CHECKPOINT_PATH="/netscratch/aslam/TCAV/text-inflation/EXP1/20251001_204742/best_model.pth"

DUMP_PATH= os.path.join(os.path.dirname(CHECKPOINT_PATH),"cav-testing")
os.makedirs(DUMP_PATH,exist_ok=True)

# Method to normalize an image to Imagenet mean and standard deviation

def transform(img):

    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])(img)

def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return transform(img)

def load_image_tensors(path:str, transform=True,max_files=50):


    # print("path ",path)
    
    filenames = os.listdir(path)

    rng = np.random.default_rng(42)
    filenames = rng.choice(filenames,size=max_files,replace=False)

    filenames = [os.path.join(path,file) for file in filenames]


    # filenames = np.random.choice(filenames,size=max_files,replace=False)
    

    tensors = []
    for filename in filenames:
        img = Image.open(filename).convert('RGB')
        tensors.append(transform(img) if transform else img)
    
    return tensors

def assemble_concept(name, id, concepts_path=CONCEPTS_DIR):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)


def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def assemble_scores(scores, experimental_sets, idx, score_layer, score_type):
    score_list = []
    for concepts in experimental_sets:
        score_list.append(scores["-".join([str(c.id) for c in concepts])][score_layer][score_type][idx].cpu())
        
    return score_list


def get_pval(scores, experimental_sets, score_layer, score_type, alpha=0.05, print_ret=False):
    
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


