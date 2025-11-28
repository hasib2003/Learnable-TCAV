
import os
import numpy as np
# ..........torch imports............
import torch
from torchvision import transforms

#.... Captum imports..................
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.attr import  LayerIntegratedGradients
from captum.concept import TCAV

#.... Custom imports..................
from classifier.svm import CuMLSVMClassifier
from classifier.default import DefaultClassifier
from classifier.signal_cav import SignalCav


from PIL import Image

# ..........torch imports............
import numpy as np
from scipy.stats import ttest_ind


SUPPORTED_LAYERS = [
    "layer2.1.relu",   # deeper conv in layer2
    "layer3.1.relu",   # deeper conv in layer3
    "layer4.1.relu",    # deeper conv in layer4
    "layer4.1.bn2",    # deeper conv in layer4
    "avgpool",    # deeper conv in layer4
]



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

def assemble_concept(name, id, concepts_path):
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

def get_classifier(classifier:str):
    
    if classifier == "cumlsvm":
        return CuMLSVMClassifier()
    if classifier == "default":
        return DefaultClassifier()
    if classifier == "signal":   
        return SignalCav()

    raise ValueError(f"Invalid classifier value {classifier} passed")
    
def get_concept_significance(model:torch.nn.Module,
                             layers:list[str],
                             classifier:str | None,
                             concepts_dir:str,
                             concept_name:str,
                             class_idx:int,
                             eval_images_path:str,
                             trainable:bool,
                             score_type:str,
                             device:str,
                             num_rand_concepts:int=10,
                             alpha:float=0.05,
                             random_prefix:str="random_"
                             ):
    
    """
    used to check if the concept is significantly present in the model  or not for specified class

    returns:
    
    alignment : aggregated directional derivative score
    statistics bool :  true if it is statistically significant other wise false
    
    """


    assert all(l_name in SUPPORTED_LAYERS for l_name in layers), f"Expected all layers to be in {SUPPORTED_LAYERS}"
    assert concept_name in os.listdir(concepts_dir) , f"Expected concepts to be in {os.listdir(concepts_dir)}"
    assert score_type in ["magnitude","sign_count"]

    assert not trainable or (trainable and score_type != "sign_count")




    # if len(layers) > 1 and not trainable:
    #     raise ValueError("LCAV with more than one layer is not yet supported")

    

    target_concept = assemble_concept(concept_name, 0, concepts_path=concepts_dir)
    eval_images = load_image_tensors(path=eval_images_path, transform=False)
    eval_tensors = torch.stack([transform(img) for img in eval_images])

    model.eval()
    model = model.to(device)
    eval_tensors = eval_tensors.to(device)

    assert next(model.parameters()).device == eval_tensors.device, "Model and tensors on different devices!"



    random_concepts = [assemble_concept(random_prefix + str(i+0), (i+2),concepts_path=concepts_dir) for i in range(0, num_rand_concepts)] 
    experimental_sets = [[target_concept, random_concept] for random_concept in random_concepts]

    clf = None
    if classifier is not None:
        clf = get_classifier(classifier)


    mytcav = TCAV(model=model,
                layers=layers,
                classifier=clf,
                layer_attr_method = LayerIntegratedGradients(
                model, None, multiply_by_inputs=False))

    mytcav.compute_cavs(experimental_sets, force_train=True)
    scores = mytcav.interpret(eval_tensors, experimental_sets, class_idx, n_steps=5)

    if trainable:

        assembled_scores = assemble_scores(scores,experimental_sets,0,score_layer=layers[-1],score_type=score_type)
        return torch.stack(assembled_scores).mean()

    results = {}
    for layer in layers:

        P1, P2, pval, _ = get_pval(scores, experimental_sets, layer, score_type=score_type , alpha=alpha, print_ret=False)
        results[layer] = {"concept":float(np.mean([t.cpu().item() for t in P1])),"random":float(np.mean([t.cpu().item() for t in P2])),"pval":pval}

    return results

def get_CAV(model:torch.nn.Module,
            classifier:str | None,
            concepts_dir:str,
            concept_name:str,
            device:str,
            num_rand_concepts:int=10,
            layers=["avgpool"],
            weight_idx=0,
            random_prefix:str="random_",
            force_train=True
            ):
    
    """
    computes the CAV

    input:

        model               : model,
        concept_name        : name of the concept (for which a dir must exists in CONCEPTS_DIR)
        device              : model's device
        num_rand_concepts   : number of experiments to be conducted (each with different random dir)
    
    returns:

        cavs                : dict["str":tensor]
    
    """


    assert all(l_name in SUPPORTED_LAYERS for l_name in layers), f"Expected all layers to be in {SUPPORTED_LAYERS}"
    assert concept_name in os.listdir(concepts_dir) , f"Expected concepts to be in {os.listdir(concepts_dir)}"

    

    target_concept = assemble_concept(concept_name, 0, concepts_path=concepts_dir)
    model = model.to(device)

    random_concepts = [assemble_concept(random_prefix + str(i+0), (i+2),concepts_path=concepts_dir) for i in range(0, num_rand_concepts)] 
    experimental_sets = [[target_concept, random_concept] for random_concept in random_concepts]

    clf = None
    if classifier is not None:
        clf = get_classifier(classifier)


    model.eval()
    with torch.no_grad():
        mytcav = TCAV(model=model,
                    layers=layers,
                    classifier=clf,
                    layer_attr_method = LayerIntegratedGradients(
                    model, None, multiply_by_inputs=False))

    if not force_train:
        print(f"Warning: cached cavs are being used")
    cavs = mytcav.compute_cavs(experimental_sets, force_train=force_train)

    all_cavs = {}

    for layer in layers:

        list_weights = []
        for _, cav_obj in cavs.items():


            list_weights.append(cav_obj[layer].stats["weights"][weight_idx])
        all_cavs[layer] = torch.stack(list_weights)


    return all_cavs



# if __name__ == "__main__":

#     from utils.models import get_model

#     model = get_model("resnet18",2,True)

#     cavs = get_CAV(model,None,"/netscratch/aslam/TCAV/PetImages/Concepts/","CAT-TEXT","cuda")

#     print("cavs ",cavs)
#     print("cavs.shape ",cavs.shape)

#     sim = cavs @ cavs.T

#     print(f"{sim=}")

#     print("mean cav ",torch.mean(cavs,dim=0).shape)

#     # print("cavs @ cavs")
