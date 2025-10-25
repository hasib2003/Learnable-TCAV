
import os
import sys

import numpy as np
from scipy.stats import ttest_ind

# ..........torch imports............
import torch
import torch.optim as optim

#.... Captum imports..................
from captum.attr import  LayerIntegratedGradients
from captum.concept import TCAV
#.... Custom imports..................

from tqdm import tqdm

current_file = os.path.abspath(__file__)
parent_parent = os.path.dirname(os.path.dirname(current_file))
if parent_parent not in sys.path:
    sys.path.insert(0, parent_parent)

from utils.captum import transform, load_image_tensors, assemble_concept, get_pval,assemble_scores,format_float


CONCEPTS_DIR = "/netscratch/aslam/TCAV/PetImages/Concepts/"

SUPPORTED_LAYERS = [
    "layer2.1.relu",   # deeper conv in layer2
    "layer3.1.relu",   # deeper conv in layer3
    "layer4.1.relu",    # deeper conv in layer4
    "layer4.1.bn2",    # deeper conv in layer4
    "avgpool",    # deeper conv in layer4
]

SUPPORTED_CONCEPTS = [concept for concept in os.listdir(CONCEPTS_DIR) if "random" not in concept]

def get_concept_significance(model:torch.nn.Module,layers:list[str],concept_name:str,class_idx:int,eval_images_path:str,device:str,num_rand_concepts:int=10):
    
    """
    used to check if the concept is significantly present in the model  or not for specified class

    returns:
    
    alignment : aggregated directional derivative score
    statistics bool :  true if it is statistically significant other wise false
    
    """


    assert all(l_name in SUPPORTED_LAYERS for l_name in layers), f"Expected all layers to be in {SUPPORTED_LAYERS}"
    
    assert concept_name in SUPPORTED_CONCEPTS , f"Expected concepts to be in {SUPPORTED_CONCEPTS}"
    
    if len(layers) > 1:
        raise ValueError("concept correction for more than one layer is not yet supported")



    target_concept = assemble_concept(concept_name, 0, concepts_path=CONCEPTS_DIR)


    concept_images = load_image_tensors(path=eval_images_path, transform=False)
    concept_tensors = torch.stack([transform(img) for img in concept_images])

    model = model.to(device)
    concept_tensors = concept_tensors.to(device)

    assert next(model.parameters()).device == concept_tensors.device, "Model and tensors on different devices!"



    random_concepts = [assemble_concept('random_' + str(i+0), (i+2)) for i in range(0, num_rand_concepts)]

    
     
    experimental_sets = [[target_concept, random_concept] for random_concept in random_concepts]


    mytcav = TCAV(model=model,
                layers=layers,
                layer_attr_method = LayerIntegratedGradients(
                model, None, multiply_by_inputs=False))



    mytcav.compute_cavs(experimental_sets, force_train=True)
    scores = mytcav.interpret(concept_tensors, experimental_sets, class_idx, n_steps=5,grad_kwargs={"create_graph": True, "retain_graph": True})

    # P1, P2, pval, relation = get_pval(scores, experimental_sets, layers[-1], "sign_count")

    assembled_scores = assemble_scores(scores,experimental_sets,0,score_layer=layers[-1],score_type="magnitude")

    mean_score = torch.stack(assembled_scores).mean()

    # print("mean_score ",mean_score)

    return mean_score

      

import matplotlib.pyplot as plt
import os
import math
import numpy as np


def eval_concept_significance(model:torch.nn.Module,layers:list[str],concept_name:str,class_idx:int,eval_images_path:str,save_path:str,device:str,num_rand_concepts:int=10,alpha:float=0.05):
    
    """
    used to check if the concept is significantly present in the model  or not for specified class

    returns:
    
    alignment : aggregated directional derivative score
    statistics bool :  true if it is statistically significant other wise false
    
    """


    assert all(l_name in SUPPORTED_LAYERS for l_name in layers), f"Expected all layers to be in {SUPPORTED_LAYERS}"
    
    assert concept_name in SUPPORTED_CONCEPTS , f"Expected concepts to be in {SUPPORTED_CONCEPTS}"
    

    target_concept = assemble_concept(concept_name, 100, concepts_path=CONCEPTS_DIR)
    concept_imgs = load_image_tensors(path=eval_images_path, transform=False)
    concept_tensors = torch.stack([transform(img) for img in concept_imgs]).to(device)


    random_concepts = [assemble_concept('random_' + str(i+0), (i+1)) for i in range(0, num_rand_concepts)]

    model.train()
    model = model.to(device)

    experimental_sets = []
    experimental_sets.extend([[target_concept, random_concept] for random_concept in random_concepts])

    tcav_obj= TCAV(model=model,
            layers=layers,
            layer_attr_method = LayerIntegratedGradients(
            model, None, multiply_by_inputs=False) )



    tcav_obj.compute_cavs(experimental_sets, force_train=True)
    scores = tcav_obj.interpret(concept_tensors, experimental_sets, class_idx, n_steps=5,grad_kwargs={"create_graph": True, "retain_graph": True})


    os.makedirs(save_path, exist_ok=True)

    n_layers = len(layers)
    ncols = math.ceil(math.sqrt(n_layers))
    nrows = math.ceil(n_layers / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if n_layers > 1 else [axes]

    results = {}

    for i, layer in enumerate(layers):


        P1, P2, pval, _ = get_pval(scores, experimental_sets, layer, "sign_count", alpha=alpha, print_ret=False)
        results[layer] = {"concept":float(np.mean([t.cpu().item() for t in P1])),"random":float(np.mean([t.cpu().item() for t in P2])),"pval":pval}


        ax = axes[i]
        bp = ax.boxplot(
            [P1, P2],
            labels=["Concept", "Random"],
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7, linewidth=1.5),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5)
        )

        ax.set_title(f"{layer}\n(p={pval:.3e})", fontsize=11)
        ax.set_ylabel("TCAV Score")
        ax.grid(True, linestyle="--", alpha=0.4)

        # Add significance marker
        max_val = np.nanmax([np.max(P1), np.max(P2)]) if len(P1) and len(P2) else 0
        if pval < 0.05:
            ax.text(1.5, max_val * 1.05 if max_val != 0 else 0.05, "*", ha='center', va='bottom', color='red', fontsize=18, fontweight='bold')

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"TCAV {concept_name=} {class_idx=}", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(save_path, f"tcav_{concept_name=}_{class_idx=}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined TCAV boxplot image at: {out_path}")

    return results

def train_correction_epoch(model:torch.nn.Module,layers:list[str],concept_name:str,class_idx:int,device:str,num_rand_concepts:int=10,max_epochs:int=1):

    assert all(l_name in SUPPORTED_LAYERS for l_name in layers), f"Expected all layers to be in {SUPPORTED_LAYERS}"
    
    assert concept_name in SUPPORTED_CONCEPTS , f"Expected concepts to be in {SUPPORTED_CONCEPTS}"
    
    if len(layers) > 1:
        raise ValueError("concept correction for more than one layer is not yet supported")

    device="cpu"

        
    target_concept = assemble_concept(concept_name, 100, concepts_path=CONCEPTS_DIR)
    concept_imgs = load_image_tensors(concept_name, transform=False)
    concept_tensors = torch.stack([transform(img) for img in concept_imgs]).to(device)

    random_concepts = [assemble_concept('random_' + str(i+0), (i+2)) for i in range(0, num_rand_concepts)]
    experimental_sets = []
    experimental_sets.extend([[target_concept, random_concept] for random_concept in random_concepts])


    model = model.to(device=device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=5e-2)

    tcav_obj= TCAV(model=model,
            layers=layers,
            layer_attr_method = LayerIntegratedGradients(
            model, None, multiply_by_inputs=False))



    tq = tqdm(range(max_epochs),f"Dibaising concept {concept_name} ...")
    for epoch in tq:


        tcav_obj.compute_cavs(experimental_sets, force_train=True)
        scores = tcav_obj.interpret(concept_tensors, experimental_sets, class_idx, n_steps=5,grad_kwargs={"create_graph": True, "retain_graph": True})
        P1,P2 ,pval, relation = get_pval(scores, experimental_sets, layers[-1], "sign_count")

        print(f"{pval=}")
        if relation != "Disjoint":

            print(f"Concept {concept_name} is not present")
            print(f"Training Stopped")
            
            break

        loss_scores = assemble_scores(scores,experimental_sets,0,score_layer=layers[-1],score_type="magnitude")
        loss = torch.stack(loss_scores).mean()

        if loss > 0 :

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            tq.set_postfix({"epoch":epoch,"pval":pval,"loss":format_float(loss.item())})
            

    return model



def generate_concept_alignment_report(model,device,save_path):


    assert device == "cpu", f"Concept evaluation without cpu is buggy"        


    concepts = ["CAT","CAT-TEXT","DOG","DOG-TEXT"]
      
    eval_dirs = ["/netscratch/aslam/TCAV/PetImages/train/with-text/Cat","/netscratch/aslam/TCAV/PetImages/train/with-text/Cat","/netscratch/aslam/TCAV/PetImages/train/with-text/Dog","/netscratch/aslam/TCAV/PetImages/train/with-text/Dog"]


    indicies = [0,1]
    results = {}

    for concept_name,dir_path in zip(concepts,eval_dirs):
        results[concept_name] = {}

        for idx in indicies:
            
            model = model.to(device)
            idx_result = eval_concept_significance(model=model, layers=["layer4.1.relu","avgpool"], concept_name=concept_name,class_idx=idx,eval_images_path=dir_path,device=device,save_path=save_path)            
            results[concept_name][idx] = idx_result


    return results

