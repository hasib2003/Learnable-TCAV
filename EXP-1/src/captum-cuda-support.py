# for checking and in shah Allah making TCAV compatible with Cuda


import os
import sys

# ..........torch imports............
import torch
import torch.optim as optim

from gptClf import TorchClassifier

#.... Captum imports..................
from captum.attr import  LayerIntegratedGradients
from captum.concept import TCAV
#.... Custom imports..................

from tqdm import tqdm

current_file = os.path.abspath(__file__)
parent_parent = os.path.dirname(os.path.dirname(current_file))
if parent_parent not in sys.path:
    sys.path.insert(0, parent_parent)

from utils.models import get_model
from utils.captum import transform, load_image_tensors, assemble_concept , assemble_scores,get_pval

from fastcav import FastCAVCaptumClassifier


import statistics

def run_instance(classifier="default"):

    CONCEPTS_DIR = "/netscratch/aslam/TCAV/PetImages/Concepts/"
    CHECKPOINT_PATH="/netscratch/aslam/TCAV/text-inflation/EXP1/20251001_204742/best_model.pth"


    DUMP_PATH= os.path.join(os.path.dirname(CHECKPOINT_PATH),"cav-testing")
    os.makedirs(DUMP_PATH,exist_ok=True)


    cat_text_concept = assemble_concept("CAT-TEXT", 1, concepts_path=CONCEPTS_DIR)

    model = get_model(name="resnet18",num_classes=2)


    layers = [
        "layer2.1.relu",   # deeper conv in layer2
        "layer3.1.relu",   # deeper conv in layer3
        "layer4.1.relu",    # deeper conv in layer4
    ]
    


    clf     = None
    DEVICE  = "cpu"
    # DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

    if classifier == "custom":
    
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        clf = TorchClassifier(device=DEVICE,epochs=100)
    
    if classifier == "fastcav":
        clf = FastCAVCaptumClassifier()
        DEVICE = "cpu"
    

    print(f"{DEVICE=}")
    print(f"{classifier=}")


    model.to(DEVICE)
    population = []

    for i in range(10):
        
        print(f"Instance={i}")
        
        mytcav = TCAV(model=model,
                    layers=layers,
                    classifier=clf,
                    layer_attr_method = LayerIntegratedGradients(
                    model, None, multiply_by_inputs=False))

        # Load sample images from folder
        cat_imgs = load_image_tensors('/netscratch/aslam/TCAV/PetImages/train/without-text/Cat', transform=False)

        # Load sample images from folder
        cat_img_tensors = torch.stack([transform(img) for img in cat_imgs]).to(DEVICE)

        cat_idx = 0
        n = 10


        random_concepts = [assemble_concept('random_' + str(i+0), (i+2)) for i in range(0, n)]
        experimental_sets = [[cat_text_concept, random_concept] for random_concept in random_concepts]


        mytcav.compute_cavs(experimental_sets, force_train=True)
        scores = mytcav.interpret(cat_img_tensors, experimental_sets, cat_idx, n_steps=5)

        # print("scores ",scores)

        # scores = [score.cpu() for score in scores]

        #  checking if the scores are statistically significant
        P1, P2, pval, relation = get_pval(scores, experimental_sets, layers[-1], "sign_count")


        loss_scores = assemble_scores(scores,experimental_sets,0,score_layer=layers[-1],score_type="magnitude")

        print(f"{pval=}")
        print(f"{relation=}")

        loss = torch.stack(loss_scores).mean().cpu().item()
        print(f"{loss=}")

        population.append(loss)

    print(f"{population=}")
    print(f"mean ",statistics.mean(population))
    print(f"std_population ",statistics.pstdev(population))
    print(f"std ",statistics.stdev(population))


def main():

    import sys
    

    if len(sys.argv) < 2:
        print(" No classifier specified. Using default: 'none'")
        classifier = "none"
    else:
        classifier = sys.argv[1]

    run_instance(classifier)

if __name__ == "__main__":
    main()
