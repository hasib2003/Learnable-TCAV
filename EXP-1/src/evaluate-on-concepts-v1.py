
import os
import sys

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

from utils.models import load_from_checkpoint , get_model
from utils.captum import transform, load_image_tensors, assemble_concept , assemble_scores,get_pval


CONCEPTS_DIR = "/netscratch/aslam/TCAV/PetImages/Concepts/"
CHECKPOINT_PATH="/netscratch/aslam/TCAV/text-inflation/EXP1/20251001_204742/best_model.pth"


DUMP_PATH= os.path.join(os.path.dirname(CHECKPOINT_PATH),"cav-testing")
os.makedirs(DUMP_PATH,exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

cat_img_concept = assemble_concept("CAT", 0, concepts_path=CONCEPTS_DIR)
cat_text_concept = assemble_concept("CAT-TEXT", 1, concepts_path=CONCEPTS_DIR)

model = get_model(name="resnet18",num_classes=2)
model = load_from_checkpoint(model,path=CHECKPOINT_PATH,device=DEVICE)

optimizer = optim.Adam(model.parameters(), lr=5e-2)
model = model.train()
model.to(DEVICE)

layers = [
    "layer2.1.relu",   # deeper conv in layer2
    "layer3.1.relu",   # deeper conv in layer3
    "layer4.1.relu"    # deeper conv in layer4
]


mytcav = TCAV(model=model,
              layers=layers,
              layer_attr_method = LayerIntegratedGradients(
                model, None, multiply_by_inputs=False))


# Load sample images from folder
cat_imgs = load_image_tensors('CAT', transform=False)
text_imgs = load_image_tensors('CAT-TEXT', transform=False)

# Load sample images from folder
cat_img_tensors = torch.stack([transform(img) for img in cat_imgs])
cat_text_tensors = torch.stack([transform(img) for img in text_imgs]).to(DEVICE)

cat_idx = 0


n = 10

print("cat_text_tensors[0] ",cat_text_tensors[0])

random_concepts = [assemble_concept('random_' + str(i+0), (i+2)) for i in range(0, n)]

# print(random_concepts)

# experimental_sets = [[cat_img_concept, random_0_concept], [cat_img_concept, random_1_concept]]
experimental_sets = []
experimental_sets.extend([[cat_text_concept, random_concept] for random_concept in random_concepts])

# print("experimental_sets ",len(experimental_sets) )


# Run TCAV

for epoch in tqdm(range(10),"Dibaising concept TEXT ..."):



    # scores = mytcav.interpret(cat_text_tensors, experimental_sets, cat_idx, n_steps=5)
    
    print("cat_text_tensors.device ",cat_text_tensors.device)
    mytcav.compute_cavs(experimental_sets, force_train=True)
    scores = mytcav.interpret(cat_text_tensors, experimental_sets, cat_idx, n_steps=5,grad_kwargs={"create_graph": True, "retain_graph": True})
    print("scores ",scores["1-2"]["layer2.1.relu"]["magnitude"])
    # print("scores ",scores["1-2"]["layer2.1.relu"]["magnitude"][0].backward())
    # optimizer.step()
    # exit()

    #  checking if the scores are statistically significant
    P1, P2, pval, relation = get_pval(scores, experimental_sets, layers[-1], "sign_count")

    # print("pval ",pval)

    if relation != "Disjoint":
        print("concepts are not statistically significant")    
    else:
        print("concepts are statistically significant")

        loss_scores = assemble_scores(scores,experimental_sets,0,score_layer=layers[-1],score_type="magnitude")
        
        # print("loss_scores ",loss_scores)

    
        loss = torch.stack(loss_scores).mean()
        # if loss < 0.001 :
        #     break 



        print("loss ",loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        


        # loss = 

checkpoint_path = os.path.join(DUMP_PATH, "text_unbaised_model.pth")

torch.save({
            'epoch': -1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': -1,
            'test_loss': -1,
        },checkpoint_path)


print(f"saved model to {checkpoint_path}")