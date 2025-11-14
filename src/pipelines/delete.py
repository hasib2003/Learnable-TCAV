

import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
from tqdm import tqdm

from utils.models import load_from_checkpoint, get_model, freeze_backbone, unfreeze
from utils.train import test, train_epoch
from utils.captum import get_concept_significance, get_CAV

from utils.common import setup_checkpoint_dir, save_checkpoint, save_results, setup_data_loaders
from utils.logger import Logger

from pipelines.args.train import args_train_with_concepts
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])


def optimizer_to(optim, device):
    for param in optim.state.values():
        for k, v in param.items():
            if isinstance(v, torch.Tensor):
                param[k] = v.to(device)


def scheduler_to(scheduler, device):
    """Move scheduler state to device if needed."""
    if hasattr(scheduler, 'state_dict'):
        for key, value in scheduler.state_dict().items():
            if isinstance(value, torch.Tensor):
                value.to(device)


def get_target_concepts(path: str):
    """Load target concepts from json file path."""
    concepts = None
    with open(path, 'r') as f:
        concepts = json.load(f)
    
    return concepts


def generate_concept_alignment_report(model, target_concepts, activation_layers: list[str], classifier, concepts_dir, device):
    reports = []

    for concept in target_concepts:
        concept_report = get_concept_significance(
            model=model,
            layers=activation_layers,
            classifier=classifier,
            concepts_dir=concepts_dir,
            concept_name=concept["name"],
            class_idx=concept["idx"],
            eval_images_path=concept["eval_dir_path"],
            trainable=False,
            score_type="sign_count",
            device=device
        )

        reports.append({"info": concept, "results": concept_report})
    
    return reports


def orthogonalize_row_grad(param: torch.nn.Parameter, row_idx: int, v: torch.Tensor, eps: float = 1e-8):
    """
    Modify param.grad in-place so that the gradient of the row `row_idx` is orthogonal to v.
    - param: 2D parameter (e.g. linear.weight) shape (out_features, in_features)
    - row_idx: index of the row to orthogonalize
    - v: 1D tensor with length == in_features
    """
    if param.grad is None:
        return

    g = param.grad[row_idx]           # shape: (in_features,)
    v = v.to(g.device).type_as(g)     # ensure same dtype/device

    v_norm = torch.norm(v)

    v = v / v_norm  # ensure that v is a unit vector

    if v_norm.item() <= eps:
        # v is (nearly) zero vector — nothing to remove
        print(f"CAV's norm is almost zero")
        return

    proj = torch.dot(g, v) * v           # scalar
    param.grad[row_idx] = g - proj    # in-place replacement (assign)

def orthogonalize_row_weight(param: torch.nn.Parameter, row_idx: int, v: torch.Tensor, eps: float = 1e-8):
    """
    Modify param.data in-place so that the row `row_idx` is orthogonal to v.
    - param: 2D parameter (e.g. linear.weight) shape (out_features, in_features)
    - row_idx: index of the row to orthogonalize
    - v: 1D tensor with length == in_features
    - eps: threshold for considering v as zero vector
    """
    if param.data is None:
        return
    
    with torch.no_grad():  # Don't track this operation in autograd
        
        w = param.data[row_idx]           # shape: (in_features,)
        v = v.to(w.device, dtype=w.dtype) # ensure same dtype/device
        
        # Check if v is nearly zero BEFORE normalization
        v_norm = torch.norm(v)
        if v_norm <= eps:
            print(f"Warning: CAV's norm ({v_norm.item():.2e}) is nearly zero, skipping orthogonalization")
            return
        
        # Normalize v to unit vector
        v_unit = v / v_norm
        
        wv = torch.dot(w, v_unit)
        proj = wv * v_unit
        
        # Orthogonalize in-place
        param.data[row_idx] = w - proj

def train_ortho_epoch(
    epoch,
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    target_concepts,
    classifier,
    concepts_dir,
    num_exps=10
):
    """
    Train one epoch with orthogonality constraint on one or more rows of the fc layer.
    Returns only Python primitives (floats/ints).
    """
    model = model.to(device)
    optimizer_to(optimizer, device)

    total_loss = 0.0
    correct = 0
    total = 0

    # --- Compute orthogonality constraints (CAVs) ---

    model.eval()
    print("Computing CAVs...")
    orthogonality_constraints = {}

    for cls_name, concept_config in target_concepts.items():
        print(f"# Processing {cls_name}")
        cavs = get_CAV(
            model,
            classifier=classifier,
            concepts_dir=concepts_dir,
            concept_name=concept_config["concept_name"],
            device=device,
            num_rand_concepts=num_exps,
        )
        mean_cav = torch.mean(cavs, dim=0).detach()
        orthogonality_constraints[concept_config["idx"]] = mean_cav

    
    # print("Pre Training Orthogonalization ...")
    # Apply orthogonality constraint to FC layer gradients
    # for class_idx, cav_tensor in orthogonality_constraints.items():
    #     orthogonalize_row_weight(model.fc.weight, class_idx, cav_tensor)

    model.train()
    # --- Training loop ---
    tq = tqdm(train_loader, desc=f"Epoch {epoch} | Constrained Training", unit="batch")

    for batch_idx, (inputs, targets) in enumerate(tq):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(inputs)
        loss = criterion(preds, targets)

        # Standard loss tracking
        total_loss += float(loss.detach().cpu().item())
        preds_class = preds.argmax(dim=1)
        batch_total = int(targets.size(0))
        batch_correct = int(preds_class.eq(targets).sum().cpu().item())

        total += batch_total
        correct += batch_correct

        # Backprop
        loss.backward()

        # Apply orthogonality constraint to FC layer gradients
        # for class_idx, cav_tensor in orthogonality_constraints.items():
        #     orthogonalize_row_grad(model.fc.weight, class_idx, cav_tensor)
        
        optimizer.step()

        for class_idx, cav_tensor in orthogonality_constraints.items():
            orthogonalize_row_grad(model.fc.weight, class_idx, cav_tensor)


        # Progress bar info
        batch_acc = batch_correct / batch_total if batch_total > 0 else 0.0
        tq.set_postfix_str(f"Acc {batch_acc:.2f} | Loss {float(loss.cpu().item()):.3f}")

    avg_loss = total_loss / (batch_idx + 1)
    avg_acc = 100.0 * correct / total if total > 0 else 0.0

    # Return only pure Python types
    return float(avg_loss), float(avg_acc)


def check_othogonality(
    model,
    target_concepts,
    classifier,
    concepts_dir:str,
    device:str,
    num_exps=10
):
    """
    computes the dot product between target CAVs and FC weights
    """
    
    model.eval()
    model = model.to(device)

    # --- Compute orthogonality constraints (CAVs) ---
    print("Computing CAVs...")
    orthogonality_stats = []

    for cls_name, concept_config in target_concepts.items():
        print(f"# Processing {cls_name}")
        cavs = get_CAV(
            model,
            classifier=classifier,
            concepts_dir=concepts_dir,
            concept_name=concept_config["concept_name"],
            device=device,
            num_rand_concepts=num_exps,
        )
        mean_cav = torch.mean(cavs, dim=0).detach().to(device)

        weight_vec = model.fc.weight[concept_config["idx"]]
        cosine_sim = torch.nn.functional.cosine_similarity(
            mean_cav.unsqueeze(0), weight_vec.unsqueeze(0)
        ).cpu().item()
        orthogonality_stats.append({"cosine_sim":cosine_sim,"info":concept_config})

        print(f"="*15,"before orthogonalization","="*15)
        print(f"\n{concept_config=}")
        print(f"{cosine_sim=}")
        print("="*50)


        orthogonalize_row_weight(model.fc.weight,0,mean_cav)
        
        cosine_sim = torch.nn.functional.cosine_similarity(
            mean_cav.unsqueeze(0), weight_vec.unsqueeze(0)
        ).cpu().item()
        orthogonality_stats.append({"cosine_sim":cosine_sim,"info":concept_config})

        print(f"="*15,"after orthogonalization","="*15)
        print(f"\n{concept_config=}")
        print(f"{cosine_sim=}")
        print("="*50)




    return orthogonality_stats


def main():


    resume_checkpoint = "/netscratch/aslam/TCAV/text-inflation/bechmarking-ortho/with-concept-loss/20251103_220120/best_model.pth"
    concept_config_train = "config/ortho/config.json"
    concept_config_test  = "config/concept_test.json"
    concepts_dir  = "/netscratch/aslam/TCAV/PetImages/Concepts/"    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print config

    # compute max key length for neat alignment
    model = get_model("resnet18", 2,False)



    assert os.path.isfile(resume_checkpoint), f"Invalid checkpoint path"
    
    chkpnt = torch.load(resume_checkpoint, map_location=device)

    model.load_state_dict(chkpnt["model_state_dict"])
    

    model = model.to(device)

    for idx,module in enumerate(model.named_children()):
        print(f"idx :{module}")
    
    exit()

    # assert os.path.isfile(args.concept_config_train), f"Invalid path to config file"
    target_concepts = get_target_concepts(concept_config_train)

    # stats = {
    #     "train_acc": [],
    #     "train_loss": [],
    #     "test_acc": [],
    #     "test_loss": [],
    #     "learning_rate": []
    # }

    test_concepts = get_target_concepts(path=concept_config_test)
    activation_layers = ["avgpool"]



    ortho_stats = check_othogonality(model,target_concepts,None,concepts_dir,device)


    concepts_report = generate_concept_alignment_report(
        model=model,
        target_concepts=test_concepts,
        activation_layers=activation_layers,
        classifier=None,
        concepts_dir=concepts_dir,
        device=device,
    )

    print(f"{ortho_stats=}")
    print(f"{concepts_report=}")

    # Save results
    # save_results(checkpoint_dir, train_acc, test_acc, all_labels, all_preds,
    #             test_dataset.classes, test_dataset.class_to_idx, concepts_report, stats,ortho_stats)

    # print(f"Saved to .. {checkpoint_dir}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()