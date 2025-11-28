import torch
import torch.nn as nn
import os
import json
from utils.models import  get_model
from utils.captum import get_concept_significance, get_CAV
from pipelines.args.eval import args_evaluate_tcav
from torchvision import transforms
import json
import csv           

class ResNet18Partial(nn.Module):
    """
    Hardcoded partial forward:
    start: layer3.1.relu output (activation already given)
    end:   avgpool output
    """

    def __init__(self, full_model: nn.Module,start_layer=4):
        super().__init__()

        # We only need the modules AFTER layer3.1.relu:
        # which is:
        # - layer3[1].conv2 was already consumed
        # - layer4 (all)
        # - avgpool
        # - flatten (but NOT fc)
        self.start_layer    = start_layer
        self.layer3         = full_model.layer3
        self.layer4         = full_model.layer4
        self.avgpool        = full_model.avgpool

    def forward(self, x):
        """
        x is expected to be the activation from layer3.1.relu.
        """

        if self.start_layer < 3:
            # layer 2 
            x = self.layer3(x)
            
        if self.start_layer < 4:
            x = self.layer4(x)

        x = self.avgpool(x)

        return x

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

def get_target_concepts(path: str):
    """Load target concepts from json file path."""
    concepts = None
    with open(path, 'r') as f:
        concepts = json.load(f)
    
    return concepts

def generate_concept_alignment_report(model, target_concepts, activation_layers: list[str], classifier, concepts_dir,random_prefix, device):
    

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
            device=device,
            random_prefix=random_prefix,
        )

        reports.append({"info": concept, "results": concept_report})
    
    return reports

def generate_concept_report_fast(model, target_concepts, activation_layers: list[str], classifier, concepts_dir, random_prefix,device,num_exps=10):

    reports = []    
    for concept in target_concepts:
        all_cavs = get_CAV(
                        model=model,
                        classifier=classifier,
                        concepts_dir=concepts_dir,
                        concept_name=concept["name"],
                        layers=activation_layers,
                        num_rand_concepts=num_exps,
                        device=device,
                        random_prefix=random_prefix,
                        force_train=True
                        )

        layer_2_stats = {}
            
        for layer,layer_cavs in all_cavs.items():

            if layer == "avgpool":
                projected_cavs = layer_cavs
            elif layer == "layer2.1.relu":
                print(f"{layer=}")
                print(f"{layer_cavs.shape=}")
                projected_cavs = ResNet18Partial(model,start_layer=2)(layer_cavs.reshape(num_exps,128,28,28)).squeeze().squeeze()
            
            elif layer == "layer3.1.relu":
                # print(f"{layer=}")
                # print(f"{layer_cavs.shape=}")
                projected_cavs = ResNet18Partial(model,start_layer=3)(layer_cavs.reshape(num_exps,256,14,14)).squeeze().squeeze()
                # print(f"{projected_cavs.shape=}")
            elif layer == "layer4.1.relu":
                # print(f"{layer=}")
                # print(f"{layer_cavs.shape=}")
                projected_cavs = ResNet18Partial(model,start_layer=4)(layer_cavs.reshape(num_exps,512,7,7)).squeeze().squeeze()
                # print(f"{projected_cavs.shape=}")

            else:
                raise ValueError(f"Logic is not defined for layer {layer}")
    
            dot_product = projected_cavs @ model.fc.weight[concept["idx"]].reshape(-1,1)

            pos_counts = torch.sum(dot_product > 0).cpu().item()
            neq_counts = num_exps - pos_counts

            layer_2_stats[layer] = {"concept":pos_counts/num_exps,"random":neq_counts/num_exps}
        
        reports.append({"info": concept, "results": layer_2_stats})

    return reports



def handle_concept_generation(mode,**kwargs):
    """"""
    
    valid_modes = ["exact","approx"]

    assert mode in valid_modes, f"{mode=} not in {valid_modes=}" 

    if mode == "exact":
        return generate_concept_alignment_report(**kwargs)
    
    elif mode == "approx":
        return generate_concept_report_fast(**kwargs)
    


def main():
    
    args = args_evaluate_tcav()

    num_classes = args.num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print config
    print(f"{'='*20} Config {'='*20}")
    args_dict = vars(args)
    max_key_len = max(len(k) for k in args_dict.keys())
    for key, val in args_dict.items():
        print(f"{key:<{max_key_len + 3}}: {val}")
    print(f"{'='*50}")


    assert os.path.isfile(args.checkpoint), f"No file found at {args.checkpoint}"
    os.makedirs(args.save_dir,exist_ok=True)

        
    model = get_model(args.model, num_classes, True)
    # chkpnt = torch.load(args.checkpoint, map_location=device)
    # model.load_state_dict(chkpnt["model_state_dict"])
    print(f"Weights loaded successfully ")

    model = model.to(device)


    assert os.path.isfile(args.concept_config), f"Invalid path to concept config file"
    target_concepts = get_target_concepts(args.concept_config)


    if args.mode != "all":
        modes = [args.mode]
    else:
        modes = ["approx","exact"]

    for mode in modes:

        concepts_report = handle_concept_generation(
            mode=mode,
            model=model,
            target_concepts=target_concepts,
            activation_layers=args.target_layers,
            classifier=args.classifier,
            concepts_dir=args.concepts_dir,
            random_prefix=args.random_prefix,
            device=device,
        )
        
        lines = []
        if concepts_report:
            for experiement in concepts_report:
        
                eval_dir = experiement["info"]["eval_dir_path"]  
                exp_config = [args.model,experiement["info"]["name"],mode,experiement["info"]["idx"],eval_dir]

                for layer_name,layer_results in experiement["results"].items():
                    
                    concept_score = layer_results["concept"]
                    random_score = layer_results["random"]
                    p_val = layer_results.get("pval",-1)
                    
                    line = exp_config.copy()
                    line.extend([layer_name,concept_score,random_score,p_val])

                    lines.append(line)
        
        
        save_path = os.path.join(args.save_dir,f"results.csv")
        with open(save_path, "a", newline="", encoding="utf-8") as f:
            
            writer = csv.writer(f)
            writer.writerows(lines)
            print(f"Saved to .. {save_path}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()