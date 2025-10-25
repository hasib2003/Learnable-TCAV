import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import argparse
from tqdm import tqdm

from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys

this_file = os.path.abspath(__file__)

parent_parent = os.path.dirname(os.path.dirname(this_file))

if parent_parent not in sys.path:
    sys.path.insert(0, parent_parent)

from utils.models import load_from_checkpoint , get_model
from src.concept_correction import generate_concept_alignment_report,get_concept_significance

def test(model, loader, criterion, device,return_preds=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if return_preds:
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(predicted.cpu().tolist())

    if return_preds:
        return total_loss / len(loader), 100. * correct / total, all_labels,all_preds

    
    return total_loss / len(loader), 100. * correct / total



def evaluate_concepts(model,dataset_path):

    device = "cpu"

    activation_layer = "avgpool"

   
    target_concepts = [        
        
        {"eval_dir_path":os.path.join(dataset_path,"Cat"),"name":"CAT","idx":0,"desired":True},
        
        {"eval_dir_path":os.path.join(dataset_path,"Cat"),"name":"CAT-TEXT","idx":0,"desired":False},
        
        {"eval_dir_path":os.path.join(dataset_path,"Dog"),"name":"DOG","idx":1,"desired":True},
        
        {"eval_dir_path":os.path.join(dataset_path,"Dog"),"name":"DOG-TEXT","idx":1,"desired":False}
    ]
    
    
    model.train()


    

    concept_losses = {}

    for concept in target_concepts:



        mean_align_val = get_concept_significance(model=model, layers=[activation_layer], concept_name=concept["name"],class_idx=concept["idx"],eval_images_path=concept["eval_dir_path"],device=device)


        concept_losses[concept["name"]] = {}
        concept_losses[concept["name"]]["raw"] = mean_align_val.cpu().item().__round__(3)

        print(f"{concept['name']=}") 


    return concept_losses



def main():
    # Argument parser

    parser = argparse.ArgumentParser(description='Test the Resnet 18 from checkpoint')
    parser.add_argument('--dataset_dir', type=str, default='data', help='Path to data directory to test')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of Classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--checkpoint_path', type=str,required=True, help='Path to checkpoint to test on')
    parser.add_argument('--num_workers', type=int,default=2, help='Num dataloader workers')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Configuration: {vars(args)}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # Model, loss, optimizer
    model = get_model("resnet18",args.num_classes)
    model = model.to(device)
    
    assert os.path.isfile(args.checkpoint_path), f"No file found at {args.checkpoint_dir}"

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20250930_141537
    base_dir = os.path.join(os.path.dirname(args.checkpoint_path),f"Test-{current_time}")
    os.makedirs(base_dir,exist_ok=True)


    print(f"\nTesting on test dataset")
    print(f"loading best model ...")


    criterion = nn.CrossEntropyLoss()
    
    model = load_from_checkpoint(model,path=args.checkpoint_path,device=device)

    test_dataset = datasets.ImageFolder(args.dataset_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)

    test_loss, test_acc,all_labels,all_preds = test(model, test_loader, criterion, device,return_preds=True)
    concept_reports = generate_concept_alignment_report(model,device="cpu",save_path=base_dir)
    concept_maginitudes  = evaluate_concepts(model,args.dataset_dir)




    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    class_names = test_dataset.classes  

    cm = confusion_matrix(all_labels, all_preds)
    cm_list = cm.tolist()  

    # 2. Save metrics to JSON
    result_dict = {
        "accuracy": test_acc,
        "confusion_matrix": cm_list,
        "class_names": class_names,
        "class_2_idx":test_dataset.class_to_idx,
        "concept_sign":concept_reports,
        "concept_magnitudes":concept_maginitudes
    }

    with open(os.path.join(base_dir, f"summary.json"), "w") as f:
        json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    main()