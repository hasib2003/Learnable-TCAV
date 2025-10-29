import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import argparse
from tqdm import tqdm

from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

from utils.models import load_from_checkpoint , get_model
from utils.train import test,train_epoch
from utils.logger import Logger
from utils.concept_correction import generate_concept_alignment_report,get_concept_significance
from loss.concept import ConceptLoss


def freeze_until(model: nn.Module, layer_name: str):
    """
    Freeze all layers up to and including `layer_name`.
    Works for nested modules and any torchvision ResNet variant.
    """
    freeze = True
    for name, child in model.named_children():
        if freeze:
            for param in child.parameters():
                param.requires_grad = False

        # Recursively check children (because ResNet is nested)
        if name == layer_name:
            freeze = False
        else:
            freeze_until(child, layer_name)

def optimizer_to(optim, device):
    for param in optim.state.values():
        for k, v in param.items():
            if isinstance(v, torch.Tensor):
                param[k] = v.to(device)


def train_epoch_with_concept(epoch,model, train_loader, test_loader, criterion, optimizer, device,alpha=0.5,patience=30):

    activation_layer = "avgpool"

    concept_loss = ConceptLoss()


    target_concepts = [        
        
        {"eval_dir_path":os.path.join(args.train_dir,"Cat"),"name":"CAT","idx":0,"desired":True},
        {"eval_dir_path":os.path.join(args.train_dir,"Cat"),"name":"CAT-TEXT","idx":0,"desired":False},        
        
        {"eval_dir_path":os.path.join(args.train_dir,"Dog"),"name":"CAT-TEXT","idx":1,"desired":False},        
        {"eval_dir_path":os.path.join(args.train_dir,"Dog"),"name":"DOG","idx":1,"desired":True},

        # {"eval_dir_path":os.path.join(args.train_dir,"Cat"),"name":"DOG","idx":0,"desired":False},

        
        # {"eval_dir_path":os.path.join(args.train_dir,"Dog"),"name":"DOG-TEXT","idx":1,"desired":False},
        # {"eval_dir_path":os.path.join(args.train_dir,"Cat"),"name":"DOG-TEXT","idx":0,"desired":False},
    ]
    
    
    model.train()
    # freeze_until(model,activation_layer)

    total_loss = 0
    correct = 0
    total = 0

    triggers = 0

        
    model = model.to(device)
    optimizer_to(optimizer, device)
    
    tq = tqdm(train_loader, desc="Training")
    
    for batch_idx,(images, labels) in enumerate(tq):

        images, labels = images.to(device), labels.to(device)
        model = model.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        concept_losses = {}
        c_losses = []

        for concept in target_concepts:

            mean_align_val = get_concept_significance(model=model, layers=[activation_layer], concept_name=concept["name"],class_idx=concept["idx"],eval_images_path=concept["eval_dir_path"],device=device)

            key = f'{concept["name"]}_{concept["idx"]}'

            concept_losses[key] = {}
            concept_losses[key]["raw"] = mean_align_val.cpu().item().__round__(3)

                        
            if (concept["desired"] and mean_align_val < 0) or (not concept["desired"] and mean_align_val > 0):
                # there is some loss 
                
                # print("mean_align_val ",mean_align_val)
                c_loss = concept_loss(mean_align_val,concept["desired"])
                # print(f"{c_loss=}")        

                c_losses.append(c_loss)
                # loss = ((1 - alpha) * loss) + (alpha * c_loss)
                concept_losses[key]["loss"] = c_loss.cpu().item().__round__(3)
                concept_losses[key]["desired"] = concept["desired"]
        

        total_c_loss = sum(c_losses)

        if len(c_losses) != 0:
            total_c_loss = total_c_loss / len(c_losses)

        if total_c_loss > 1e-4:
            loss = (1-alpha) * loss + alpha * total_c_loss
            print("\nConcept Losses ", concept_losses)
            print(f"{total_c_loss=}")

            

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)

        b_size = labels.size(0)
        b_corr = predicted.eq(labels).sum().item()
        total += b_size
        correct += b_corr

        tq.set_postfix_str(f"Acc: {b_corr/b_size:.4f}   loss: {loss.cpu().item():.4f}")

        if batch_idx % 30 == 0:
            test_loss, test_acc = test(model,test_loader,criterion,device)
            print(f"{test_acc=} {test_loss=}")


            # saving the latest model weights
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, last_checkpoint_path)
            
            # Save best model
            global best_test_acc

            if test_acc > best_test_acc:
                
                best_test_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'test_loss': test_loss,

                }, best_checkpoint_path)
                print(f"Saved best model (Best  Acc: {test_acc:.2f}%)")
        
    return total_loss / (batch_idx+1), 100. * correct / total
    

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Train ResNet-18 on image classification')
    parser.add_argument('--train_dir', type=str, default='data', help='Path to train data directory')
    parser.add_argument('--test_dir', type=str, default='data', help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--checkpoint_dir', type=str, default='checkoints', help='Directory to save checkpoints')
    parser.add_argument('--resume_checkpoint', type=str, help='Checkpoint path to load model from')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    parser.add_argument('--train_strategy', type=str, default="joint_optimization",choices=['joint_optimization', 'n_batch_alignment','none'])
    parser.add_argument('--target_layer', type=str, default="avgpool",choices=['avgpool'])




    parser.add_argument('--n', type=int, default=10, help='no of batches to align for, required only if strategy is n_batch_alignment')

    #important
    parser.add_argument('--c_corr_after', type=int, default=2, help='Number of epochs after which correction will be triggered through LCAV')



    global args
    args = parser.parse_args()  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dict_args = vars(args)
    print(f"Configuration: {dict_args}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(args.train_dir, transform=transform)
    num_classes = len(full_dataset.classes)
    print(f"Found {num_classes} classes: {full_dataset.classes}")
    
    # Train/test split
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, _ = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)


    test_dataset = datasets.ImageFolder(args.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {len(test_dataset)}")
    
    # Model, loss, optimizer
    model = get_model("resnet18",num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        
        chkpnt = torch.load(args.resume_checkpoint,map_location=device)

        model.load_state_dict(chkpnt["model_state_dict"])
        optimizer.load_state_dict(chkpnt["optimizer_state_dict"])


        print(f"Loaded model/optim weights from {args.resume_checkpoint}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)



    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20250930_141537
    checkpoint_dir = os.path.join(args.checkpoint_dir, current_time)
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(dict_args, f, indent=4)


    Logger(log_file=os.path.join(checkpoint_dir,"log.txt")).enable()

    
    # Training loop

    global best_checkpoint_path,last_checkpoint_path, best_test_acc
    
    
    best_test_acc = 0.0
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(checkpoint_dir, "last_model.pth")


  
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")         

        if (epoch+1) % args.c_corr_after == 0:            
    
            if args.train_strategy == "joint_optimization":
                train_loss,train_acc = train_epoch_with_concept(epoch,model, train_loader,test_loader, criterion, optimizer, device)

            elif args.train_strategy == "n_batch_alignment":
                train_loss,train_acc = train_n_batch_alignment(model, train_loader, criterion, optimizer, device,n=args.n)

            else:
                print(f"Unsupported training strategy passed {args.train_strategy}, CE loss only")
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        else:
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        #val_loss, val_acc = test(model, val_loader, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        #print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")



        # saving the latest model weights
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'test_loss': test_loss,
        }, last_checkpoint_path)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'val_acc': val_acc,
                #'val_loss': val_loss,
                'test_acc': test_acc,
                'test_loss': test_loss,

            }, best_checkpoint_path)
            print(f"Saved best model (Best  Acc: {test_acc:.2f}%)")
    
    print(f"\nTraining complete! Best test accuracy: {best_test_acc:.2f}%")
    
    print(f"\nTesting on test dataset")
    print(f"loading best model ...")

    
    model = load_from_checkpoint(model,path=best_checkpoint_path,device=device)



    test_loss, test_acc,all_labels,all_preds = test(model, test_loader, criterion, device,return_preds=True)
    # print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%", flush=True)

    
    class_names = test_dataset.classes   # <-- critical

    # test_loss, test_acc, all_labels, all_preds = test(...)

    # 1. Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_list = cm.tolist()  # JSON needs a serializable format
    concepts_report = generate_concept_alignment_report(model,device=device,save_path=checkpoint_dir)

    # 2. Save metrics to JSON
    result_dict = {
        "train_acc": train_acc,
        # "val_acc": val_acc,
        "test_acc": test_acc,
        "confusion_matrix": cm_list,
        "class_names": class_names,
        "class_2_idx":test_dataset.class_to_idx,
        "concept_report":concepts_report
    }
    with open(os.path.join(checkpoint_dir, "results.json"), "w") as f:
        json.dump(result_dict, f, indent=4)

    # 3. Plot and save confusion matrix with class labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "confusion_matrix.png"))
    plt.close()

    print(f"Results + labeled confusion matrix saved to {checkpoint_dir}")
    print("cm ",cm)
    
    # print("concept_report",concept_report)
    



if __name__ == "__main__":

    import warnings
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    main()
