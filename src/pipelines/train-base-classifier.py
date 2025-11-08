import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import argparse

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
from utils.train import train_epoch,test
from src.concept_correction import generate_concept_alignment_report


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Train ResNet-18 on image classification')
    parser.add_argument('--train_dir', type=str, default='data', help='Path to train data directory')
    parser.add_argument('--test_dir', type=str, default='data', help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
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

    # Load dataset
    full_dataset = datasets.ImageFolder(args.train_dir, transform=transform)
    num_classes = len(full_dataset.classes)
    print(f"Found {num_classes} classes: {full_dataset.classes}")
    
    # Train/test split
    train_size = int(args.train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    print(f"Train size: {train_size}, Test size: {test_size}")
    
    # Model, loss, optimizer
    model = get_model("resnet18",num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20250930_141537
    checkpoint_dir = os.path.join(args.checkpoint_dir, current_time)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, val_acc = test(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {test_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_loss': test_loss,
            }, checkpoint_path)
            print(f"Saved best model (Test Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining complete! Best test accuracy: {best_val_acc:.2f}%")
    
    print(f"\nTesting on test dataset")
    print(f"loading best model ...")

    
    model = load_from_checkpoint(model,path=checkpoint_path,device=device)

    test_dataset = datasets.ImageFolder(args.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)

    test_loss, test_acc,all_labels,all_preds = test(model, test_loader, criterion, device,return_preds=True)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    class_names = test_dataset.classes   # <-- critical

    # test_loss, test_acc, all_labels, all_preds = test(...)

    # 1. Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_list = cm.tolist()  # JSON needs a serializable format
    concepts_report = generate_concept_alignment_report(model,save_path=checkpoint_dir,device="cpu")



    # 2. Save metrics to JSON
    result_dict = {
        "train_acc": train_acc,
        "val_acc": val_acc,
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
    



if __name__ == "__main__":
    main()