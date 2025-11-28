import os
import csv           
import copy
import torch
import torch.nn as nn

from utils.models import  get_model
from utils.train import test

from pipelines.args.linearity import args_linearity_check
from torchvision import transforms , datasets

NONLINEAR = (
    nn.ReLU, nn.ReLU6,
    nn.LeakyReLU, nn.ELU,
    nn.GELU, nn.Sigmoid,
    nn.Tanh, nn.SELU,
    nn.SiLU, nn.Mish,
    nn.Hardsigmoid, nn.Hardtanh,
    nn.Hardswish, nn.Softplus,
)

def clone_with_identity(model: nn.Module):
    # First, shallow copy the module itself
    new_model = copy.copy(model)
    # Now process its children
    for name, child in model.named_children():
        if isinstance(child, NONLINEAR):
            # Replace nonlinearity with Identity
            setattr(new_model, name, nn.Identity())
        else:
            # Recursively clone submodules
            setattr(new_model, name, clone_with_identity(child))

    return new_model

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

def main():

    num_classes = 2
    
    args = args_linearity_check()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print config
    print(f"{'='*20} Config {'='*20}")
    args_dict = vars(args)
    max_key_len = max(len(k) for k in args_dict.keys())
    for key, val in args_dict.items():
        print(f"{key:<{max_key_len + 3}}: {val}")
    print(f"{'='*50}")


    assert os.path.isfile(args.checkpoint), f"No file found at {args.checkpoint}"
    os.makedirs(args.save_dir,exist_ok=True)

        
    model = get_model(args.model, num_classes, False)
    chkpnt = torch.load(args.checkpoint, map_location=device)
    
    dataset = datasets.ImageFolder(root=args.data_dir,transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False)

    model.load_state_dict(chkpnt["model_state_dict"])
    print(f"Weights loaded successfully ")

    criterion = nn.CrossEntropyLoss()

    print(f"Using device: {device}")

    test_acc,test_loss = test(model,dataloader,criterion,device,return_preds=False)    
    print(f"Vanilla {test_acc=} {test_loss=}")

    model.layer4 = clone_with_identity(model.layer4)
    test_acc,test_loss = test(model,dataloader,criterion,device,return_preds=False)    
    print(f"Linearized {test_acc=} {test_loss=}")




    # save_path = os.path.join(args.save_dir,f"results.csv")
    # with open(save_path, "a", newline="", encoding="utf-8") as f:
        
    #     writer = csv.writer(f)
    #     writer.writerows(lines)
    #     print(f"Saved to .. {save_path}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()