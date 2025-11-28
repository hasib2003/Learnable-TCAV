import os
import torch
import torchvision.models as models
import torch

def freeze_backbone(model):
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final fully connected layer
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Model does not have a standard fc layer — check architecture.")
    
    return model
def unfreeze(model):
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = True

    return model



def get_model(name: str, num_classes: int | None, pretrained: bool = False):
    # Check if model exists
    if not hasattr(models, name):
        raise ValueError(f"Model '{name}' not found in torchvision.models")

    model_fn = getattr(models, name)

    # Handle pretrained argument properly for newer/older torchvision
    try:
        if pretrained:
            # Try to get default weights (torchvision >= 0.13)
            weights_enum = getattr(models, f"{name}_Weights", None)
            weights = weights_enum.DEFAULT if weights_enum is not None else None
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)
    except TypeError:
        # Fallback for older versions
        model = model_fn(pretrained=pretrained)

    
    if num_classes is None:
        print(f"Loaded model with default fc")
        return model


    # Replace classifier head depending on architecture
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        # e.g. ResNet, ResNeXt
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, torch.nn.Linear):
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        elif isinstance(model.classifier, torch.nn.Sequential):
            last_idx = -1
            if isinstance(model.classifier[last_idx], torch.nn.Linear):
                in_features = model.classifier[last_idx].in_features
                model.classifier[last_idx] = torch.nn.Linear(in_features, num_classes)
            else:
                raise ValueError(f"Unsupported classifier structure for model '{name}'")
        else:
            raise ValueError(f"Unsupported classifier structure for model '{name}'")
    else:
        raise ValueError(f"Cannot identify classifier layer for model '{name}'")

    return model

def load_from_checkpoint(model:torch.nn.Module,path:str,device:str):

    assert os.path.isfile(path), f"No file found at {path}"

    checkpoint = torch.load(path,map_location=device)
    
    assert "model_state_dict" in checkpoint,f"Checkpoint has no keys model_state_dict"

    model.load_state_dict(checkpoint["model_state_dict"])
    return model