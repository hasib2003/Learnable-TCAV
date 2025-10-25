import os
import torch
import torchvision.models as models

def get_model(name: str, num_classes: int):
    # check if the model exists in torchvision
    if not hasattr(models, name):
        raise ValueError(f"Model '{name}' not found in torchvision.models")

    # dynamically get the model constructor
    model_fn = getattr(models, name)

    # instantiate model without pretrained weights
    try:
        model = model_fn(weights=None)  # for torchvision >= 0.13
    except TypeError:
        model = model_fn(pretrained=False)  # fallback for older versions

    # replace classifier head depending on architecture
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        # ResNet, ResNeXt, etc.
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        # MobileNet, DenseNet, EfficientNet, etc.
        if isinstance(model.classifier, torch.nn.Linear):
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        elif isinstance(model.classifier, torch.nn.Sequential):
            last_idx = -1
            if isinstance(model.classifier[last_idx], torch.nn.Linear):
                in_features = model.classifier[last_idx].in_features
                model.classifier[last_idx] = torch.nn.Linear(in_features, num_classes)
            else:
                raise ValueError(f"Unsupported classifier type for model {name}")
        else:
            raise ValueError(f"Unsupported classifier type for model {name}")
    else:
        raise ValueError(f"Don’t know how to replace final layer for model {name}")

    return model


def load_from_checkpoint(model:torch.nn.Module,path:str,device:str):

    assert os.path.isfile(path), f"No file found at {path}"

    checkpoint = torch.load(path,map_location=device)
    
    assert "model_state_dict" in checkpoint,f"Checkpoint has no keys model_state_dict"

    model.load_state_dict(checkpoint["model_state_dict"])
    return model