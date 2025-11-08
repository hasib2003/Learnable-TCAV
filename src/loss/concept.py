
import torch

class ConceptLoss(torch.nn.Module):
    def __init__(self):
        super(ConceptLoss, self).__init__()

    def forward(self, x,desired:bool):
        
        if desired:
            return -torch.sigmoid(x) + 0.5

        return torch.sigmoid(x) - 0.5