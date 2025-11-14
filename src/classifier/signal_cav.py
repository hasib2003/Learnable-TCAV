# replacement of SKSG liner model, using cuda support



import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Union
from captum.concept._utils.classifier import Classifier

class SignalCav(Classifier):
    r"""
    Signal-based Concept Activation Vector (Signal-CAV) implementation.
    Computes pattern vectors (h_pat) via covariance between activations and
    concept labels, following Haufe et al. (2014) and Eq.(3) in the
    Pattern-based CAV paper.

    The convention is:
        - label == 0 → concept present
        - label != 0 → non-concept / random concept samples
    """

    def __init__(self) -> None:
        self.h_pat: torch.Tensor = torch.empty(0)
        self.class_list: List[int] = []

    def train_and_eval(
        self,
        dataloader: DataLoader,
        **kwargs: Any,
    ) -> Union[Dict, None]:
        # Collect activations and labels
        activations, labels = [], []
        for inputs, lbls in dataloader:
            activations.append(inputs)
            labels.append(lbls)


        A = torch.cat(activations).float()  # (N, F)
        t = torch.cat(labels).float()       # (N,)

        t = (t == 0).float()



        # Compute means
        A_mean = A.mean(dim=0, keepdim=True)
        t_mean = t.mean()
        sigma_t2 = ((t - t_mean) ** 2).mean()

        if sigma_t2 == 0:
            raise ValueError("Label variance σ_t^2 is zero; need both concept and non-concept samples.")

        # Compute pattern vector
        self.h_pat = ((A - A_mean).T @ (t - t_mean)) / (A.shape[0] * sigma_t2)

        # Normalize to unit vector
        self.h_pat = self.h_pat / (torch.norm(self.h_pat) + 1e-12)

        # store class ids
        self.class_list = sorted(list(torch.unique(t).cpu().int().numpy()))

        # Optional evaluation metric: correlation between prediction and labels
        preds = (A @ self.h_pat).squeeze()
        corr = torch.corrcoef(torch.stack([preds, t]))[0, 1].item()
        
        self.h_pat = self.h_pat.cpu()
        
        return {"corr": corr}

    def weights(self) -> torch.Tensor:
        if self.h_pat.numel() == 0:
            raise RuntimeError("You must call train_and_eval() before accessing weights().")
        # Return in shape [C, F]; here we define C=2 (concept, non-concept)
        # so Captum can process it consistently.
        return torch.stack([-1*self.h_pat, self.h_pat])

    def classes(self) -> List[int]:
        # Captum expects classes in same order as weight rows.
        # Concept (id=0) first, then non-concept (>0)
        if not self.class_list:
            return [0, 1]
        return self.class_list


if __name__ == "__main__":

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # ---- import your SignalCAVClassifier implementation ----
    # from your_module import SignalCAVClassifier
    # (For quick testing, just paste the SignalCAVClassifier class definition above this block)

    # --------------------------------------------------------
    # 1. Generate synthetic activations
    # --------------------------------------------------------
    torch.manual_seed(42)
    N_concept = 50
    N_random = 50
    F = 10  # number of features

    # Concept-present samples: mean shifted along +e0 and +e1
    A_concept = torch.randn(N_concept, F) + torch.tensor([1.5, 1.0] + [0]*(F-2))

    # Non-concept/random samples: centered at zero
    A_random = torch.randn(N_random, F)

    # Stack everything
    A_all = torch.cat([A_concept, A_random], dim=0)
    t_all = torch.cat([torch.zeros(N_concept), torch.ones(N_random)])  # 0 = concept, 1 = non-concept

    dataset = TensorDataset(A_all, t_all)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --------------------------------------------------------
    # 2. Train the Signal-CAV classifier
    # --------------------------------------------------------
    clf = SignalCav()
    stats = clf.train_and_eval(loader)
    print("Training stats:", stats)

    # --------------------------------------------------------
    # 3. Inspect learned weights
    # --------------------------------------------------------
    weights = clf.weights()
    print("\nWeights shape:", weights.shape)
    print("First 5 values of h_pat:", weights[1, :5])

    # --------------------------------------------------------
    # 4. Test correlation directionality
    # --------------------------------------------------------
    # Projection scores (dot product of activations with h_pat)
    scores = A_all @ weights[1]
    corr = torch.corrcoef(torch.stack([scores, t_all]))[0, 1]
    print(f"\nCorrelation between projection and labels: {corr:.4f}")

    # --------------------------------------------------------
    # 5. Optional sanity check: concept mean vs non-concept mean along h_pat
    # --------------------------------------------------------
    proj_concept = (A_concept @ weights[1]).mean()
    proj_random = (A_random @ weights[1]).mean()
    print(f"\nMean projection (concept):     {proj_concept:.4f}")
    print(f"Mean projection (non-concept): {proj_random:.4f}")
    print(f"Difference (should be positive): {(proj_concept - proj_random):.4f}")
