# replacement of SKSG liner model, using cuda support



import warnings
import torch
import numpy as np
from cuml.svm import LinearSVC
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Union

from captum.concept._utils.classifier import Classifier, DefaultClassifier

class CuMLSVMClassifier(Classifier):
    r"""
    Drop-in CUDA-accelerated replacement for DefaultClassifier.
    Uses cuML's LinearSVC to emulate sklearn's SGDClassifier(loss='hinge').
    """

    def __init__(self) -> None:
        # warnings.warn(
        #     "Using CuMLSVMClassifier (GPU-accelerated LinearSVC). "
        #     "Ensure data fits in GPU memory; this keeps full train/test sets.",
        #     stacklevel=2,
        # )
        self.lm = LinearSVC(
            C=1.0,                # roughly corresponds to inverse of alpha
            max_iter=1000,
            tol=1e-3,
            fit_intercept=True,
            verbose=0,
        )
        self._classes = None
        self._coef_ = None

    def train_and_eval(
        self,
        dataloader: DataLoader,
        test_split_ratio: float = 0.33,
        **kwargs: Any,
    ) -> Union[Dict, None]:
        inputs, labels = [], []

        for x, y in dataloader:
            inputs.append(x)
            labels.append(y)

        X = torch.cat(inputs).detach().cpu().numpy().astype(np.float32)
        y = torch.cat(labels).detach().cpu().numpy().astype(np.int32)

        # manual random split
        n = len(X)
        idx = np.random.permutation(n)
        split = int(n * (1 - test_split_ratio))
        train_idx, test_idx = idx[:split], idx[split:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # fit GPU SVM
        self.lm.fit(X_train, y_train)

        # store info for downstream use
        self._classes = self.lm.classes_.copy()
        self._coef_ = self.lm.coef_.copy()

        y_pred = self.lm.predict(X_test)
        acc = (y_pred == y_test).mean()

        return {"accs": torch.tensor(acc, dtype=torch.float32)}

    def weights(self) -> torch.Tensor:
        r"""
        Returns C x F tensor weights as torch tensor on CPU.
        Matches DefaultClassifier.weight() output.
        """
        if self._coef_ is None:
            raise RuntimeError("Model not trained. Call train_and_eval() first.")

        w = torch.from_numpy(self._coef_).float()
        if w.shape[0] == 1:
            # binary case -> mirror weights for 2-class setup
            return torch.stack([-w[0], w[0]]).cpu()
        return w.cpu()

    def classes(self) -> List[int]:
        if self._classes is None:
            raise RuntimeError("Model not trained yet. Call train_and_eval() first.")
        return self._classes.tolist()


if __name__ == "__main__":

    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.datasets import make_classification

    # make small dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42
    )

    print(f"{y=}")

    # Convert to Torch DataLoader
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=256)

    # CPU baseline (SGDClassifier, hinge loss)
    default_clf = DefaultClassifier()
    cpu_stats = default_clf.train_and_eval(loader)


    # GPU classifier
    gpu_clf = CuMLSVMClassifier()
    gpu_stats = gpu_clf.train_and_eval(loader)

    # Compare results

    print(f"Default accuracy: {cpu_stats['accs'].item():.4f}")

    print(f"GPU (CuML LinearSVC) accuracy: {gpu_stats['accs'].item():.4f}")
    print(f"Weight shape CPU: {default_clf.weights().shape}, GPU: {gpu_clf.weights().shape}")

    # sanity check similarity of weights (directional cosine)
    w_cpu = default_clf.weights().flatten()
    w_gpu = gpu_clf.weights().flatten()

    print(f"{w_cpu.shape=}")
    print(f"{w_gpu.shape=}")
    
    print(f"{w_cpu[5:10]=}")
    print(f"{w_gpu[5:10]=}")



    cos_sim = torch.nn.functional.cosine_similarity(w_cpu, w_gpu, dim=0)
    print(f"Weight cosine similarity (CPU vs GPU): {cos_sim.item():.4f}")