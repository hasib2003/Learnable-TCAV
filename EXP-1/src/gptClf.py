import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, Dict, Optional

# Minimal Classifier that follows the expected interface:
class TorchClassifier:
    """
    Train a simple linear classifier in PyTorch and expose:
      - train_and_eval(dataloader, **kwargs) -> stats dict (or None)
      - weights() -> numpy array shape (n_classes, n_features)
      - classes() -> numpy array or list of class labels
    """

    def __init__(
        self,
        device: Optional[str] = None,
        lr: float = 1e-2,
        epochs: int = 200,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lr = lr
        self.epochs = int(epochs)
        self.batch_size = batch_size
        self.verbose = verbose

        # internal placeholders (populated after training)
        self._model: Optional[nn.Module] = None
        self._classes: Optional[np.ndarray] = None
        self._trained = False

    def _build_model(self, in_features: int, n_classes: int) -> nn.Module:
        # single linear layer (no activation); CrossEntropyLoss expects raw logits
        m = nn.Linear(in_features, n_classes, bias=True)
        return m.to(self.device)

    def train_and_eval(self, dataloader, **kwargs) -> Optional[Dict[str, Any]]:
        """
        dataloader yields (inputs, labels), where inputs shape = (batch, feat)
        and labels are integers.
        """
        # gather some basic shape info from one batch (dataloader may be generator)
        it = iter(dataloader)
        try:
            sample_x, sample_y = next(it)
        except StopIteration:
            raise RuntimeError("Empty dataloader passed to TorchClassifier")

        # ensure shapes and device
        sample_x = sample_x.to(self.device)
        n_features = sample_x.shape[1]
        # collect all classes from dataloader or from labels in batch
        # NOTE: dataloader will be iterated twice below; better if dataloader is finite
        labels_list = [sample_y]
        for bx, by in it:
            labels_list.append(by)
        all_labels = torch.cat(labels_list, dim=0)
        unique_classes = torch.unique(all_labels).cpu().numpy()
        n_classes = int(unique_classes.size)

        # Build model
        self._model = self._build_model(n_features, n_classes)
        optimizer = optim.SGD(self._model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        # Optionally pack the batches (we already exhausted one iterator above),
        # so construct a fresh dataloader iterator for training.
        # NOTE: some DataLoader objects are generators that can only be iterated once.
        # So to be safe, we'll re-create an iterator from the original dataloader object.
        def data_iter():
            for x, y in dataloader:
                yield x.to(self.device), y.to(self.device).long()

        # Simple training loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_seen = 0
            for x_batch, y_batch in data_iter():
                optimizer.zero_grad()
                logits = self._model(x_batch)  # shape (B, n_classes)
                # map labels into contiguous indices 0..n_classes-1 if labels are arbitrary
                # Build mapping from class value -> index
                # We compute mapping once per training (cheap).
                if epoch == 0 and n_seen == 0:
                    # compute mapping
                    unique_vals = torch.unique(y_batch).cpu().numpy()
                    # Build mapping dict from observed labels (but better to use unique_classes)
                    # We'll use unique_classes computed earlier:
                    class_to_idx = {int(v): i for i, v in enumerate(unique_classes)}
                    self._class_to_idx = class_to_idx  # store for later

                # convert y_batch to idx space
                y_idx = torch.tensor([self._class_to_idx[int(v)] for v in y_batch.cpu().numpy()], device=self.device, dtype=torch.long)

                loss = loss_fn(logits, y_idx)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item()) * x_batch.size(0)
                n_seen += x_batch.size(0)

            if self.verbose and (epoch % max(1, self.epochs // 5) == 0):
                print(f"[TorchClassifier] Epoch {epoch+1}/{self.epochs} loss={epoch_loss / max(1,n_seen):.4f}")

        # mark trained
        self._trained = True
        # Save classes in the same order as model outputs
        self._classes = unique_classes.astype(int)

        # Optionally compute simple metrics (accuracy on training set)
        correct = 0
        total = 0
        for x_batch, y_batch in data_iter():
            logits = self._model(x_batch)
            preds = logits.argmax(dim=1)
            # map true labels to idx
            true_idx = torch.tensor([self._class_to_idx[int(v)] for v in y_batch.cpu().numpy()], device=self.device)
            correct += (preds == true_idx).sum().item()
            total += preds.size(0)
        acc = (correct / total) if total > 0 else None

        stats = {"train_accuracy": acc} if acc is not None else {}
        return stats

    def weights(self) -> Optional[np.ndarray]:
        if not self._trained or self._model is None:
            return None
        # linear weight shape: (n_classes, n_features)
        w = self._model.weight.detach().cpu().numpy().copy()
        return w

    def classes(self) -> Optional[np.ndarray]:
        return self._classes.copy() if self._classes is not None else None
