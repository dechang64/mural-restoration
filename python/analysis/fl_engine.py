"""
Federated Learning engine for multi-institution mural analysis.

Implements FedAvg for collaborative training of mural defect classifiers
and style feature extractors across multiple conservation institutions.

Each institution (client) trains locally on its private mural data,
then shares only model updates (gradients/weights) with the server.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import copy
import time


@dataclass
class FLConfig:
    """Federated learning configuration."""
    num_clients: int = 3
    rounds: int = 10
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    input_dim: int = 768
    num_classes: int = 6  # 6 defect types
    val_split: float = 0.2
    seed: int = 42


@dataclass
class RoundMetrics:
    """Metrics for a single FL round."""
    round_num: int
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    client_metrics: List[Dict[str, float]] = field(default_factory=list)
    elapsed_ms: float = 0.0


class DefectClassifier(nn.Module):
    """Simple MLP classifier for mural defect types.

    Architecture:
        input_dim → 256 → 128 → num_classes (6 defect types)
    """

    def __init__(self, input_dim: int = 768, num_classes: int = 6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MuralFLEngine:
    """Federated Learning engine for mural defect classification.

    Implements FedAvg aggregation across multiple conservation institutions.
    Each client trains locally on its private mural feature data.
    """

    def __init__(self, config: Optional[FLConfig] = None):
        self.config = config or FLConfig()
        self.global_model = DefectClassifier(
            input_dim=self.config.input_dim,
            num_classes=self.config.num_classes,
        )
        self.history: List[RoundMetrics] = []
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.global_model.to(self._device)

    def _split_data(self, features: np.ndarray, labels: np.ndarray,
                    num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data across clients (non-IID simulation)."""
        rng = np.random.RandomState(self.config.seed)
        indices = rng.permutation(len(features))
        client_size = len(features) // num_clients
        splits = []
        for i in range(num_clients):
            start = i * client_size
            end = start + client_size if i < num_clients - 1 else len(features)
            client_idx = indices[start:end]
            splits.append((features[client_idx], labels[client_idx]))
        return splits

    def _train_client(self, model: nn.Module,
                      features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Train a single client for local_epochs."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Split into train/val
        n_val = int(len(features) * self.config.val_split)
        if n_val == 0:
            n_val = max(1, len(features) // 10)
        val_idx = np.random.choice(len(features), n_val, replace=False)
        train_idx = np.array([i for i in range(len(features)) if i not in val_idx])

        if len(train_idx) == 0:
            train_idx = val_idx

        train_dataset = TensorDataset(
            torch.FloatTensor(features[train_idx]),
            torch.LongTensor(labels[train_idx]),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        total_loss = 0.0
        n_batches = 0
        for _ in range(self.config.local_epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self._device), batch_y.to(self._device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_x = torch.FloatTensor(features[val_idx]).to(self._device)
            val_y = torch.LongTensor(labels[val_idx]).to(self._device)
            val_output = model(val_x)
            val_loss = criterion(val_output, val_y).item()
            val_acc = (val_output.argmax(dim=1) == val_y).float().mean().item()

        return {
            "train_loss": total_loss / max(n_batches, 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

    def _fedavg_aggregate(self, client_models: List[nn.Module]) -> None:
        """FedAvg: average client model parameters."""
        global_state = self.global_model.state_dict()
        aggregated = {}

        for key in global_state.keys():
            stacked = torch.stack([
                client.state_dict()[key].float()
                for client in client_models
            ], dim=0)
            aggregated[key] = stacked.mean(dim=0).to(global_state[key].dtype)

        self.global_model.load_state_dict(aggregated)

    def run(self, features: np.ndarray, labels: np.ndarray,
            num_clients: Optional[int] = None,
            rounds: Optional[int] = None) -> List[RoundMetrics]:
        """Run federated training.

        Args:
            features: Feature vectors (N, input_dim)
            labels: Defect type labels (N,), values 0-5
            num_clients: Override number of clients
            rounds: Override number of FL rounds

        Returns:
            List of RoundMetrics for each round
        """
        num_clients = num_clients or self.config.num_clients
        rounds = rounds or self.config.rounds

        client_data = self._split_data(features, labels, num_clients)
        self.history = []

        for round_num in range(1, rounds + 1):
            start = time.time()
            client_models = []
            client_metrics = []

            for client_features, client_labels in client_data:
                client_model = copy.deepcopy(self.global_model)
                metrics = self._train_client(client_model, client_features, client_labels)
                client_models.append(client_model)
                client_metrics.append(metrics)

            self._fedavg_aggregate(client_models)

            # Global validation
            self.global_model.eval()
            with torch.no_grad():
                all_x = torch.FloatTensor(features).to(self._device)
                all_y = torch.LongTensor(labels).to(self._device)
                criterion = nn.CrossEntropyLoss()
                output = self.global_model(all_x)
                val_loss = criterion(output, all_y).item()
                val_acc = (output.argmax(dim=1) == all_y).float().mean().item()

            avg_train_loss = np.mean([m["train_loss"] for m in client_metrics])
            elapsed = (time.time() - start) * 1000

            round_metrics = RoundMetrics(
                round_num=round_num,
                train_loss=round(avg_train_loss, 4),
                val_loss=round(val_loss, 4),
                val_acc=round(val_acc, 4),
                client_metrics=client_metrics,
                elapsed_ms=round(elapsed, 1),
            )
            self.history.append(round_metrics)

        return self.history

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict defect types for feature vectors.

        Args:
            features: Feature vectors (N, input_dim)

        Returns:
            (predictions, probabilities) - predictions (N,), probabilities (N, num_classes)
        """
        self.global_model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self._device)
            logits = self.global_model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
        return preds, probs
