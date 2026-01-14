from typing import Annotated

import pandas as pd
from zenml import ArtifactConfig, step
from zenml.enums import ArtifactType

# Try to import PyTorch components, but make them optional
try:
    import platform
    from pathlib import Path

    if platform.system() == "Windows":
        import ctypes
        from importlib.util import find_spec

        try:
            if (
                (spec := find_spec("torch"))
                and spec.origin
                and Path(
                    dll_path := Path(Path(spec.origin).parent, "lib", "c10.dll")
                ).exists
            ):
                ctypes.CDLL(str(dll_path))
        except ImportError:
            HAS_PYTORCH = False

    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
    import torchmetrics
    from lightning import LightningModule, Trainer
    from torch.utils.data import DataLoader, TensorDataset

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False


class DNNModel(LightningModule if HAS_PYTORCH else object):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.save_hyperparameters()

        self.layer1 = tnn.Linear(input_dim, hidden_dim)
        self.layer2 = tnn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = tnn.Linear(hidden_dim // 2, num_classes)
        self.dropout = tnn.Dropout(0.2)

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        return self.layer3(x)

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.accuracy.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self):
        acc = self.accuracy.compute()
        self.log("val_acc", acc, on_epoch=True)
        self.accuracy.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class PyTorchDNNModel:
    def __init__(self):
        if not HAS_PYTORCH:
            msg = (
                "PyTorch, PyTorch Lightning, and torchmetrics are required for this model. "
                "Please install them with: pip install torch pytorch-lightning torchmetrics"
            )
            raise ImportError(msg)
        self.model = None
        self.trainer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.LongTensor(y.values)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize model
        input_dim = X.shape[1]
        num_classes = y.nunique()
        self.model = DNNModel(input_dim, num_classes)

        # Initialize trainer
        self.trainer = Trainer(
            max_epochs=10,
            accelerator="auto",
            devices=1,
            enable_progress_bar=True,
            logger=False,  # Disable logging for simplicity
        )

        # Train the model
        self.trainer.fit(self.model, dataloader, dataloader)

    def predict(self, X: pd.DataFrame):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            predictions = self.model(X_tensor)
            return torch.argmax(predictions, dim=1).numpy()


@step
def dnn_trainer(
    dataset_trn: pd.DataFrame,
    target: str | None = "target",
) -> Annotated[
    PyTorchDNNModel,
    ArtifactConfig(name="dnn_model", artifact_type=ArtifactType.MODEL),
]:
    """Configure and train a DNN on the training dataset using PyTorch Lightning.

    This is an example of a model training step that takes in a dataset artifact
    previously loaded and pre-processed by other steps in your pipeline, then
    configures and trains a DNN on it. The model is then returned as a step
    output artifact.

    Args:
        dataset_trn: The preprocessed train dataset.
        target: The name of the target column in the dataset.

    Returns:
        The trained DNN artifact.

    Raises:
        ImportError: If PyTorch, PyTorch Lightning, or torchmetrics are not installed.
    """
    if not HAS_PYTORCH:
        msg = (
            "PyTorch, PyTorch Lightning, and torchmetrics are required for this model. "
            "Please install them with: pip install torch pytorch-lightning torchmetrics"
        )
        raise ImportError(msg)

    # Initialize the model with the hyperparameters indicated in the step
    # parameters and train it on the training set.
    dnn_model = PyTorchDNNModel()
    dnn_model.fit(dataset_trn.drop(columns=[target]), dataset_trn[target])
    return dnn_model
