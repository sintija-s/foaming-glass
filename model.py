from torch import optim, nn, cat
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class MLP(LightningModule):
    """
    A Multi-Layer Perceptron (MLP) class for multi-target regression, implemented using PyTorch Lightning.

    Args:
        NN (nn.Sequential): Defines the neural network structure with two hidden layers followed by an output layer.
        lr (float): Learning rate for the optimizer.
        loss (nn.SmoothL1Loss): Loss function used for training the model.
        mse (torchmetrics.MeanSquaredError): Metric for calculating the Mean Squared Error (MSE) during training.
    """

    def __init__(self, lr=0.01):
        super(MLP, self).__init__()
        self.NN = nn.Sequential(
            # 1st hidden layer
            nn.Linear(9, 6),
            nn.ReLU(),
            # 2nd hidden layer
            nn.Linear(6, 4),
            nn.ReLU(),
            # output layer
            nn.Linear(4, 2),
        )
        self.lr = lr
        self.loss = nn.SmoothL1Loss(reduction="mean")
        self.mse = torchmetrics.MeanSquaredError(num_outputs=2)

    def forward(self, x):
        return self.NN(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        loss, mse, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {"train_loss": loss, "train_mse": mse.mean()}, prog_bar=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mse, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        loss, mse, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True, on_epoch=False, on_step=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def _common_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        mse = self.mse(y_pred, y)
        return loss, mse, y


def initialize_callbacks():
    """
    Initializes and returns a list of desired callbacks for the training process.
    """

    return [EarlyStopping(monitor="val_loss"), RichProgressBar()]


def train_model(trainds, valds, testds, max_epochs, lr=0.001):
    """
    Trains the MLP model with given datasets and parameters, logs the training process,
    and evaluates the model on the test dataset.

    Args:
        trainds: DataLoader for the training dataset.
        valds: DataLoader for the validation dataset.
        testds: DataLoader for the test dataset.
        max_epochs (int): Maximum number of epochs for training.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
        Tensor: Concatenated predictions from the test dataset.
    """

    model = MLP(lr)
    logger = TensorBoardLogger("lightning_logs", name="mlp_model")
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        deterministic=True,
        callbacks=initialize_callbacks(),
        log_every_n_steps=5,
    )

    trainer.fit(model, trainds, valds)
    trainer.test(model, testds)
    predictions = cat(trainer.predict(model, testds))

    return predictions


def calculate_metric(pred, true):
    """
    Calculates the Pearson correlation coefficient between model predictions and true values.

    This function initializes a PearsonCorrCoef object configured for multiple outputs and computes the
    Pearson correlation coefficient for the given predictions and true values.
    The Pearson correlation coefficient is a measure of the linear correlation between two variables,
    ranging from -1 to 1, where 1 means total positive linear correlation, 0 means no linear correlation,
    and -1 means total negative linear correlation.

    Args:
        pred (Tensor): The model's predictions, as a tensor.
        true (Tensor): The true values, as a tensor.

    Returns:
        Tensor: A tensor containing the Pearson correlation coefficient for each output.
    """

    # Ensure pred and true are tensors with the same shape
    if not (pred.shape == true.shape):
        raise ValueError("Predictions and true values must have the same shape.")

    pearson = torchmetrics.PearsonCorrCoef(num_outputs=2)
    return pearson(pred, true)
