from torch import optim, nn, cat, no_grad
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import warnings
from joblib import load

import dataset

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class MLP(LightningModule):
    """Multi-Layer Perceptron for multi-target regression using PyTorch Lightning.

    This MLP model is designed with two hidden layers and an output layer for multi-target regression tasks.
    It incorporates a Smooth L1 Loss for training and tracks the Mean Squared Error (MSE) as a performance metric.

    Args:
        lr (float): Learning rate for the optimizer. Defaults to 0.01.
    """

    def __init__(self, lr=0.001):
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
        """Performs a forward pass through the network."""
        return self.NN(x)

    def configure_optimizers(self):
        """Configures the model's optimizer."""
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """Executes the training step."""
        loss, mse, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {"train_loss": loss, "train_mse": mse.mean()}, prog_bar=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Executes the validation step."""
        loss, mse, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        """Executes the test step."""
        loss, mse, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True, on_epoch=False, on_step=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Executes the prediction step (accessible by using the trainer)."""
        x, y = batch
        return self(x)

    def _common_step(self, batch, batch_idx):
        """A common step for training, validation, and test steps."""
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        mse = self.mse(y_pred, y)
        return loss, mse, y


def initialize_callbacks():
    """Initializes training callbacks.

    Prepares and returns a list of callbacks to be used during the training process.
    These callbacks include early stopping to prevent overfitting and a rich progress
    bar for enhanced training progress visualization.

    Returns:
        list: A list containing an EarlyStopping callback, which monitors validation loss
        for early stopping, and a RichProgressBar callback for improved training UI.
    """
    return [EarlyStopping(monitor="val_loss"), RichProgressBar()]


def initialize_trainer_and_train(trainds, valds, max_epochs, lr=0.001):
    """Trains an MLP model with specified callbacks and configurations.

    Initializes an MLP model with a given learning rate, sets up a TensorBoard logger for monitoring,
    and trains the model using the PyTorch Lightning Trainer with early stopping and a rich progress bar.
    The model is trained on the provided training and validation DataLoaders for a specified number of epochs.

    Args:
        trainds (DataLoader): The DataLoader containing the training dataset.
        valds (DataLoader): The DataLoader containing the validation dataset.
        max_epochs (int): The maximum number of epochs to train the model.
        lr (float, optional): The learning rate for the MLP model. Defaults to 0.001.

    Returns:
        tuple: A tuple containing the Trainer instance and the trained MLP model.
    """
    # Initialize the model
    model = MLP(lr)
    # Setup logging with TensorBoard
    logger = TensorBoardLogger("lightning_logs", name="mlp_model")
    # Initialize the trainer with specified configurations and callbacks
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        deterministic=True,
        callbacks=initialize_callbacks(),  # Get callbacks for training
        log_every_n_steps=5,
    )

    # Train the model
    trainer.fit(model, trainds, valds)

    return trainer, model


def train_test_predict_mlp(trainds, valds, testds, max_epochs, lr=0.001):
    """Trains, tests, and predicts using an MLP model with specified parameters.

    This function orchestrates the training, testing, and prediction phases of an MLP model workflow,
    utilizing the train_mlp_with_callbacks function for training. It then tests the model on a provided
    test dataset and finally generates predictions for that test dataset.

    Args:
        trainds (DataLoader): DataLoader for the training data.
        valds (DataLoader): DataLoader for the validation data.
        testds (DataLoader): DataLoader for the test data.
        max_epochs (int): Maximum number of epochs to train the model.
        lr (float, optional): Learning rate for the MLP model. Defaults to 0.001.

    Returns:
        Tensor: Predictions made by the model on the test dataset.
    """
    # Train the model using the provided datasets and parameters
    trainer, model = initialize_trainer_and_train(trainds, valds, max_epochs, lr)
    # Evaluate the model on the test dataset
    trainer.test(model, testds)
    # Generate and concatenate predictions for the test dataset
    predictions = cat(trainer.predict(model, testds))

    return predictions


def calculate_metric(pred, true):
    """Computes the Pearson correlation coefficient between predictions and true values.

    The Pearson correlation coefficient is calculated for model predictions and true values
    to assess the linear relationship between them. It ranges from -1 (total negative linear correlation),
    through 0 (no linear correlation), to 1 (total positive linear correlation).

    Args:
        pred (Tensor): Predicted values by the model.
        true (Tensor): Actual true values.

    Returns:
        Tensor: Pearson correlation coefficient for each output.
    """
    # Validate input shapes
    if not (pred.shape == true.shape):
        raise ValueError("Predictions and true values must have the same shape.")

    # Initialize Pearson correlation coefficient calculation
    pearson = torchmetrics.PearsonCorrCoef(num_outputs=2)
    return pearson(pred, true)


def predict_with_model(x, mlp):
    """Generates predictions using an MLP model and scales back to original values.

    Preprocesses the input data, performs prediction with the MLP model in evaluation mode, and
    applies inverse transformation to the predictions to scale them back to their original value range.

    Args:
        x (numpy.ndarray): The input data for making predictions.
        mlp (MLP): The trained MLP model.

    Returns:
        numpy.ndarray: The predicted values, scaled back to the original value range.
    """
    # Preprocess input data
    tensor_input = dataset.preprocess_for_prediction(x)

    # Set model to evaluation mode and predict without tracking gradients
    mlp.eval()
    with no_grad():
        y_hat = mlp(tensor_input)

    # Load the scaler used for the target variables and apply inverse transformatio
    scaler = load("model\\target_scaler.save")
    y_hat = scaler.inverse_transform(y_hat)

    return y_hat
