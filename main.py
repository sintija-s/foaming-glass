from dataset import preprocess_dataset
import model
import pandas as pd
from lightning.pytorch import seed_everything
import numpy as np

BATCH_SIZE = 10

def do_randomseed_trials():
    """
    Executes model training and evaluation across different random seeds to assess the stability of
    Pearson correlation coefficients for model predictions.

    The function iterates through a list of predetermined random seeds, setting the global random seed to ensure reproducibility.
    For each seed, it loads a dataset, preprocesses it, trains a model, and calculates the
    Pearson correlation coefficients for the model's predictions against the test set's targets.
    It aims to evaluate the impact of random initialization on the model's performance, particularly looking at the
    stability of Pearson correlation coefficients across trials.

    Args:
        None

    Returns:
        tuple: Contains the following numpy arrays:
            - pearson_coef_l: All Pearson correlation coefficients for each trial.
            - mean: The mean of Pearson correlation coefficients across trials.
            - stdev: The standard deviation of Pearson correlation coefficients across trials.

    Note:
        The dataset is expected to be in an Excel file named 'data.xlsx' in the "data" folder.
    """
    pearson_coef_l = []
    for seed in [12, 32, 42, 99, 103]:
        seed_everything(seed)
        data = pd.read_excel("data/data.xlsx")
        train, val, test = preprocess_dataset(data, seed, BATCH_SIZE)
        predictions = model.train_model(train, val, test, max_epochs=300)
        pearson_corr_coeffs = model.calculate_metric(predictions, test.dataset.Y)
        pearson_coef_l.append(pearson_corr_coeffs.numpy())
        print(
            f"Pearson Correlation Coefficients for seed {seed}:\nApparent Density: {pearson_corr_coeffs[0]} \nClosed Porosity: {pearson_corr_coeffs[1]}"
        )
    pearson_coef_l = np.vstack(pearson_coef_l)
    mean = np.mean(pearson_coef_l, axis=0)
    stdev = np.std(pearson_coef_l, axis=0)

    return pearson_coef_l, mean, stdev


if __name__ == "__main__":
    pearson_coef_l, mean, stdev = do_randomseed_trials()
    print(
        f"---------------------------\n"
        f"Mean Correlation Coefficients:\n"
        f"Apparent Density: {mean[0]} +- {stdev[0]}\n"
        f"Closed Porosity: {mean[1]} +- {stdev[1]}"
    )
