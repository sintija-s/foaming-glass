import numpy as np
from lightning.pytorch import seed_everything

from dataset import preprocess_for_trial_model
from model import calculate_metric, train_test_predict_mlp, train_and_save_final_model

BATCH_SIZE = 10
F_NAME = "data/data.xlsx"


def do_randomseed_trials():
    """Evaluates model performance stability across different random seeds.

    Trains and evaluates a model across multiple trials with different random seeds to assess the impact
    of initialization on the stability of Pearson correlation coefficients. This function iterates over a list
    of seeds, setting each as the global random seed for reproducibility, then processes data, trains the model,
    and computes Pearson correlation coefficients against the test targets.

    Returns:
        tuple: A tuple containing arrays for Pearson coefficients, their mean, and standard deviation across trials.
               - pearson_coef_l: Array of Pearson coefficients for each trial.
               - mean: Mean of Pearson coefficients across trials.
               - stdev: Standard deviation of Pearson coefficients across trials.
    """
    pearson_coef_l = []
    for seed in [12, 32, 42, 99, 103]:
        seed_everything(seed)
        train, val, test = preprocess_for_trial_model(F_NAME, seed, BATCH_SIZE)
        predictions = train_test_predict_mlp(train, val, test, max_epochs=300)
        pearson_corr_coeffs = calculate_metric(predictions, test.dataset.Y)
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

    train_and_save_final_model(
        "mlp_v0", F_NAME, max_epochs=1000, lr=0.001, batch_size=10
    )
