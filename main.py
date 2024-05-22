import numpy as np
from lightning.pytorch import seed_everything
import os.path as path

from dataset import preprocess_for_trial_model, preprocess_for_final_model
from model import (
    calculate_metric,
    train_test_predict_mlp,
    initialize_trainer_and_train,
    POSSIBLE_SEEDS
)

BATCH_SIZE = 10
DATASET_PATH = "data/data.xlsx"
LEARNING_RATE = 0.001
MAX_EPOCHS = 300
MODEL_NAME = "mlp_v1"


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
        train, val, test = preprocess_for_trial_model(DATASET_PATH, seed, BATCH_SIZE)
        predictions = train_test_predict_mlp(train, val, test, max_epochs=MAX_EPOCHS)
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

    # Train 10 models with different random seeds:
    for seed in POSSIBLE_SEEDS:
        seed_everything(seed)
        # Preprocess data to get training and validation DataLoaders
        trainds, valds = preprocess_for_final_model(DATASET_PATH, BATCH_SIZE, seed)
        # Train the model with callbacks
        trainer, model = initialize_trainer_and_train(
            trainds, valds, max_epochs=MAX_EPOCHS, lr=LEARNING_RATE
        )
        # Save the final model checkpoint
        trainer.save_checkpoint(path.join("model", f"{MODEL_NAME}_{seed}.ckpt"))
