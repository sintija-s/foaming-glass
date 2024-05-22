from torch import tensor, from_numpy
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from joblib import dump, load
import os


class TorchDataset(Dataset):
    """Custom dataset for converting NumPy arrays to PyTorch tensors for DataLoader.

    Args:
        x (numpy.ndarray): Input features to be converted to PyTorch tensors.
        y (numpy.ndarray): Target labels to be converted to PyTorch tensors.
    """

    def __init__(self, x, y):
        """Initializes the dataset with features and targets."""
        super().__init__()
        self.x = tensor(x).float()
        self.Y = tensor(y).float()

    def __getitem__(self, index):
        """Retrieves the feature-target pair at the specified index.

        Args:
            index (int): The index of the item.

        Returns:
            Tuple[Tensor, Tensor]: The feature-target pair as PyTorch tensors.
        """

        return self.x[index], self.Y[index]

    def __len__(self):
        """Returns the total number of items in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.x)


def create_dataloader(x, y, batch_size=10, shuffle=False):
    """
    Creates a PyTorch DataLoader for a given dataset.

    Args:
        x (numpy.ndarray): Descriptive data.
        y (numpy.ndarray): Target data.
        batch_size (int, optional): The batch size for the DataLoader. Default is 10.
        shuffle (bool, optional): Whether to shuffle the data before each epoch. Default is False.
            Shuffling is recommended during training to improve generalization.

    Returns:
        DataLoader: A PyTorch DataLoader object for the specified dataset.
    """

    dataset = TorchDataset(x, y)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    return dataloader


def encode_and_clean_categorical(data):
    """Encodes and cleans categorical columns in a DataFrame.

    This function encodes categorical columns (types 'object' or 'category') in the DataFrame to numerical codes.
    It updates the DataFrame in-place, replacing categorical columns with their encoded counterparts.

    Args:
        data (pandas.DataFrame): The DataFrame containing categorical and non-categorical data.

    Returns:
        pandas.DataFrame: The DataFrame with categorical columns encoded as numerical codes.
    """
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        data[col] = data[col].astype("category").cat.codes

    return data


def load_and_split_data(data):
    """
    Loads data and splits it into feature and target spaces.

    Args:
        data (pandas.DataFrame): The dataset containing both descriptive features and target variables.

    Returns:
        tuple of numpy.ndarrays: The feature and target spaces as numpy arrays.
    """
    # Process features, excluding the last 5 columns
    x = data.iloc[:, :-5]
    # Encode categorical columns and convert to numpy array
    x = encode_and_clean_categorical(x).to_numpy()
    # Extract targets and convert to numpy array
    y = data[["apparent_density", "closed_porosity"]].to_numpy()

    return x, y


def preprocess_for_trial_model(fname, random_state, batch_size):
    """Prepares data for trial model runs by encoding, splitting, and scaling.

    This function reads a dataset from an Excel file, preprocesses it by encoding categorical columns,
    splits it into training, validation, and testing sets, and applies standard scaling. It then
    wraps the split datasets into DataLoader objects for use in machine learning models.

    Args:
        fname (str): Filename of the Excel file containing the dataset.
        random_state (int): Seed for random operations, ensuring reproducibility.
        batch_size (int): Number of samples per batch in the DataLoader.

    Returns:
        tuple: A tuple containing DataLoader objects for training, validation, and testing datasets.
    """
    # Load dataset from Excel file
    data = pd.read_excel(fname)
    # Split data into features and targets, encode categorical columns
    x, y = load_and_split_data(data)

    # Split data into training, testing, and validation sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=random_state
    )

    # Apply standard scaling to the data
    x_train, x_test, x_val, y_train, y_test, y_val = standard_scale_data(
        x_train, x_test, x_val, y_train, y_test, y_val
    )

    # Create DataLoader instances for each dataset
    train = create_dataloader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val = create_dataloader(x_val, y_val, batch_size=batch_size)
    test = create_dataloader(x_test, y_test, batch_size=batch_size)

    return train, val, test


def preprocess_for_final_model(fname, batch_size, seed):
    """Preprocesses data for the saved model (the one trained on the entire dataset) by encoding, splitting, and scaling.

    Reads the dataset from an Excel file, preprocesses it by encoding categorical columns, splits
    into training and validation sets, applies standard scaling to features and targets separately,
    and prepares DataLoader objects for both sets. It also saves the scaler used for targets for
    inverse transformations during prediction.

    Args:
        fname (str): Filename of the Excel file containing the dataset.
        batch_size (int): Number of samples per batch in the DataLoader.
        seed (int): Random seed for the train/test split.

    Returns:
        tuple: Contains two DataLoader objects for the training and validation datasets.
    """
    # Load dataset from Excel file
    data = pd.read_excel(fname)
    # Split data into features and targets, encode categorical columns
    x, y = load_and_split_data(data)

    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.1, random_state=seed
    )

    # Scale features and save the scaler for features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    dump(scaler, os.path.join("model", f"feature_scaler_{seed}.save"))
    x_val = scaler.transform(x_val)

    # Scale targets separately and save the scaler for targets
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    dump(y_scaler, os.path.join("model", f"target_scaler_{seed}.save"))
    y_val = y_scaler.fit_transform(y_val)

    # Create DataLoader instances for training and validation sets
    trainds = create_dataloader(x_train, y_train, batch_size=batch_size, shuffle=True)
    valds = create_dataloader(x_val, y_val, batch_size=batch_size)

    return trainds, valds


def preprocess_for_prediction(data, seed):
    """Prepares input data for prediction by scaling and converting to PyTorch tensor.

    This function scales the input data using StandardScaler and converts the scaled data into a PyTorch tensor.
    It is designed for preprocessing data just before making predictions with a trained model.

    Args:
        data (numpy.ndarray): Input data to be preprocessed.

    Returns:
        Tensor: A PyTorch tensor of the preprocessed input data.
    """
    # Load the saved feature scaler to the data
    scaler = load(os.path.join("model", f"feature_scaler_{seed}.save"))
    data = scaler.transform(data)

    # Convert the scaled data to a PyTorch tensor
    tensor_input = from_numpy(data).float()

    return tensor_input


def standard_scale_data(x_train, x_test, x_val, y_train, y_test, y_val):
    """
    Scales the feature and target datasets using the StandardScaler, fitting the scaler on the training data
    and then transforming the training, testing, and validation datasets.

    Args:
        x_train (numpy.ndarray): Feature dataset for training.
        x_test (numpy.ndarray): Feature dataset for testing.
        x_val (numpy.ndarray): Feature dataset for validation.
        y_train (numpy.ndarray): Target dataset for training.
        y_test (numpy.ndarray): Target dataset for testing.
        y_val (numpy.ndarray): Target dataset for validation.

    Returns:
        tuple of numpy.ndarrays: The function returns a tuple containing the scaled versions of
        x_train, x_test, x_val, y_train, y_test, and y_val, respectively.
    """
    # Initialize scaler for features
    scaler = StandardScaler()
    # Scale feature datasets
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)

    # Initialize scaler for targets
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)
    y_val = scaler.transform(y_val)

    return x_train, x_test, x_val, y_train, y_test, y_val
