from torch import tensor
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TorchDataset(Dataset):
    """
    A custom dataset for loading and preprocessing data compatible with DataLoader.

    This class is designed to convert NumPy arrays into PyTorch tensors and make them available
    for the DataLoader for efficient batch processing during training.

    Args:
        x (Tensor): The features (numpy.ndarray) converted to PyTorch tensor.
        y (Tensor): The targets (numpy.ndarray) converted to PyTorch tensor.
    """

    def __init__(self, x, y):
        super().__init__()
        self.x = tensor(x).float()
        self.Y = tensor(y).float()

    def __getitem__(self, index):
        return self.x[index], self.Y[index]

    def __len__(self) -> int:
        return len(self.x)


def construct_dataloaders(x, y, batch_size=10, shuffle=False):
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


def encode_clean_categorical_columns(data):
    """
    Encodes categorical columns in the DataFrame to numerical codes and retains only specified columns.

    Args:
        data (pandas.DataFrame): The DataFrame to be processed. It should contain both categorical (as 'object' or 'category')
            and non-categorical data. The function identifies columns of type 'object' or 'category', encodes them with
            numerical codes, and then filters the DataFrame to include only a predefined list of columns.

    Returns:
        pandas.DataFrame: A modified DataFrame with categorical columns encoded as numerical codes and filtered to include only
            a predefined set of columns, making it suitable for subsequent data processing or modeling tasks.
    """

    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    # Encode each categorical column and replace it in the DataFrame
    for col in categorical_cols:
        data[col] = data[col].astype("category").cat.codes

    return data


def preprocess_dataset(data, random_state, batch_size):
    """
    Preprocesses the dataset for machine learning models by encoding categorical columns,
    splitting the data into training, validation, and testing sets, and scaling the data.

    Args:
        data (pandas.DataFrame): The dataset containing both descriptive features and target variables.
        random_state (int): A seed used by the random number generator for reproducibility of the split.
        batch_size (int): The batch size for the dataloader.

    Returns:
        tuple: Contains three DataLoader objects for the training, validation, and testing sets.
               Each DataLoader object encapsulates the respective datasets.
    """
    x = data.iloc[:, :-5]
    x = encode_clean_categorical_columns(x).to_numpy()
    y = data[["apparent_density", "closed_porosity"]].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=random_state
    )

    x_train, x_test, x_val, y_train, y_test, y_val = scale_data(
        x_train, x_test, x_val, y_train, y_test, y_val
    )

    train = construct_dataloaders(x_train, y_train, batch_size=batch_size, shuffle=True)
    val = construct_dataloaders(x_val, y_val, batch_size=batch_size)
    test = construct_dataloaders(x_test, y_test, batch_size=batch_size)

    return train, val, test


def scale_data(x_train, x_test, x_val, y_train, y_test, y_val):
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

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)
    y_val = scaler.transform(y_val)

    return x_train, x_test, x_val, y_train, y_test, y_val
