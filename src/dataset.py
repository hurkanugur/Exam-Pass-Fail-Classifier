# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import config

# -------------------------
# Private Dataset Class
# -------------------------
class _ExamDataset(Dataset):
    """
    Custom Dataset for exam pass/fail classification.
    Loads features 'Study Hours' and 'Previous Exam Score' and the target 'Pass/Fail'.
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(
            df[["Study Hours", "Previous Exam Score"]].values,
            dtype=torch.float32
        )
        self.y = torch.tensor(
            df["Pass/Fail"].values,
            dtype=torch.float32
        ).reshape((-1, 1))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# Private Dataset Splitting
# -------------------------
def _split_dataset(dataset):
    """
    Split the dataset into train, validation, and test subsets based on config ratios.
    """
    n_total = len(dataset)
    n_train = int(config.TRAIN_SPLIT * n_total)
    n_val = int(config.VAL_SPLIT * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(config.RANDOM_SEED) if config.RANDOM_SEED is not None else None
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    return train_ds, val_ds, test_ds

# -------------------------
# Public DataLoader Creation
# -------------------------
def create_dataloaders():
    """
    Create DataLoaders for training, validation, and testing.
    If config.SPLIT_DATASET is False, the same dataset is used for all loaders.
    """
    dataset = _ExamDataset(config.DATA_PATH)

    if config.SPLIT_DATASET:
        train_ds, val_ds, test_ds = _split_dataset(dataset)
    else:
         # Use the same dataset for train, val, test
        train_ds = val_ds = test_ds = dataset

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader