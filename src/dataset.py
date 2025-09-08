import os
import pickle
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config


class ExamDataset(Dataset):
    """
    Dataset class for exam pass/fail classification.
    Provides training data (split + normalize) and inference data (normalize only).
    """

    def __init__(self):
        self.x_scaler = StandardScaler()

    # ----------------- PUBLIC METHODS -----------------
    def prepare_data_for_training(self):
        """Load data, split train/val/test, fit scaler, normalize features, and return DataLoaders."""
        df = self._load_csv()
        X = torch.tensor(df[["Study Hours", "Previous Exam Score"]].values, dtype=torch.float32)
        y = torch.tensor(df["Pass/Fail"].values, dtype=torch.float32).reshape((-1, 1))

        X_train, X_val, X_test, y_train, y_val, y_test = self._split_dataset(X, y)
        X_train_norm, X_val_norm, X_test_norm = self._normalize_features(X_train, X_val, X_test)
        train_loader, val_loader, test_loader = self._create_data_loaders(X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test)

        return train_loader, val_loader, test_loader

    def prepare_data_for_inference(self, df: pd.DataFrame):
        """Normalize new input features for inference."""
        X = torch.tensor(df[["Study Hours", "Previous Exam Score"]].values, dtype=torch.float32)
        X_norm = torch.tensor(self.x_scaler.transform(X), dtype=torch.float32)
        return X_norm

    def get_input_dim(self, data_loader):
        """Return number of input features dynamically."""
        sample_X, _ = next(iter(data_loader))
        return sample_X.shape[1]

    def save_normalization_params(self):
        """Save fitted scaler for inference."""
        os.makedirs(os.path.dirname(config.NORM_PARAMS_PATH), exist_ok=True)
        with open(config.NORM_PARAMS_PATH, "wb") as f:
            pickle.dump(self.x_scaler, f)
        print(f"• Normalization parameters saved: {config.NORM_PARAMS_PATH}")

    def load_normalization_params(self):
        """Load fitted scaler from file (must be called before inference)."""
        with open(config.NORM_PARAMS_PATH, "rb") as f:
            self.x_scaler = pickle.load(f)
        print(f"• Normalization parameters loaded: {config.NORM_PARAMS_PATH}")

    # ----------------- PRIVATE METHODS -----------------
    
    def _load_csv(self):
        """Load CSV into a pandas DataFrame."""
        df = pd.read_csv(config.DATASET_CSV_PATH)
        print(f"• Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def _split_dataset(self, X, y):
        """Split dataset into training, validation, and test subsets."""

        if not config.SPLIT_DATASET:
            print("• Dataset splitting disabled → using same data for train/val/test.")
            return X, X, X, y, y, y
    
        dataset = TensorDataset(X, y)
        n_total = len(dataset)
        n_train = int(config.TRAIN_SPLIT_RATIO * n_total)
        n_val = int(config.VAL_SPLIT_RATIO * n_total)
        n_test = n_total - n_train - n_val

        generator = (
            torch.Generator().manual_seed(config.SPLIT_RANDOMIZATION_SEED)
            if config.SPLIT_RANDOMIZATION_SEED is not None else None
        )

        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test], 
            generator=generator
        )

        X_train, y_train = train_ds[:][0], train_ds[:][1]
        X_val, y_val = val_ds[:][0], val_ds[:][1]
        X_test, y_test = test_ds[:][0], test_ds[:][1]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _normalize_features(self, X_train, X_val, X_test):
        """Fit scaler on training set, transform train/val/test features."""
        X_train_norm = torch.tensor(self.x_scaler.fit_transform(X_train), dtype=torch.float32)
        X_val_norm = torch.tensor(self.x_scaler.transform(X_val), dtype=torch.float32)
        X_test_norm = torch.tensor(self.x_scaler.transform(X_test), dtype=torch.float32)
        return X_train_norm, X_val_norm, X_test_norm

    def _create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Return train, val, test DataLoaders."""
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)
        return train_loader, val_loader, test_loader
