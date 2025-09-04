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
    Handles loading, splitting, normalization, and provides DataLoaders.
    """

    # ----------------- CLASS ATTRIBUTES -----------------
    FEATURE_COLUMNS = ["Study Hours", "Previous Exam Score"]
    TARGET_COLUMN = "Pass/Fail"

    def __init__(self):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.norm_params = None

    # ----------------- PUBLIC METHODS -----------------
    def prepare_data(self):
        """
        Perform all preprocessing steps:
        1. Load CSV
        2. Extract features and target
        3. Split into train/val/test
        4. Normalize data
        5. Create DataLoaders
        Returns:
            train_loader, val_loader, test_loader
        """
        df = self._load_csv()
        X, y = self._extract_features_and_target(df)
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_train_val_test(X, y)
        X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm = self._normalize_train_val_test(X_train, X_val, X_test, y_train, y_val, y_test)
        train_loader, val_loader, test_loader = self._create_dataloaders(X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm)
        return train_loader, val_loader, test_loader

    def get_input_dim(self):
        """Return number of features dynamically."""
        return len(self.FEATURE_COLUMNS)
    
    def save_normalization_params(self):
        """Save normalization stats (mean, std) to a file."""
        os.makedirs(os.path.dirname(config.NORM_PARAMS_PATH), exist_ok=True)
        with open(config.NORM_PARAMS_PATH, "wb") as f:
            pickle.dump(self.norm_params, f)
        print(f"• Normalization parameters saved: {config.NORM_PARAMS_PATH}")

    def load_normalization_params(self):
        """
        Load saved normalization scalers from file.
        Updates self.x_scaler and self.y_scaler accordingly.
        """
        with open(config.NORM_PARAMS_PATH, "rb") as f:
            self.norm_params = pickle.load(f)
        
        self.x_scaler = self.norm_params["x_scaler"]
        self.y_scaler = self.norm_params["y_scaler"]
        print(f"• Normalization parameters loaded: {config.NORM_PARAMS_PATH}")

    # ----------------- PRIVATE METHODS -----------------
    def _load_csv(self):
        """Load CSV into a pandas DataFrame."""
        df = pd.read_csv(config.DATA_PATH)
        print(f"• Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def _extract_features_and_target(self, df):
        """Extract features and target tensors from DataFrame."""
        X = torch.tensor(df[self.FEATURE_COLUMNS].values, dtype=torch.float32)
        y = torch.tensor(df[self.TARGET_COLUMN].values, dtype=torch.float32).reshape((-1, 1))
        return X, y

    def _split_train_val_test(self, X, y):
        """Split data into train/validation/test sets using config ratios."""

        if not config.SPLIT_DATASET:
            print("• Dataset splitting disabled. Using full dataset for train/val/test")
            return X, X, X, y, y, y

        n_total = len(X)
        n_train = int(config.TRAIN_SPLIT * n_total)
        n_val = int(config.VAL_SPLIT * n_total)
        n_test = n_total - n_train - n_val

        dataset = torch.utils.data.TensorDataset(X, y)
        generator = torch.Generator().manual_seed(config.RANDOM_SEED) if config.RANDOM_SEED is not None else None
        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

        # Extract tensors back from TensorDataset
        X_train, y_train = train_ds[:][0], train_ds[:][1]
        X_val, y_val = val_ds[:][0], val_ds[:][1]
        X_test, y_test = test_ds[:][0], test_ds[:][1]

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _normalize_train_val_test(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Normalize features and target using StandardScaler fitted on training set,
        according to config flags.
        """
        # -------------------------
        # Normalize features (X)
        # -------------------------
        if config.NORMALIZE_FEATURES:
            X_train_norm = torch.tensor(self.x_scaler.fit_transform(X_train), dtype=torch.float32)
            X_val_norm = torch.tensor(self.x_scaler.transform(X_val), dtype=torch.float32)
            X_test_norm = torch.tensor(self.x_scaler.transform(X_test), dtype=torch.float32)
        else:
            X_train_norm, X_val_norm, X_test_norm = X_train, X_val, X_test

        # -------------------------
        # Normalize target  (y): Only do this for regression, NOT classification!
        # -------------------------
        if config.NORMALIZE_TARGETS:
            y_train_norm = torch.tensor(self.y_scaler.fit_transform(y_train), dtype=torch.float32)
            y_val_norm = torch.tensor(self.y_scaler.transform(y_val), dtype=torch.float32)
            y_test_norm = torch.tensor(self.y_scaler.transform(y_test), dtype=torch.float32)
        else:
            y_train_norm, y_val_norm, y_test_norm = y_train, y_val, y_test

        # Save scalers for later use
        self.norm_params = {"x_scaler": self.x_scaler, "y_scaler": self.y_scaler}

        return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm

    def _create_dataloaders(self,  X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Return train, val, test DataLoaders.
        """

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        test_ds = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

        return train_loader, val_loader, test_loader