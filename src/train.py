import torch
import config
from visualize import LossMonitor

def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    loss_monitor: LossMonitor,
):
    """
    Train a PyTorch model with optional validation and live loss monitoring.

    Args:
        model (torch.nn.Module): the neural network to train
        train_loader (DataLoader): DataLoader for training dataset
        val_loader (DataLoader): DataLoader for validation dataset
        optimizer (torch.optim.Optimizer): optimizer for training
        loss_fn (callable): loss function (e.g., BCEWithLogitsLoss)
        device (torch.device): device to run training on (CPU or GPU)
        loss_monitor (LossMonitor, optional): instance to dynamically plot training/validation loss

    Notes:
        - Validation is computed every config.VAL_INTERVAL epochs.
        - train_loss is always updated; val_loss is updated only when validation is run.
    """
    
    last_val_loss = None  # to carry forward for plotting

    for epoch in range(config.NUM_EPOCHS):
        # -------------------------
        # Training step
        # -------------------------
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # -------------------------
        # Validation step
        # -------------------------
        val_loss = None
        if (epoch + 1) % config.VAL_INTERVAL == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    val_loss += loss.item()
            last_val_loss = val_loss

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}")

        # -------------------------
        # Update LossMonitor
        # -------------------------
        if loss_monitor is not None:
            loss_monitor.update(train_loss, last_val_loss)