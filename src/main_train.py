import torch
import torch.nn as nn
import config
from dataset import ExamDataset
from model import ExamClassificationModel
from visualize import LossMonitor

def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, loss_monitor):
    """Train a PyTorch model with optional validation and live loss monitoring."""
    
    last_val_loss = None

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # -------------------------
        # Training Step
        # -------------------------
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # -------------------------
        # Validation Step
        # -------------------------
        val_loss = None
        if epoch % config.VAL_INTERVAL == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            last_val_loss = val_loss
            print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}")
        else:
            print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.8f}")

        # -------------------------
        # Update Training/Validation Loss Graph
        # -------------------------
        loss_monitor.update(train_loss, val_loss)


def test_model(model, test_loader, device):
    """Evaluate a trained classification model on a test dataset."""
    model.eval()
    total_samples = 0
    correct_predictions = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            probabilities = torch.sigmoid(output)
            predictions = (probabilities > config.CLASSIFICATION_THRESHOLD).float()
            correct_predictions += (predictions == y_batch).sum().item()
            total_samples += y_batch.size(0)

    accuracy = correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")



def main():

    # Select CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    # Load and prepare data
    exam_dataset = ExamDataset()
    train_loader, val_loader, test_loader = exam_dataset.prepare_data_for_training()

    # Initialize model, optimizer, loss
    input_dim = exam_dataset.get_input_dim(train_loader)
    model = ExamClassificationModel(input_dim=input_dim, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    # Initialize LossMonitor
    loss_monitor = LossMonitor()

    # Train the model 
    train_model(model, train_loader, val_loader, optimizer, loss_fn, device, loss_monitor)

    # Test the model
    test_model(model, test_loader, device)

    # Save the model and normalization parameters
    model.save()
    exam_dataset.save_normalization_params()

    # Keep the final plot displayed
    loss_monitor.close()


if __name__ == "__main__":
    main()
