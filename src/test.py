import torch
from config import CLASSIFICATION_THRESHOLD

def test_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Evaluate a trained classification model on a test dataset.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        test_loader (DataLoader): DataLoader providing test samples.
        device (torch.device): Device to run evaluation on (CPU or GPU).

    Returns:
        float: Test accuracy in the range [0, 1].
    """
    model.eval()
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > CLASSIFICATION_THRESHOLD).float()

            correct_predictions += (predictions == y_batch).sum().item()
            total_samples += y_batch.size(0)

    accuracy = correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy