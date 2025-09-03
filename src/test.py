import torch

from config import CLASSIFICATION_THRESHOLD

def test_model(model, test_loader, device):
    """
    Evaluate the model on a test dataset.

    Args:
        model (torch.nn.Module): trained model
        test_loader (DataLoader): DataLoader for test dataset
        device (torch.device): device to run evaluation on (CPU or GPU)

    Returns:
        float: test accuracy (0 to 1)
    """

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.sigmoid(outputs) > CLASSIFICATION_THRESHOLD
            correct += (preds.type(torch.float32) == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    print(f"Final Test Accuracy: {accuracy:.2f}")
    return accuracy
