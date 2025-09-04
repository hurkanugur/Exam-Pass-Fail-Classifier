import torch
import config
from dataset import ExamDataset
from model import ExamClassifier

def main():
    # -------------------------
    # Select CPU or GPU
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    # -------------------------
    # Load normalization parameters
    # -------------------------
    exam_dataset = ExamDataset()
    exam_dataset.load_normalization_params()

    # -------------------------
    # Load trained model
    # -------------------------
    input_dim = exam_dataset.get_input_dim()  # dynamic input dim
    model = ExamClassifier(input_dim, device=device)
    model.load()

    # -------------------------
    # Example real-world input (raw)
    # -------------------------
    X_real = torch.tensor([
        [5.0, 70.0],
        [2.5, 55.0],
        [8.0, 90.0]
    ], dtype=torch.float32)

    # -------------------------
    # Normalize input if enabled
    # -------------------------
    if config.NORMALIZE_FEATURES:
        X_real_norm = exam_dataset.x_scaler.transform(X_real.numpy())
        X_real_norm = torch.tensor(X_real_norm, dtype=torch.float32).to(device)
    else:
        X_real_norm = X_real.to(device)

    # -------------------------
    # Make predictions
    # -------------------------
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_real_norm))
        predictions = (probabilities > config.CLASSIFICATION_THRESHOLD).float()

    # -------------------------
    # Display results
    # -------------------------
    for i, (p, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Student {i+1}: Predicted {'Pass' if p.item() == 1 else 'Fail'}, Probability: {prob.item():.2f}")


if __name__ == "__main__":
    main()
