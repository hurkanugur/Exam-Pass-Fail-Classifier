import torch
import config
from dataset import create_dataloaders
from model import ExamClassifier

def main():

    # -------------------------
    # Select CPU or GPU
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    # -------------------------
    # Load trained model
    # -------------------------
    model = ExamClassifier(device=device)
    model.load()

    # -------------------------
    # Example real-world input
    # Each row = [Study Hours, Previous Exam Score]
    # Could come from user input, API, or production system
    # -------------------------
    X_real = torch.tensor([
        [5.0, 70.0],
        [2.5, 55.0],
        [8.0, 90.0]
    ], dtype=torch.float32)

    # -------------------------
    # Make predictions
    # -------------------------
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_real))
        predictions = (probabilities > config.CLASSIFICATION_THRESHOLD).float()

    # -------------------------
    # Display results
    # -------------------------
    for i, (p, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Student {i+1}: Predicted {'Pass' if p.item() == 1 else 'Fail'}, Probability: {prob.item():.2f}")


if __name__ == "__main__":
    main()
