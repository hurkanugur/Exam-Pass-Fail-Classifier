import numpy as np
import pandas as pd
import torch
import config
from dataset import ExamDataset
from model import ExamClassificationModel

def main():
    # -------------------------
    # Select CPU or GPU
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    # -------------------------
    # Load normalization parameters
    # -------------------------
    dataset = ExamDataset()
    dataset.load_normalization_params()

    # -------------------------
    # Example real-world input
    # -------------------------
    df = pd.DataFrame([
        {"Study Hours": 4, "Previous Exam Score": 81, "Pass/Fail": 0},
        {"Study Hours": 9, "Previous Exam Score": 72, "Pass/Fail": 1},
        {"Study Hours": 7, "Previous Exam Score": 48, "Pass/Fail": 0},
        {"Study Hours": 6, "Previous Exam Score": 88, "Pass/Fail": 1},
        {"Study Hours": 2, "Previous Exam Score": 81, "Pass/Fail": 0},
    ], dtype=np.float32)

    X = dataset.prepare_data_for_inference(df)
    input_dim = X[0].numel()

    # -------------------------
    # Load trained model
    # -------------------------
    model = ExamClassificationModel(input_dim=input_dim, device=device)
    model.load()

    # -------------------------
    # Make predictions
    # -------------------------
    model.eval()
    X = X.to(device=device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X))
        predictions = (probabilities > config.CLASSIFICATION_THRESHOLD).float()

    # -------------------------
    # Display results
    # -------------------------
    for i, (p, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Student {i+1}: Predicted {'Pass' if p.item() == 1 else 'Fail'}, Probability: {prob.item():.2f}")


if __name__ == "__main__":
    main()
