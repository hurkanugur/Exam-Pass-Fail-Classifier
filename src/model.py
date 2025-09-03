# model.py
import torch
import torch.nn as nn
import config

class ExamClassifier(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        return self.net(x)

    def save(self):
        """Save model state_dict using the path from config."""
        torch.save(self.state_dict(), config.MODEL_PATH)
        print(f"Model saved to {config.MODEL_PATH}")

    def load(self):
        """Load model state_dict using the path from config."""
        self.load_state_dict(torch.load(config.MODEL_PATH, map_location=self.device))
        self.to(self.device)
        self.eval()
        print(f"Model loaded from {config.MODEL_PATH}")