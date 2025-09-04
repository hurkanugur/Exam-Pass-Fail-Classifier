import torch
import torch.nn as nn
import config

class ExamClassificationModel(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1) 
        )

        self.net.apply(self.init_weights)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.net(x)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def save(self):
        """Save model state_dict using the path from config."""
        torch.save(self.state_dict(), config.MODEL_PATH)
        print(f"• Model saved to {config.MODEL_PATH}")

    def load(self):
        """Load model state_dict using the path from config."""
        self.load_state_dict(torch.load(config.MODEL_PATH, map_location=self.device))
        self.to(self.device)
        print(f"• Model loaded from {config.MODEL_PATH}")