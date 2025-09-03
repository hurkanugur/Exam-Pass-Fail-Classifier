import torch
import torch.nn as nn
import config
from dataset import create_dataloaders
from model import ExamClassifier
import test
import train
from visualize import LossMonitor

def main():

    # Select CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    # Load data
    train_loader, val_loader, test_loader = create_dataloaders()

    # Initialize model, optimizer, loss
    model = ExamClassifier(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    # Initialize LossMonitor
    loss_monitor = LossMonitor()

    # Train the model 
    train.train_model(model, train_loader, val_loader, optimizer, loss_fn, device, loss_monitor)

    # Test the model
    test.test_model(model, test_loader, device)

    # Save the model
    model.save()

    # Keep the final plot displayed
    loss_monitor.close()


if __name__ == "__main__":
    main()
