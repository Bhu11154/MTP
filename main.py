import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from CNN import SimpleCNN
from create_dataset import create_cifar10_datasets
from train import train_fixmatch, test_model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets and data loaders
    labeled_dataset, unlabeled_dataset, test_dataset = create_cifar10_datasets()
    
    labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize the model, optimizer, and loss function
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(100):  # Number of epochs
        train_fixmatch(model, labeled_loader, unlabeled_loader, optimizer, device)
        print(f'Epoch {epoch + 1} completed.')
    
    # Test the model and print accuracy
    accuracy = test_model(model, test_loader, device)
    print(f'Testing accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main()
