import torch
import torch.nn.functional as F
from transformations import weak_transform, strong_transform, apply_transform_to_batches

# Loss function
def fixmatch_loss(labeled_preds, labels, unlabeled_preds, pseudo_labels, threshold=0.75):
    ce_loss = F.cross_entropy(labeled_preds, labels)
    
    # Only consider pseudo-labels with high confidence
    mask = torch.max(F.softmax(unlabeled_preds, dim=1), dim=1)[0] > threshold
    pseudo_loss = F.cross_entropy(unlabeled_preds[mask], pseudo_labels[mask])
    
    return ce_loss + pseudo_loss

# Training loop
def train_fixmatch(model, labeled_loader, unlabeled_loader, optimizer, device):
    model.train()
    
    for (labeled_data, labels), (unlabeled_data, _) in zip(labeled_loader, unlabeled_loader):
        # Move data to the device (GPU/CPU)
        labeled_data, labels = labeled_data.to(device), labels.to(device)
        unlabeled_data = unlabeled_data.to(device)
        
        # Generate weak and strong augmentations for unlabeled data
        weak_aug = apply_transform_to_batches(unlabeled_data, weak_transform)
        strong_aug = apply_transform_to_batches(unlabeled_data, strong_transform)
        
        # Pseudo-labeling
        with torch.no_grad():
            pseudo_labels = torch.argmax(model(weak_aug), dim=1)
        
        # Forward pass for labeled and unlabeled data
        labeled_preds = model(labeled_data)
        unlabeled_preds = model(strong_aug)
        
        # Compute FixMatch loss
        loss = fixmatch_loss(labeled_preds, labels, unlabeled_preds, pseudo_labels)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test function
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
