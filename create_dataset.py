import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
from transformations import weak_transform

class CustomDataset(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)  # Get image and label
        pil_image = transforms.ToPILImage()(img)  # Convert to PIL image
        img = weak_transform(pil_image)  # Apply your weak transformation
        return img, target
    
# CIFAR-10 dataset with labeled and unlabeled splits
def create_cifar10_datasets():
    train_dataset = CustomDataset(root='./data', train=True, download=True, transform=weak_transform)
    test_dataset = CustomDataset(root='./data', train=False, download=True, transform=weak_transform)

    # Get the indices for each class
    indices = np.arange(len(train_dataset.targets))
    labeled_indices = []
    unlabeled_indices = []
    
    for i in range(10):
        class_indices = np.where(np.array(train_dataset.targets) == i)[0]
        # Use 1% of the data as labeled
        n_labeled = int(0.01 * len(class_indices))
        labeled_indices.extend(class_indices[:n_labeled])
        unlabeled_indices.extend(class_indices[n_labeled:])
    
    # Create subsets for labeled and unlabeled data
    labeled_dataset = Subset(train_dataset, labeled_indices)
    unlabeled_dataset = Subset(train_dataset, unlabeled_indices)
    
    return labeled_dataset, unlabeled_dataset, test_dataset