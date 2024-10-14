import torch
from torchvision import transforms

# Transformations for weak and strong augmentations
weak_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

strong_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def apply_transform_to_batches(data, transform):
    # List to hold the augmented images
    augmented_images = []
    for i in range(data.size(0)): 
        single_image = data[i] 
        pil_image = transforms.ToPILImage()(single_image)
        weak_aug_image = transform(pil_image)
        augmented_images.append(weak_aug_image)
    
    return torch.stack(augmented_images)