import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loader(batch_size: int = 32, split: str = "test", seed: int = 0):
    torch.manual_seed(seed)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.CIFAR10(
        root='./data', 
        train=(split == "train"), 
        download=True, 
        transform=transform
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed)
    )
    
    return loader