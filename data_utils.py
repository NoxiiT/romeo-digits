import torch
import torchvision
import torchvision.transforms as transforms


def split_dataset(dataset, train_pct, val_pct):
    total = len(dataset)
    train_size = int(total * train_pct)
    val_size = int(total * val_pct)
    test_size = total - train_size - val_size
    # Pour éviter test_size négatif si train_pct+val_pct>1
    if test_size < 0:
        val_size = total - train_size
        test_size = 0
    splits = [train_size, val_size, test_size]
    # Retire les splits à 0 pour éviter erreur
    splits = [s for s in splits if s > 0]
    return torch.utils.data.random_split(dataset, splits, generator=torch.Generator().manual_seed(42))


def get_mnist_dataloaders(batch_size, train_pct=0.8, val_pct=0.1, num_workers=0):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    splits = split_dataset(full_train_set, train_pct, val_pct)
    if len(splits) == 3:
        train_set, val_set, _ = splits
    elif len(splits) == 2:
        train_set, val_set = splits
    else:
        train_set = splits[0]
        val_set = None
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_set else None
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, 10


def get_fashionmnist_dataloaders(batch_size, train_pct=0.8, val_pct=0.1, num_workers=0):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_train_set = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    splits = split_dataset(full_train_set, train_pct, val_pct)
    if len(splits) == 3:
        train_set, val_set, _ = splits
    elif len(splits) == 2:
        train_set, val_set = splits
    else:
        train_set = splits[0]
        val_set = None
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_set else None
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, 10


def get_cifar10_dataloaders(batch_size, train_pct=0.8, val_pct=0.1, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    splits = split_dataset(full_train_set, train_pct, val_pct)
    if len(splits) == 3:
        train_set, val_set, _ = splits
    elif len(splits) == 2:
        train_set, val_set = splits
    else:
        train_set = splits[0]
        val_set = None
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_set else None
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, 10
