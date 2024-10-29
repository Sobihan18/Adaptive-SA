from typing import Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_encoder_decoder(dataset: str, latent_size: int) -> Tuple[nn.Module, nn.Module]:
    if dataset == 'CIFAR10':
        return get_cifar10_encoder_decoder(latent_size)
    elif dataset == 'FMNIST':
        return get_fmnist_encoder_decoder(latent_size)
    else:
        raise ValueError(f"Dataset '{dataset}' is not supported.")


def load_dataset(params: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    transform_list = [
        transforms.ToTensor()
    ]
    if params['dataset'] == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose(transform_list))
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose(transform_list))
    elif params['dataset'] == 'FMNIST':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose(transform_list))
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.Compose(transform_list))
    else:
        raise ValueError(f"Dataset '{params['dataset']}' is not supported.")

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)

    return train_loader, test_loader



def get_cifar10_encoder_decoder(latent_size: int) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # Input: 3x32x32, Output: 32x16x16
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x8x8
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x4x4
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU()
    )
    fc_mu = nn.Linear(256, latent_size)
    fc_logvar = nn.Linear(256, latent_size)

    decoder = nn.Sequential(
        nn.Linear(latent_size, 256),
        nn.ReLU(),
        nn.Linear(256, 128 * 4 * 4),
        nn.ReLU(),
        nn.Unflatten(1, (128, 4, 4)),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x8x8
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32x16x16
        nn.ReLU(),
        nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # Output: 3x32x32
        nn.Sigmoid()
    )

    return encoder, fc_mu, fc_logvar, decoder


def get_fmnist_encoder_decoder(latent_size: int) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    encoder = nn.Sequential(
        nn.Linear(784, 400),
        nn.ReLU()
    )
    fc_mu = nn.Linear(400, latent_size)
    fc_logvar = nn.Linear(400, latent_size)

    decoder = nn.Sequential(
        nn.Linear(latent_size, 400),
        nn.ReLU(),
        nn.Linear(400, 784),
        nn.Sigmoid()
    )

    return encoder, fc_mu, fc_logvar, decoder


