# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading data
batch_size = 4

training_data = datasets.CIFAR10(
    root='./datasets', train=True, transform=transforms.ToTensor(), download=True
)
train_loaders = DataLoader(
    training_data, batch_size=batch_size, shuffle=True
)

testing_data = datasets.CIFAR10(
    root='./datasets', train=False, transform=transforms.ToTensor(), download=True
)
test_loaders = DataLoader(
    testing_data, batch_size=batch_size, shuffle=False
)

