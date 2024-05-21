# Import libraries
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms

# Cuda
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading data
batch_size = 20

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

training_data = torchvision.datasets.CIFAR10(
    root="./datasets", train=True, transform=transform, download=True
)
train_loaders = torch.utils.data.DataLoader(
    training_data, batch_size=batch_size, shuffle=True
)

testing_data = torchvision.datasets.CIFAR10(
    root="./datasets", train=False, transform=transform, download=True
)
test_loaders = torch.utils.data.DataLoader(
    testing_data, batch_size=batch_size, shuffle=False
)

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

examples = iter(train_loaders)
img, labels = next(examples)
# print(img.shape)


# Build model
class cnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.network(x)


model = cnnModel()

# Training loop
def train(model, loss_fn, optimizer, train_dl, num_epoch, learning_rate):
    optim = optimizer(model.parameters(), lr=learning_rate)
    train_loss = 0
    num_correct = 0
    num_smaples = 0
    for epoch in range(num_epoch):
        model.train()
        for i, (images, labels) in enumerate(train_dl):
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()  # delta_loss/ delta_model_parameters
            optim.step()
            train_loss += loss.item()
            # accuracy
            _, prediction = logits.max(1)
            num_correct += (prediction == labels).sum().item()
            num_smaples += prediction.size(0)
            accuracy = float(num_correct) / float(num_smaples)*100
        print(
            f"Epoch: {epoch+1} | Training loss: {train_loss/len(train_dl):.2f} | Accuracy: {accuracy:.2f}"
        )
    print("Training Finished!")


model.to(device)
learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam
num_epochs = 5

train(
    model=model,
    loss_fn=loss_func,
    optimizer=optim,
    train_dl=train_loaders,
    num_epoch=num_epochs,
    learning_rate=learning_rate,
)

# Testing loop
test_loss = 0.0
num_correct = 0
num_smaples = 0
with torch.inference_mode():
    model.eval()
    for i, (data, labels) in enumerate(test_loaders):
        data, labels = data.to(device), labels.to(device)
        y_pred = model(data)
        loss = loss_func(y_pred, labels)
        test_loss += loss.item()
        _, prediction = y_pred.max(1)
        num_correct += (prediction == labels).sum().item()
        num_smaples += prediction.size(0)
        accuracy = float(num_correct)/float(num_smaples)*100
print(f"Test Loss: {test_loss/len(test_loaders):.4}") 
print(
    f"{num_correct}/{num_smaples} correctly classified | accuracy : {accuracy:.2f}%"
)