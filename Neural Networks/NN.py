# Import Libraries
import torch 
import torch.nn.functional as F

import torch.nn as nn # neural networks module
import torch.utils
from torch.utils.data import DataLoader # turns data into batches 
import torchvision.datasets as datasets
import torchvision.transforms as transforms # transformations on the data

# CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Data 
batch_size = 128

training_data = datasets.MNIST(
    root='./datasets', train=True, transform=transforms.ToTensor(), download=True
    
)
train_loader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True
)

testing_data = datasets.MNIST(
    root='./datasets', train=False, transform=transforms.ToTensor(), download=True
)
test_loaders = DataLoader(
    testing_data, batch_size=batch_size, shuffle=False
)

# Build model
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Training loop
input_size = 784 # 28x28 = 784, size of MNIST image
num_classes = 10
num_epochs = 5
learning_rate = 0.01

model = NeuralNet(input_size=input_size, num_classes=num_classes)
model.to(device)

loss_fn = nn.CrossEntropyLoss() # Loss Function
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=learning_rate)  # Adam optimizer

for epoch in range(num_epochs):
    print(f'Epoch: {epoch+1}')
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)# ([128, 1, 28, 28])
        inputs = inputs.reshape(inputs.shape[0], -1) # ([128, 784])
        
        # model training mode
        model.train()
        
        # forward propogation
        optimizer.zero_grad() # zero your gradient for every batch
        y_pred = model(inputs) # make predictions for this batch
        
        #loss and its gradient
        loss = loss_fn(y_pred, labels)
        
        # backpropagation
        loss.backward()
        
        # adam step
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if batch_idx % 1000 == 0:
            print(f'Training loss: {running_loss/10:.2f}')
            running_loss = 0.0
    print('Finished training')

# testing loop
num_correct = 0
num_smaples = 0
# model evaluation mode
model.eval()
with torch.inference_mode():
    for inputs, labels in test_loaders:
        inputs, labels = inputs.to(device), labels.to(device) #[(128, 1, 28, 28)]
        inputs = inputs.reshape(inputs.shape[0], -1) #[(128, 784)]
        
        outputs = model(inputs)
        _, prediction = outputs.max(1)
        num_correct += (prediction==labels).sum().item()
        num_smaples += prediction.size(0)
        
print(f"{num_correct}/{num_smaples} are correct with accuracy : {float(num_correct)/float(num_smaples)*100}%")