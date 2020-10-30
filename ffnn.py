def digit2words(i): 
    x=["zero", "one","two","three","four","five","six","seven","eight","nine","ten"]
    return x[i]

print(digit2words(1))
print(digit2words(7))

def num2words(i):
    if (i < 10):
        return digits2words(i)
    else:
        wordmap = {
            11: "eleven",
            12: "twelve",
            13: "thirteen",
            14: "fourteen",
            15: "fifteen",
            16: "sixteen",
            17: "seventeen",
            18: "eighteen",
            19: "nineteen",
            20:"twenty",
            30:"thirty",
            40:"forty",
            50:"fifty",
            60:"sixty",
            70:"seventy",
            80:"eighty",
            90:"ninety"
        }
        
        res = wordmap.get(i,-1)
        if(res != -1):
            return res
        else:
            # we know it must be between 21 and 99
            (quo,rem) = divmod(i, 10)
            return (wordmap.get(quo*10) +  " " + digit2words(rem))
        
print(num2words(19))
print(num2words(27))
print(num2words(33))





import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

# REGULAR MNIST
train1_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test1_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)

# Fashion MNIST
train2_dataset = dsets.FashionMNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test2_dataset = dsets.FashionMNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor(),
                            download=True)

# CIFAR 10
train3_dataset = dsets.CIFAR10(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test3_dataset = dsets.CIFAR10(root='./data',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)

#USPS
train4_dataset = dsets.USPS(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test4_dataset = dsets.USPS(root='./data',
                           train=False,
                           transform=transforms.ToTensor(),
                          download=True)
                        
train_dataset = train1_dataset
test_dataset = test1_dataset

batch_size = 100
n_iters = 2000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity
        self.sigmoid = nn.Sigmoid()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)
        # Non-linearity  # NON-LINEAR
        out = self.sigmoid(out)
        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out
        
    input_dim = 28*28
hidden_dim = 100
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(model.parameters())
print(len(list(model.parameters())))
# Hidden Layer Parameters
print(list(model.parameters())[0].size())
# FC1 Bias Parameters
print(list(model.parameters())[1].size())
# FC2 Parameters
print(list(model.parameters())[2].size())
# FC2 Bias Parameters
print(list(model.parameters())[3].size())


iter = 0
prev_accuracy = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variable
        images = Variable(images.view(-1, input_dim))
        labels = Variable(labels)
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        
        # Forward pass to get output/logits
        outputs = model(images)
        
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()
        
        iter += 1
        
        
        if iter % 10 == 0:
            # Calculate Accuarcy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Load images to a Torch Variable
                images = Variable(images.view(-1, input_dim))
                
                # Forward pass only to get logits/output
                outputs = model(images)
                
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                # Total number of labels
                total += labels.size(0)
                
                # Total correct predictions
                correct += (predicted == labels).sum()
                
            accuracy = 100 * correct // total
            
                # Print Loss
            if (accuracy > prev_accuracy):
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
                prev_accuracy = accuracy

                
print("All done")

from idx_tools import Idx
import matplotlib.pyplot as plt
import numpy as np
mnist_data = Idx.load_idx('./data/MNIST/raw/train-images-idx3-ubyte')
# Plot a random image
for i in range(10):
    plt.figure()
    plt.imshow(mnist_data[i], cmap='gray')
