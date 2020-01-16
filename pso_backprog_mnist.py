#%%
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from pypso import pso

## Profiling
import cProfile, pstats, io
import pandas as pd



#%%
class MLP(nn.Module):
    def __init__(self,N_input,N_hidden,N_output):
        super(MLP,self).__init__()
        self.d1 = nn.Linear(N_input,N_hidden)
        self.d2 = nn.Linear(N_hidden,N_output)
    def forward(self,x):
        #x.view(x.size(0), -1)
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        return torch.sigmoid(x)
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


#%% Data and training parameters
batch_size = 128
test_batch_size = 32

train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=batch_size, shuffle=True,drop_last=True)
test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=test_batch_size, shuffle=True,drop_last=True)

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images_1D = images.view(images.shape[0],-1)
N_input = len(images_1D[0])

# show images
imshow(torchvision.utils.make_grid(images))

#%% Corrected to convert 2D images to 1D
def train_fullbackprog(model, device, train_loader, optimizer, epoch):
    model.train()
    torch_device = torch.device(device)
    loss = nn.CrossEntropyLoss(reduction='sum')
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(torch_device), target.to(torch_device)
        data = data.view(data.shape[0], -1)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        #loss = F.cross_entropy(output,target)
        train_loss = loss(output,target)
        train_loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader)* len(data),
                100. * batch_idx / len(train_loader), train_loss)) 
def test(model, device, test_loader, lossfn_name, number_class):
    model.eval()
    test_loss = 0
    correct = 0
    torch_device = torch.device(device) 
    if lossfn_name == 'MSE':
        t_onehot = torch.FloatTensor(test_loader.batch_size, number_class).to(torch_device) 
        loss = nn.MSELoss(reduction='sum')   
    elif lossfn_name == 'cross_entropy':
        loss = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(torch_device), target.to(torch_device)
            data = data.view(data.shape[0], -1)
            output = model(data)
            if lossfn_name == 'MSE':
                t_onehot.zero_()
                t_onehot.scatter_(1, target.unsqueeze(1), 1)
                test_loss += loss(output, t_onehot) # sum up batch loss
            else:
                test_loss += loss(output, target)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    acc = 100. * correct / (len(test_loader)*test_loader.batch_size)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader)*test_loader.batch_size,acc))
    return acc, test_loss

#%%
device = "cpu"
epochs = 2
Nparticles = 300
w = 1; c1=2 ; c2=2
loss_train_pso = np.zeros(40)
loss_val_pso = np.zeros(40)

# PSO
for i in [50]:
#for i in range(10,100):
    net = MLP(N_input,i,10).to(torch.device(device))
    pso_method = pso(net,Nparticles,train_loader,device=device, 
                        lossfn_name = 'MSE', classes=10, debug=False, verbose = True) # instantiate class
    pso_method.init_pso_part()
    for epoch in range(1, epochs + 1):
        pso_method.pso(w,c1,c2,epoch)
    test(net,device,test_loader,'MSE',10)

# %%
# Backprog
epochs = 2
device = "cpu"
lr = 0.01
momentum = 0.5
acc = np.zeros(100)
test_loss = np.zeros(100)
for i in range(1,100):
    print('Number of hidden neurons: {}'.format(i))
    net = MLP(N_input,i,10).to(torch.device(device))
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    for epoch in range(epochs):
        train_fullbackprog(net, device, train_loader, optimizer, epoch)
    acc[i],test_loss[i] = test(net,device,test_loader,'cross_entropy',10)

# %%
# plot results
hidden_neurons = np.arange(1, 100)

ax = plt.subplot(1,1,1)
line1 = ax.plot(hidden_neurons, acc[1:100], 'r--',label="Backp")
#line2 = ax.plot(t[0:39], test_acc_pso[0:39], 'b-',label="PSO")
ax.legend()
plt.xlabel("# hidden neurons")
plt.ylabel("test accuracy")
fig = plt.gcf()
plt.show()
fig.savefig('acc_backp_mnist.png')


# %%
