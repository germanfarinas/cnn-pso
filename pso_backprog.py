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



# %%
class MLP(nn.Module):
    def __init__(self,N_input,N_hidden,N_output):
        super(MLP,self).__init__()
        self.d1 = nn.Linear(N_input,N_hidden)
        self.d2 = nn.Linear(N_hidden,N_output)
    def forward(self,x):
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        return torch.sigmoid(x)

class ionosphere_uci(): # dataset ionosphere from uci
    def __init__(self):
        file = "ionosphere.data"
        self.data = pd.read_csv(file,delim_whitespace=False,header=None)
        self.data = np.array(self.data)   
        np.random.shuffle(self.data) 
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        features = np.array(self.data[idx, 0:34], dtype = "float32")
        label = self.data[idx,34]
        label = 1 if label == 'g' else 0
        return torch.from_numpy(features), label

dataset = ionosphere_uci()
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_batch_size = 64
test_batch_size = 4

train_loader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler, drop_last=True)
validation_loader = DataLoader(dataset, batch_size=test_batch_size, sampler=valid_sampler, drop_last=True)

#%%
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
        loss = nn.MSELoss()   
    elif lossfn_name == 'cross_entropy':
        loss = nn.CrossEntropyLoss(reduction = 'sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(torch_device), target.to(torch_device)
            data = data.view(data.shape[0], -1)
            output = model(data)
            if lossfn_name == 'MSE':
                t_onehot.zero_()
                t_onehot.scatter_(1, target.unsqueeze(1), 1)
                test_loss += loss(output, t_onehot).item() # sum up batch loss
            else:
                test_loss += loss(output, target)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader)*test_loader.batch_size,
        100. * correct / (len(test_loader)*test_loader.batch_size)))


#%%
device = "cpu"
epochs = 20
Nparticles = 300
w = 1; c1=2 ; c2=2
loss_train_pso = np.zeros(40)
loss_val_pso = np.zeros(40)

## PSO
for i in range(10,20):
    net = MLP(34,i,2).to(torch.device(device))
    pso_method = pso(net,Nparticles,train_loader,device=device, 
                        lossfn_name = 'MSE', classes=2, debug=False, verbose = True) # instantiate class
    pso_method.init_pso_part()
    for epoch in range(1, epochs + 1):
        pso_method.pso(w,c1,c2,epoch)
    test(net,device,validation_loader,'MSE',2)

# %%
# Backprog
epochs = 100
device = "cpu"
lr = 0.001
momentum = 0.5
net = MLP(34,10,2).to(torch.device(device))
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
for epoch in range(epochs):
    train_fullbackprog(net, device, train_loader, optimizer, epoch)
test(net,device,validation_loader,'MSE',2)

# %%
