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
import matplotlib.pyplot as plt
import numpy as np
from pypso import pso
## Profiling
import cProfile, pstats, io
import pandas as pd

#%%
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.d1 = nn.Linear(26*26*32,128)
        self.d2 = nn.Linear(128,10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(-1, 26*26*32)
        x = self.d1(x)
        return self.d2(x)
class MLP(nn.Module):
    def __init__(self,N_input,N_hidden,N_output):
        super(MLP,self).__init__()
        self.d1 = nn.Linear(N_input,N_hidden)
        self.d2 = nn.Linear(N_hidden,N_output)
    def forward(self,x):
        x = self.d1(x)
        x = F.relu(x)
        return self.d2(x)
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#%% Data and training parameters
batch_size =32
test_batch_size =32

train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=batch_size, shuffle=True,)
test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=test_batch_size, shuffle=True,)

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

#%%
def train_fullbackprog(model, device, train_loader, optimizer, epoch):
    model.train()
    torch_device = torch.device(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(torch_device), target.to(torch_device)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))   

#%%
# Creating dataset
batch_size =32
# class reduced MNIST dataset

class reduced_MNIST(Dataset):
    def __init__(self,train_loader,Nsamples):
        self.Nsamples = Nsamples
        #self.data = train_loader.dataset.data[0:Nsamples-1]
        #self.target = train_loader.dataset.targets[0:Nsamples-1]
    def __len__(self):
        return self.Nsamples
    def __getitem__(self, idx):
        image, label = train_loader.dataset[idx]
        #image.unsqueeze_(0)
        #label = self.target[idx]
        return image,label
train_reduced_MNIST = reduced_MNIST(train_loader,10000)
trainloader_reduced_MNIST = DataLoader(train_reduced_MNIST,batch_size=batch_size,shuffle=True)

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
train_ionosphere = ionosphere_uci()
batch_size = 2
trainloader_ionosphere = DataLoader(train_ionosphere,batch_size=batch_size,shuffle=True,drop_last=True)

#%%
#PSO training
#torch.set_num_threads(1)
device = "cpu"
epochs = 50
Nparticles = 10
w = 1; c1=2 ; c2=2
#net = Net().to(torch.device(device))
net = MLP(34,10,2).to(torch.device(device))
pso_method = pso(net,Nparticles,device=device, lossfn_name = 'MSE', classes=2, debug=False) # instantiate class
print("Initializing particles")
pso_method.init_pso_part()
print("Init done")

pr = cProfile.Profile() # Profiling
pr.enable()
for epoch in range(1, epochs + 1):
    #pso_method.pso(train_loader,w,c1,c2,epoch)
    #pso_method.pso(trainloader_reduced_MNIST,w,c1,c2,epoch)
    pso_method.pso(trainloader_ionosphere,w,c1,c2,epoch)
    #train(model, device, train_loader, optimizer, epoch)
    #test(model, device, test_loader)
pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

#%% Backpropagation
device = "cpu"
epochs = 50
lr = 0.01
momentum = 0.5
net = MLP(34,10,2).to(torch.device(device))
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
#optimizer = optim.Adam(model.parameters(),lr=lr)

pr = cProfile.Profile() # Profiling
pr.enable()
for epoch in range(1, epochs + 1):
    train_fullbackprog(net, device, trainloader_ionosphere, optimizer, epoch)
    #test(model, device, test_loader)
pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())


# %%
#PSO vs Backpropagation