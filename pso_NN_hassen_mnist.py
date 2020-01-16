#%%
from __future__ import print_function
import sys
#sys.path.append("/home/wuchty/combpso/")
import numpy as np
#from swarm import Swarm, Particle
from algorithms.swarm import Swarm, Particle
#import config
from algorithms import config

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from pypso import pso
## Profiling
import cProfile, pstats, io
import pandas as pd

#%%
# MLP model and dataset using pytorch
class MLP(nn.Module):
    def __init__(self,N_input,N_hidden,N_output):
        super(MLP,self).__init__()
        self.d1 = nn.Linear(N_input,N_hidden)
        self.d2 = nn.Linear(N_hidden,N_output)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        #x.view(x.size(0), -1)
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        #x = torch.sigmoid(x)
        x = self.softmax(x)
        return x
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

batch_size = int(60000/10)
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

# %%
class train_net_pso():
    def __init__(self,model,Nparticles,train_loader,lossfn_name='cross_entropy',number_class=10,epochs=2,
                            dim_reduction = None):
        self.model = model
        self.train_loader = train_loader
        self.epochs = epochs
        self.Nparticles = Nparticles
        self.lossfn_name = lossfn_name
        self.emb_dim = dim_reduction
        self.number_weights = len(self.convert_param2vector())
        if self.emb_dim is not None: # create embedding matrix for decoder for low dim to high dim
            self.emb_matrix = np.random.rand(self.emb_dim, self.number_weights) 
        if self.lossfn_name == 'MSE':
            self.loss = nn.MSELoss(reduction = 'sum') 
        elif self.lossfn_name == 'MAE':  
            self.loss = nn.L1Loss(reduction = 'sum')
        elif self.lossfn_name == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss(reduction = 'sum')
        self.device = device
        self.torch_device = torch.device(device)
        # creating one hot vector for loss functions other than cross entropy
        if self.lossfn_name != 'cross_entropy':
            self.t_onehot = torch.FloatTensor(self.train_loader.batch_size, number_class).to(self.torch_device)
    def convert_param2vector(self):
        param_vect = parameters_to_vector(self.model.parameters())
        return param_vect.detach().numpy()
    def convert_vector2param(self,vec): # load vector to net (torch model) parameters.
        vect = torch.from_numpy(vec).float()[0] if self.emb_dim is not None else torch.from_numpy(vec).float()
        #vect = torch.from_numpy(vec).float() 
        vector_to_parameters(vect, self.model.parameters())
    def decoder(self, vec_Ldim):
        x = np.matrix(vec_Ldim)
        x = x*self.emb_matrix
        return x  ## return a torch tensor/vector ready for vector_to_parameter method
    def classification_perfomance(self,p,data,target):
        if self.emb_dim is not None: # check if dim_reduction is used, if so decode low dimension vector to high dim params 
            self.convert_vector2param(self.decoder(p.x))
        else:
            self.convert_vector2param(p.x) 
        with torch.no_grad(): # do not track operations for gradient calculation
            #data, target = data.to(self.torch_device), target.to(self.torch_device)
            if self.lossfn_name == "cross_entropy":
                output = self.model(data)
                cost = self.loss(output,target)
            elif (self.lossfn_name == "MSE") or (self.lossfn_name == "MAE"):
                self.t_onehot.zero_()
                self.t_onehot.scatter_(1, target.unsqueeze(1), 1)
                output = self.model(data) 
                cost = self.loss(output,self.t_onehot) #+ 0.1*np.dot(p.x,p.x) # added regularization
                if cost == 6000:
                    #print('t_onehot: {}, cost: {}, ouput: {}'.format(self.t_onehot,cost,output))
                    print('output is zero ?: {}'.format(output.sum() == 0))
                    output = self.model(data)
            #loss = F.mse_loss(output,target)
        return cost, cost

    def print_stat_swarm(self):
        df = pd.DataFrame([p.x, p.v, p._pbest_x, p._pbest_cost, p._pbest_cp] for p in self.sw._P)
        print(df)
        print('particle size = {}'.format(len(self.sw._P[0].x)))
    
    def init_swarm(self, particles=None):
        if particles is not None:
            self.Nparticles = particles
        # config parameters
        config.particle_size = self.emb_dim if self.emb_dim is not None else self.number_weights
        config.feature_init = int(config.particle_size/2) # number of weights that goes to xmax
        config.swarm_size = self.Nparticles
        config.fitness_function = self.classification_perfomance
        config.max_nbr_iter = len(self.train_loader)*self.epochs
        config.swarm_reinit_frac = 0.5
        # initialize the swarm
        self.sw = Swarm()
        self.sw._gbest_cost = 100000 # an arbitrary large number
        swarm_size = self.Nparticles
        i = 0
        dataiter = iter(self.train_loader)
        data, target = dataiter.next()
        data, target = data.to(self.torch_device), target.to(self.torch_device)
        data = data.view(data.shape[0], -1) # This is for 2D data, if 1D then comment this line
        while i < swarm_size:
            p = Particle(sw=self.sw)
            p._pbest_cost = 100000 # an arbitrary large number
            p.update_pbest(self.sw, data, target)
            p.init_gbest(sw=self.sw)
            if i < swarm_size:
                self.sw._P.append(p)
            #print(i)
            i += 1

    def combpso(self):
        #config.max_nbr_iter = len(self.train_loader)*self.epochs
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.torch_device), target.to(self.torch_device)
                data = data.view(data.shape[0], -1) # This is for 2D data, if 1D then comment this line
                # update the swarm
                #for t in range(config.max_nbr_iter):
                t = epoch*len(self.train_loader)+batch_idx
                self.sw._w = config.w_min + (config.w_max - config.w_min) * 1 \
                        / (1 + (t / (config.max_nbr_iter*config.w_a))**config.w_b)
                self.sw._c1 = config.c_min + (config.c_max - config.c_min) * 1 \
                        / (1 + (t / (config.max_nbr_iter * config.w_a))**config.w_b)
                self.sw._c2 = config.c_max - (config.c_max - config.c_min) * 1 \
                        / (1 + (t / (config.max_nbr_iter * config.w_a))**config.w_b)
                self.sw._R = np.random.random_sample(size=(3, config.particle_size))

                for p in self.sw._P:
                    p.update_particle(self.sw)
                    p.update_pbest(self.sw, data, target)
                    p.update_gbest(self.sw)
                print(self.print_stat_swarm())
                # verify if the max number of iterations the swarm can stay idle is reached, if so call partial reinit
                # in all cases increment the gbest counter by 1
                if self.sw._gbest_idle_counter >= self.sw._gbest_idle_max_counter:
                    self.sw._reinit_partial_swarm()
                    print('{}% of particles reinitiated'.format(100*config.swarm_reinit_frac))
                    print(self.print_stat_swarm())
                self.sw._gbest_idle_counter += 1
                print('\n ### Epoch {}, cost {},sample {}/{} {:.0f}% ### \n'.format(epoch,self.sw._gbest_cp,batch_idx*len(data),\
                        len(train_loader)*len(data),100*batch_idx/len(train_loader)))

            print('\n ### Final gbest in epoch {}: {} ### \n'.format(epoch,self.sw._gbest_cp))
            #print(self.sw._gbest_cp)
            #if sw._local_search(): print(sw._final_str())

def test(model, device, test_loader, lossfn_name, number_class):
    model.eval()
    test_loss = 0
    correct = 0
    torch_device = torch.device(device) 
    if lossfn_name == 'MSE':
        t_onehot = torch.FloatTensor(test_loader.batch_size, number_class).to(torch_device) 
        loss = nn.MSELoss(reduction='sum') 
    elif lossfn_name == 'MAE':
        t_onehot = torch.FloatTensor(test_loader.batch_size, number_class).to(torch_device) 
        loss = nn.L1Loss(reduction='sum')
    elif lossfn_name == 'cross_entropy':
        loss = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(torch_device), target.to(torch_device)
            data = data.view(data.shape[0], -1)
            output = model(data)
            if lossfn_name != 'cross_entropy':
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

# %%
device = 'cpu'
net = MLP(N_input,20,10).to(torch.device(device))
epochs = 10
Nparticles = 100
trainpso = train_net_pso(net, Nparticles, train_loader, lossfn_name='MSE', 
                                number_class=10,epochs=epochs,dim_reduction=5)
trainpso.init_swarm()
trainpso.print_stat_swarm()
trainpso.combpso()
#### Training results
print('Accuracy in training set')
test(net,device,train_loader,'MSE',10)
#### Test results after training
print('Accuracy in testing set')
test(net,device,test_loader,'MSE',10)

print('Final statistics')
trainpso.print_stat_swarm()






# %%
