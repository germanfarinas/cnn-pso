#%%
from __future__ import print_function
import sys
#sys.path.append("/home/wuchty/combpso/")
import numpy as np
#from swarm import Swarm, Particle
from algorithms.swarm_v2 import Swarm, Particle
#import config
from algorithms import config_v2 as config

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
    def forward(self,x):
        #x.view(x.size(0), -1)
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = torch.sigmoid(x)
        return x
class simple_cnn(nn.Module):
    def __init__(self):
        super(simple_cnn,self).__init__()
        #self.conv1 = nn.Conv2d(1,16,3)
        self.conv1_1 = nn.Conv2d(1,4,3)
        self.conv1_2 = nn.Conv2d(1,4,3)
        self.conv1_3 = nn.Conv2d(1,4,3)
        self.conv1_4 = nn.Conv2d(1,4,3)
        #self.conv2 = nn.Conv2d(16,16,3)
        self.conv2_1 = nn.Conv2d(16,4,3)
        self.conv2_2 = nn.Conv2d(16,4,3)
        self.conv2_3 = nn.Conv2d(16,4,3)
        self.conv2_4 = nn.Conv2d(16,4,3)
        self.pool = nn.MaxPool2d(2)
        #self.d1 = nn.Linear(5*5*16,10)
        self.d_1 = nn.Linear(5*5*16,5)
        self.d_2 = nn.Linear(5*5*16,5)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        #x = self.conv1(x)
        ### conv 1 layer
        x_1 = self.conv1_1(x)
        x_2 = self.conv1_2(x)
        x_3 = self.conv1_3(x)
        x_4 = self.conv1_4(x)
        x = torch.cat((x_1, x_2, x_3, x_4), 1)
        ###
        x = F.relu(x)
        x = self.pool(x)
        #x = self.conv2(x)
        ### conv 2 layer
        x_1 = self.conv2_1(x)
        x_2 = self.conv2_2(x)
        x_3 = self.conv2_3(x)
        x_4 = self.conv2_4(x)
        x = torch.cat((x_1, x_2, x_3, x_4), 1)
        ###
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 5*5*16)
        ### FC layer
        #x = self.d1(x)
        x_1 = self.d_1(x)
        x_2 = self.d_2(x)
        x = torch.cat((x_1, x_2), 1)
        x = self.softmax(x)
        return x
class backp_cnn(nn.Module):
    def __init__(self):
        super(backp_cnn,self).__init__()
        ch1 = 4
        self.ch2 = 4
        self.conv1 = nn.Conv2d(1,ch1,3)
        self.conv2 = nn.Conv2d(ch1,self.ch2,3)
        self.pool = nn.MaxPool2d(2)
        self.d1 = nn.Linear(5*5*self.ch2,10)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 5*5*self.ch2)
        x = self.d1(x)
        x = self.softmax(x)
        return x
class global_avg_pool_cnn(nn.Module):
    def __init__(self):
        super(global_avg_pool_cnn,self).__init__()
        self.conv1 = nn.Conv2d(1,16,3)
        #self.conv2 = nn.Conv2d(16,16,3,groups=2)
        self.conv2_1 = nn.Conv2d(16,4,3)
        self.conv2_2 = nn.Conv2d(16,4,3)
        self.conv2_3 = nn.Conv2d(16,4,3)
        self.conv2_4 = nn.Conv2d(16,4,3)
        self.pool = nn.MaxPool2d(2)
        self.gavgpool = nn.AvgPool2d(11)
        self.d1 = nn.Linear(16,10)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        #x = self.conv2(x)
        ### conv 2 layer
        x_1 = self.conv2_1(x)
        x_2 = self.conv2_2(x)
        x_3 = self.conv2_3(x)
        x_4 = self.conv2_4(x)
        x = torch.cat((x_1, x_2, x_3, x_4), 1)
        ###
        x = F.relu(x)
        #x = self.pool(x)
        x = self.gavgpool(x)
        x = x.view(-1, 1*1*16)
        x = self.d1(x)
        x = self.softmax(x)
        return x
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

batch_size = int(60000/20)
#batch_size = 256
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
    def convert_vector2param(self,vec,param_idx): # load vector to net (torch model) parameters.
        if param_idx is not None: # load only specific layer param
            params = list(self.model.parameters())
            vec = vec.reshape(params[param_idx].shape)
            params[param_idx].data = torch.from_numpy(vec).float()
        else: # whole parameter as a vector
            vect = torch.from_numpy(vec).float()[0] if self.emb_dim is not None else torch.from_numpy(vec).float()
            #vect = torch.from_numpy(vec).float() 
            vector_to_parameters(vect, self.model.parameters())
    def decoder(self, vec_Ldim):
        x = np.matrix(vec_Ldim)
        x = x*self.emb_matrix
        return x  ## return a torch tensor/vector ready for vector_to_parameter method
    def classification_perfomance(self,px,data,target,param_idx = None):
        with torch.no_grad(): # do not track operations for gradient calculation
            if self.emb_dim is not None: # check if dim_reduction is used, if so decode low dimension vector to high dim params 
                self.convert_vector2param(self.decoder(px), param_idx)
            else:
                self.convert_vector2param(px, param_idx) 
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
        return cost

    def print_stat_swarm(self,sw):
        print('mean_x, std_x, mean_v, std_v, pbest_cost, nbdim_max, nbdim_min, w, c1, c2 ')
        for p in sw._P:
            nbdim_max = np.where(p.x == config.x_max)
            nbdim_min = np.where(p.x == config.x_min)
            nbdim_max = 0 if nbdim_max[0].size == 0 else nbdim_max[0].size
            nbdim_min = 0 if nbdim_min[0].size == 0 else nbdim_min[0].size
            print('{:1.2f}, {:1.2f}, {:1.2f}, {:1.2f}, {:1.2f}, {}, {}, {:1.2f}, {:1.2f}, {:1.2f}'
            .format(np.mean(p.x),np.std(p.x), np.mean(p.v), np.std(p.v) , 
            p._pbest_cost,nbdim_max,nbdim_min,sw._w, sw._c1, sw._c2))
        #df = pd.DataFrame([np.mean(p.x),np.std(p.x), np.mean(p.v), np.std(p.v) , p._pbest_cost] for p in sw._P)
        #print(df)
        print('particle size = {}'.format(len(sw._P[0].x)))
    
    # def init_swarm(self, particles=None):
    #     if particles is not None:
    #         self.Nparticles = particles
    #     # config parameters
    #     config.particle_size = self.emb_dim if self.emb_dim is not None else self.number_weights
    #     config.feature_init = int(config.particle_size/2) # number of weights that goes to xmax
    #     config.swarm_size = self.Nparticles
    #     config.fitness_function = self.classification_perfomance
    #     config.max_nbr_iter = len(self.train_loader)*self.epochs
    #     config.swarm_reinit_frac = 0.5
    #     # initialize the swarm
    #     self.sw = Swarm()
    #     self.sw._gbest_cost = 100000 # an arbitrary large number
    #     swarm_size = self.Nparticles
    #     i = 0
    #     dataiter = iter(self.train_loader)
    #     data, target = dataiter.next()
    #     data, target = data.to(self.torch_device), target.to(self.torch_device)
    #     data = data.view(data.shape[0], -1) # This is for 2D data, if 1D then comment this line
    #     while i < swarm_size:
    #         p = Particle(sw=self.sw)
    #         p._pbest_cost = 100000 # an arbitrary large number
    #         p.update_pbest(self.sw, data, target)
    #         p.init_gbest(sw=self.sw)
    #         if i < swarm_size:
    #             self.sw._P.append(p)
    #         #print(i)
    #         i += 1

    # def combpso(self):
    #     #config.max_nbr_iter = len(self.train_loader)*self.epochs
    #     for epoch in range(self.epochs):
    #         for batch_idx, (data, target) in enumerate(self.train_loader):
    #             data, target = data.to(self.torch_device), target.to(self.torch_device)
    #             data = data.view(data.shape[0], -1) # This is for 2D data, if 1D then comment this line
    #             # update the swarm
    #             #for t in range(config.max_nbr_iter):
    #             t = epoch*len(self.train_loader)+batch_idx
    #             self.sw._w = config.w_min + (config.w_max - config.w_min) * 1 \
    #                     / (1 + (t / (config.max_nbr_iter*config.w_a))**config.w_b)
    #             self.sw._c1 = config.c_min + (config.c_max - config.c_min) * 1 \
    #                     / (1 + (t / (config.max_nbr_iter * config.w_a))**config.w_b)
    #             self.sw._c2 = config.c_max - (config.c_max - config.c_min) * 1 \
    #                     / (1 + (t / (config.max_nbr_iter * config.w_a))**config.w_b)
    #             self.sw._R = np.random.random_sample(size=(3, config.particle_size))

    #             for p in self.sw._P:
    #                 p.update_particle(self.sw)
    #                 p.update_pbest(self.sw, data, target)
    #                 p.update_gbest(self.sw)
    #             print(self.print_stat_swarm())
    #             # verify if the max number of iterations the swarm can stay idle is reached, if so call partial reinit
    #             # in all cases increment the gbest counter by 1
    #             if self.sw._gbest_idle_counter >= self.sw._gbest_idle_max_counter:
    #                 self.sw._reinit_partial_swarm()
    #                 print('{}% of particles reinitiated'.format(100*config.swarm_reinit_frac))
    #                 print(self.print_stat_swarm())
    #             self.sw._gbest_idle_counter += 1
    #             print('\n ### Epoch {}, cost {},sample {}/{} {:.0f}% ### \n'.format(epoch,self.sw._gbest_cp,batch_idx*len(data),\
    #                     len(train_loader)*len(data),100*batch_idx/len(train_loader)))

    #         print('\n ### Final gbest in epoch {}: {} ### \n'.format(epoch,self.sw._gbest_cp))
    #         #print(self.sw._gbest_cp)
    #         #if sw._local_search(): print(sw._final_str())

    def init_layered_pso(self, Nparticles):
        print('Initializing swarm layers')
        self.Nparticles = Nparticles
        config.fitness_function = self.classification_perfomance
        
        dataiter = iter(self.train_loader)
        data, target = dataiter.next()
        data, target = data.to(self.torch_device), target.to(self.torch_device)
        cost = self.classification_perfomance(self.convert_param2vector(),data,target)
        #data = data.view(data.shape[0], -1) # This is for 2D data, if 1D then comment this line
        ### Initializing all swarms for every layer params
        self.sw_params_list = list()
        for idx, param in enumerate(self.model.parameters()):
            #config.feature_init = int(param.data.numel()/2) # number of weights that goes to xmax
            sw = Swarm(self.Nparticles, param.data.numel())
            sw.feature_init = int(param.data.numel()/2) # number of weights that goes to xmax
            sw._gbest_x = param.detach().view(param.numel()) # assign initial parameters for swarm global best (tensor)
            #sw._gbest_x = param.view(param.numel()) # assign initial parameters for swarm global best (tensor)
            sw._gbest_x = sw._gbest_x.detach().numpy() # convert to numpy
            #sw._gbest_v = self.v
            sw._gbest_cost = cost
            sw._w = 0.9
            sw._c1 = 0.5182
            sw._c2 = 0.5182
            i = 0
            swarm_size = self.Nparticles
            while i < swarm_size:
                p = Particle(sw=sw)
                p._pbest_cost = 100000 # an arbitrary large number
                #p.update_pbest(sw, data, target, param_idx = idx)
                #p.init_gbest(sw=sw)
                if i < swarm_size:
                    sw._P.append(p)
                    #print(i)
                    i += 1
            self.sw_params_list.append(sw) # append swarm for every layer
            self.print_stat_swarm(sw)
    
    def layered_pso(self,nber_pass_iter):
        #model_parameters = self.model.parameters()
        #model_parameters.reverse() # reverse order
        for j in range(nber_pass_iter):
            number_params_layers = len(list(self.model.parameters()))
            config.max_nbr_iter = len(self.train_loader)*self.epochs
            #for i, param in enumerate(self.model.parameters()):
            for k, param in enumerate(reversed(list(net.parameters()))): ## this two lines reverse
                i = number_params_layers - k -1 # the order in which PSO execute (last layer first)
                reinit_counter = 0
                ####### update gbest_cost at the start of a new layer because the other layers changed #####
                dataiter = iter(self.train_loader)
                data, target = dataiter.next()
                data, target = data.to(self.torch_device), target.to(self.torch_device)
                cost = self.classification_perfomance(self.convert_param2vector(),data,target)
                self.sw_params_list[i]._gbest_cost = cost
                #######################################################################  
                if i==3:
                    something =1 ## for breakpoint
                for epoch in range(self.epochs):
                    for batch_idx, (data, target) in enumerate(self.train_loader):
                        data, target = data.to(self.torch_device), target.to(self.torch_device)
                        #data = data.view(data.shape[0], -1) # This is for 2D data, if 1D then comment this line
                        # update the swarm
                        #for t in range(config.max_nbr_iter):
                        t = epoch*len(self.train_loader)+batch_idx
                        self.sw_params_list[i]._w = config.w_min + (config.w_max - config.w_min) * 1 \
                                / (1 + (t / (config.max_nbr_iter*config.w_a))**config.w_b)
                        self.sw_params_list[i]._c1 = config.c_min + (config.c_max - config.c_min) * 1 \
                            / (1 + (t / (config.max_nbr_iter * config.w_a))**config.w_b)
                        self.sw_params_list[i]._c2 = config.c_max - (config.c_max - config.c_min) * 1 \
                            / (1 + (t / (config.max_nbr_iter * config.w_a))**config.w_b)
                        #self.sw_params_list[i]._R = np.random.random_sample(size=(3, param.data.numel()))
                        # self.sw_params_list[i]._w = 0.9
                        # self.sw_params_list[i]._c1 = 0.5182
                        # self.sw_params_list[i]._c2 = 0.5182

                        for p in self.sw_params_list[i]._P:
                            #p.update_particle(sw_params_list[i])
                            p.update_pbest(self.sw_params_list[i], data, target, param_idx = i)
                            p.update_gbest(self.sw_params_list[i])
                            p.update_particle(self.sw_params_list[i])
                        self.print_stat_swarm(self.sw_params_list[i])
                        # verify if the max number of iterations the swarm can stay idle is reached, if so call partial reinit
                        # in all cases increment the gbest counter by 1
                        if reinit_counter == 2: break
                        if self.sw_params_list[i]._gbest_idle_counter >= self.sw_params_list[i]._gbest_idle_max_counter:
                            reinit_counter +=1
                            #self.sw_params_list[i]._reinit_partial_swarm()
                            print('{}% of particles reinitiated'.format(100*config.swarm_reinit_frac))
                        self.sw_params_list[i]._gbest_idle_counter += 1
                        print('\n### Iter {}, param {}, Epoch {}, cost {},sample {}/{} {:.0f}% ### \n'.format(j,i,epoch,\
                        self.sw_params_list[i]._gbest_cost,batch_idx*len(data),len(train_loader)*len(data),100*batch_idx/len(train_loader)))
                    if reinit_counter == 2: break
                # load layer global best to model params
                self.convert_vector2param(self.sw_params_list[i]._gbest_x, i) # if dim reduction is used this have to be corrected as in perfomance_classification function
                # update global best cost for next layer
                # if i < number_params_layers-1 :
                #     self.sw_params_list[i+1]._gbest_cost = self.sw_params_list[i]._gbest_cost 
                # else:
                #     self.sw_params_list[0]._gbest_cost = self.sw_params_list[i]._gbest_cost



def train_fullbackprog(model, device, train_loader, optimizer, epoch):
    model.train()
    torch_device = torch.device(device)
    loss = nn.CrossEntropyLoss(reduction='sum')
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(torch_device), target.to(torch_device)
        #data = data.view(data.shape[0], -1)
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
    elif lossfn_name == 'MAE':
        t_onehot = torch.FloatTensor(test_loader.batch_size, number_class).to(torch_device) 
        loss = nn.L1Loss(reduction='sum')
    elif lossfn_name == 'cross_entropy':
        loss = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(torch_device), target.to(torch_device)
            #data = data.view(data.shape[0], -1)
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

#%%
# layered PSO
device = 'cpu'
net = simple_cnn().to(torch.device(device))
#net = global_avg_pool_cnn().to(torch.device(device))
epochs = 1
Nparticles = 10
trainpso = train_net_pso(net, Nparticles, train_loader, lossfn_name='cross_entropy', 
                                number_class=10,epochs=epochs,dim_reduction=None)
trainpso.init_layered_pso(Nparticles)
trainpso.layered_pso(10)
#### Training results
print('Accuracy in training set')
test(net,device,train_loader,'cross_entropy',10)
#### Test results after training
print('Accuracy in testing set')
test(net,device,test_loader,'cross_entropy',10)

# %%
# Backprog
epochs = 20
device = "cuda"
lr = 0.01
momentum = 0.5
#net = simple_cnn().to(torch.device(device))
#net = global_avg_pool_cnn().to(torch.device(device))
net = backp_cnn().to(torch.device(device))
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
for epoch in range(epochs):
    train_fullbackprog(net, device, train_loader, optimizer, epoch)
#### Training results
print('Accuracy in training set')
test(net,device,train_loader,'cross_entropy',10)
#### Test results after training
print('Accuracy in testing set')
test(net,device,test_loader,'cross_entropy',10)








# %%
