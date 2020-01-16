import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class pso():
    def __init__(self,model,Nparticles,train_loader,device="cuda",lossfn_name='cross_entropy',classes='None',debug=False, verbose=True):
        self.model = model
        self.Nparticles = Nparticles
        self.torch_device = torch.device(device)
        self.device = device
        self.lossfn_name = lossfn_name
        self.number_class = classes
        self.verbose = verbose
        self.debug = debug ## Debugging
        self.train_loader = train_loader
        if self.lossfn_name == 'MSE':
            self.t_onehot = torch.FloatTensor(self.train_loader.batch_size, self.number_class).to(self.torch_device) 
            self.loss = nn.MSELoss(reduction = 'sum') 
        elif self.lossfn_name == 'MAE':  
            self.t_onehot = torch.FloatTensor(self.train_loader.batch_size, self.number_class).to(self.torch_device) 
            self.loss = nn.L1Loss()
    def init_pso_part(self):
        self.x = list(); xi = list(); # X is a list of all particles. Each particle
        self.v = list(); vi = list(); # is a list with the tensor params of each layer
        #self.local_best = list(); local_best_i = list() # creating list to storage best position for each particle
        for i in range(self.Nparticles):
            for param in self.model.parameters():
                xi.append(torch.rand(param.shape).to(self.torch_device))
                vi.append(torch.rand(param.shape).to(self.torch_device))
                #local_best_i.append(torch.zeros(param.shape))
            self.x.append(xi)
            self.v.append(vi)
            #self.local_best.append(local_best_i)
            xi = list()
            vi =list()
            #local_best_i = list()
        # Initialize local best and global best
        self.local_best = self.x.copy()
        self.local_best_fitness = 1000*np.ones(self.Nparticles)
        # dataiter = iter(train_loader)
        # data, target = dataiter.next()
        # data, target = data.to(self.torch_device), target.to(self.torch_device)
        # data = data.view(data.shape[0], -1) # This is for 2D data, if 1D then comment this line
        # for i in range(self.Nparticles):
        #     self.load_particle_param(i) # load input parameters to model
        #     self.local_best_fitness[i] = self.fitness(data,target)
        # global_best_fitness_idx = np.argmin(self.local_best_fitness)
        # self.global_best = self.x[global_best_fitness_idx].copy()              
            

    def load_particle_param(self,particle_idx):
        for param, xlayer in zip(self.model.parameters(),self.x[particle_idx]):
            param.data = xlayer
    def fitness(self,data,target):
        with torch.no_grad(): # do not track operations for gradient calculation
            #data, target = data.to(self.torch_device), target.to(self.torch_device)
            if self.lossfn_name == "cross_entropy":
                output = self.model(data)
                fit = F.cross_entropy(output,target)
            elif (self.lossfn_name == "MSE") or (self.lossfn_name == "MAE"):
                self.t_onehot.zero_()
                self.t_onehot.scatter_(1, target.unsqueeze(1), 1)
                output = self.model(data)
                fit = self.loss(output,self.t_onehot)
            #loss = F.mse_loss(output,target)
        return fit
    def pso(self,w,c1,c2,epoch):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.torch_device), target.to(self.torch_device)
            data = data.view(data.shape[0], -1) # This is for 2D data, if 1D then comment this line
            for i in range(self.Nparticles):
                if self.debug == True: #### Debuging
                    x_tmp = self.x[i].copy()
                    v_tmp = self.v[i].copy()
                    fitness_value_tmp = self.fitness(data,target)
                self.load_particle_param(i) # load input parameters to model
                fitness_value = self.fitness(data,target)
                if fitness_value < self.local_best_fitness[i]:
                    self.local_best_fitness[i] = fitness_value # local best value for each particle/agent
                    self.local_best[i] = self.x[i].copy() # local best position for each particle/agent
                global_best_fitness_idx = np.argmin(self.local_best_fitness)
                self.global_best = self.x[global_best_fitness_idx].copy()
            for i in range(self.Nparticles):
                for j, (xi, vi) in enumerate(zip(self.x[i],self.v[i])): # executing pso for every layer of particle i
                    if self.device == "cpu":
                        r1 = torch.rand(xi.shape)
                        r2 = torch.rand(xi.shape)
                    else:
                        r1 = torch.cuda.FloatTensor(xi.shape).uniform_()
                        r2 = torch.cuda.FloatTensor(xi.shape).uniform_()
                    vi = w*vi + c1*r1*(self.local_best[i][j] - xi) + c2*r2*(self.global_best[j] - xi)
                    F.hardtanh_(vi,min_val=-1., max_val=1.)
                    xi = vi + xi
                    F.hardtanh_(xi,min_val=-1., max_val=1.)
                    self.x[i][j] = xi
                    self.v[i][j] = vi
                    if self.debug == True:
                        if not torch.equal(xi, self.x[i][j]):
                            print("Particle {} component {} do not update".format(i,j))
                        if not torch.equal(vi, self.v[i][j]):
                            print("Velocity of Particle {} component {} do not update".format(i,j))
                        if torch.equal(x_tmp[j], self.x[i][j]):
                            print("Particle {} component {} do not change".format(i,j))
                        if torch.equal(v_tmp[j], self.v[i][j]):
                            print("Velocity Particle {} component {} do not change".format(i,j))
                # if self.debug == True:
                #     for ii in range(len(self.x[i])):
                #         if (torch.equal(x_tmp[ii], self.x[i][ii])) & (batch_idx > 0) :
                #             print("Particle {} do not change".format(i))
                #         if (torch.equal(v_tmp[ii], self.v[i][ii])) & (batch_idx > 0) :
                #             print("Velocity of Particle {} do not change".format(i))
            #global_best_fitness_idx = np.argmax(self.local_best_fitness)
            #global_best_fitness_idx = np.argmin(self.local_best_fitness)
            #self.global_best = self.x[global_best_fitness_idx].copy()
            if (batch_idx % 10 == 0) & self.verbose == True: # debugging
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader)*self.train_loader.batch_size,
                    100. * batch_idx / len(self.train_loader), self.local_best_fitness[global_best_fitness_idx]))
        return self.local_best_fitness[global_best_fitness_idx]

    





