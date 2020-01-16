#%%
from __future__ import print_function
import sys
#sys.path.append("/home/wuchty/combpso/")
import numpy as np
from algorithms.swarm_v3 import Swarm, Particle
from algorithms import config_v3 as config
import matplotlib.pyplot as plt
## Profiling
import cProfile, pstats, io
import pandas as pd

# %%
class pso():
    def fitness(self,px,data,target):
        #fit = np.dot(px,px)
        m=10**6
        e=(np.arange(1,len(px)+1)-1)/(len(px)-1)
        fit = np.dot(np.multiply(m**(e/2),px),np.multiply(m**(e/2),px))
        return fit

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

    def init_swarm(self, swarm_size=10, particle_size=10):
        # config parameters
        config.feature_init = int(particle_size/2) # number of weights that goes to xmax
        config.fitness_function = self.fitness
        config.swarm_reinit_frac = 0.5
        # initialize the swarm
        self.sw = Swarm(swarm_size,particle_size)
        self.sw._gbest_cost = 1000000000000 # an arbitrary large number
        i = 0
        data=target=1 #dummy parameter
        while i < swarm_size:
            p = Particle(sw=self.sw)
            p._pbest_cost = 1000000000000 # an arbitrary large number
            #p.update_particle(sw)
            p.update_pbest(self.sw, data, target)
            p.init_gbest(sw=self.sw)
            if i < swarm_size:
                self.sw._P.append(p)
            #print(i)
            i += 1
        #self.sw.update_ebest()
    def run(self,nber_pass_iter):
        #model_parameters = self.model.parameters()
        #model_parameters.reverse() # reverse order
        data=target=1
        config.max_nbr_iter = nber_pass_iter
        for j in range(nber_pass_iter):
            # update the swarm
            #for t in range(config.max_nbr_iter):
            t = j
            self.sw._w = config.w_min + (config.w_max - config.w_min) * 1 \
                    / (1 + (t / (config.max_nbr_iter*config.w_a))**config.w_b)
            self.sw._c1 = config.c_min + (config.c_max - config.c_min) * 1 \
                / (1 + (t / (config.max_nbr_iter * config.w_a))**config.w_b)
            self.sw._c2 = config.c_max - (config.c_max - config.c_min) * 1 \
                / (1 + (t / (config.max_nbr_iter * config.w_a))**config.w_b)
            #self.sw_params_list[i]._R = np.random.random_sample(size=(3, param.data.numel()))
            # self.sw_params_list[i]._w = 0.9
            # self.sw_params_list[i]._c1 = 0.5182
            # self.sw_params_list[i]._c2 = 0.5182
            ##################
            # _size_subset = int(self.sw.particle_size/20)
            # a = np.array([0] * (self.sw.particle_size - _size_subset) + [1] * _size_subset)
            # np.random.shuffle(a)
            ##################
            for p in self.sw._P:
                p.update_particle(self.sw)
                #p.update_random_grouping(self.sw,a)
                p.update_pbest(self.sw, data, target)
                p.update_gbest(self.sw)
            #self.sw.update_ebest()
            #self.print_stat_swarm(self.sw)
            print('\n### Iter {}, cost {} ### \n'.format(j,self.sw._gbest_cost))
                





# %%
pso = pso()
pso.init_swarm(swarm_size=20, particle_size=100)
pso.print_stat_swarm(pso.sw)
pso.run(1000)

# %%
