# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 09:34:00 2018

@author: Hassen Dhrif
"""

from copy import copy
import numpy as np
from operator import attrgetter
from math import ceil
from heapq import heappop, heappush

#from utilities import sigmoid, RDC
from algorithms.utilities import sigmoid, RDC
from algorithms import config_v2 as config


class Swarm():
    def __init__(self, swarm_size, particle_size):

        self._P = []                # list of particle objects
        self._R = []                # Random variable
        self._w = None              # inertia weight (swarm level)
        self._c1 = None
        self._c2 = None

        self._gbest_idle_counter = config.gbest_idle_counter  # counts the number of iterations the gbest is idle
        self._gbest_idle_max_counter = config.gbest_idle_max_counter  # max number of iteration gbest is idle
        self._swarm_reinit_frac = config.swarm_reinit_frac  # percentage of the swarm size to be renitilized when gbest is idle for too long
        self.swarm_size = swarm_size
        self.particle_size = particle_size
        self.feature_init = 0 # 
        
        # swarm gbest attributes
        self._gbest_x = []     # swarm best position
        self._gbest_v = []
        self._gbest_cost = 100000 # set this to a large arbitrary number (temporary solution)

        # analysis
        self._f_calls = 0            # number of function calls
        self._nbr_reinit = 0         # total number of partial swarm initializations

        # heapq for best values
        self.h = []

        # Initialization parameters
        self._init_cost = 1.0
        self._init_cp = 0.0


    # re-initialize part of the swarm particles when gbest is idle for too long
    def _reinit_partial_swarm(self):
        # create a binary vector where the number of ones is defined by swarm_reinit_frac
        k = ceil(self.swarm_size * self._swarm_reinit_frac)
        a = np.array([1]*k + [0]*(self.swarm_size-k))
        np.random.shuffle(a)
        # randomize the velocities and positions of the randomly selected particles
        for p in np.array(self._P)[a==1]:
            p.v = np.random.uniform(config.v_min, config.v_max, size=self.particle_size)
            #p.x = np.random.uniform(config.x_min, config.x_max, size=config.particle_size)
        # reset the idle counter and increment the reinit counter
        self._gbest_idle_counter = 0
        self._nbr_reinit += 1

    def __len__(self):
        return config.swarm_size

    def __str__(self):
        st = str(f'After {self._f_calls} function calls and {self._nbr_reinit} partial reinits, \n')
        selected_genes = {config.genes[i]:[config.irdc[i],config.freq[i]]
                          for i, val in enumerate(self._gbest_b) if val ==1}
        st = st + f"Swarms best subset has {self._gbest_nbf} features with {self._gbest_cp:.2f} score \n"
        if self._gbest_nbf < 20:
            st = st + f"The features selected are: \n {selected_genes} \n"
        return st

    def _final_str(self):
        """
        Returns the final output string used after final local search
        :returns:
        string  formatted output string
        """
        selected_genes = {config.genes[i]:[config.irdc[i],config.freq[i]]
                          for i, val in enumerate(self._gbest_b) if val ==1}
        st = f"Swarms best subset has {self._gbest_nbf} features with {self._gbest_cp:.2f} score \n"
        if self._gbest_nbf < 20:
            st = st + f"The features selected are: \n {selected_genes} \n"
        return st

class Particle():
    def __init__(self, sw):

        # particle's velocity, position and binary position
        self.v = np.random.uniform(config.v_min, config.v_max, size=sw.particle_size)
        #a = np.array([0] * (sw.particle_size - sw.feature_init) + [1] * sw.feature_init)
        #np.random.shuffle(a)
        self.x = np.random.uniform(config.x_min, config.x_max, size=sw.particle_size)
        #self.x = np.zeros(sw.particle_size)
        #self.x[a == 1] = config.x_max
        #self.x[a == 0] = config.x_min
        #self.b = np.array(sw._R[0] < sigmoid(self.x)).astype(int)
        #self._nbf = self.b.sum()
        self._cost = None

        # particle's best cost and position
        self._pbest_x = self.x
        self._pbest_cost = 100000 # set this to a large arbitrary number (temporary solution)

        self.vmax_dim = np.ones(sw.particle_size)*config.v_max
        self.vmin_dim = np.ones(sw.particle_size)*config.v_min
        self.gamma = 0.1

    # update the the particle's velocity, position and binary position
    def update_particle(self, sw):
        _R = np.random.random_sample(size=(3, sw.particle_size))
        # self.v = sw._w * self.v + sw._c1 * np.multiply(sw._R[1], (self._pbest_x - self.x)) \
        #          + sw._c2 * (np.multiply(sw._R[2], (sw._gbest_x - self.x)))
        self.v = sw._w * self.v + sw._c1 * np.multiply(_R[1], (self._pbest_x - self.x)) \
                                + sw._c2 * (np.multiply(_R[2], (sw._gbest_x - self.x)))
        self.v = np.clip(self.v, self.vmin_dim, self.vmax_dim)
        self.x = np.clip(self.x + self.v, config.x_min, config.x_max)

        #### Reduce vmin and vmax per dimmension in case particle leave search space
        #self.vmax_dim = self.vmax_dim - self.gamma*(self.x ==  1)
        #self.vmin_dim = self.vmin_dim + self.gamma*(self.x == -1)

    # def update_pbest(self, sw):
    #     #if subset is not empty calculate cost and update pbest
    #     if self._nbf>0:
    #         self._cost, self._cp = config.fitness_function(self, sw)
    #         sw._f_calls += 1
    #         if self._cost < self._pbest_cost:
    #             #update pbest
    #             self._pbest_x = self.x
    #             self._pbest_v = self.v
    #             self._pbest_cost = self._cost
    #             self._pbest_cp = self._cp

    # update the particle's personal best
    def update_pbest(self, sw, data, target, param_idx = None):
        #if subset is not empty calculate cost and update pbest
        self._cost = config.fitness_function(self.x, data, target, param_idx) \
            if param_idx is not None else config.fitness_function(self, data, target) 
        sw._f_calls += 1
        if self._cost < self._pbest_cost:
            #push the previous pbest particle into the heap
            if self._pbest_cost != 1.:
                p = Particle(sw)
                p.x = copy(self._pbest_x)
                p._cost = self._pbest_cost
                heappush(sw.h, p)
            #update pbest
            self._pbest_x = self.x
            self._pbest_v = self.v
            self._pbest_cost = self._cost
        #if heap is not empty keep poping until a particle with better cost is found
        #if found update pbest with the new found particle
        elif len(sw.h)>0:
            p = heappop(sw.h)
            while len(sw.h) > 0 and p._cost > self._cost:
                p = heappop(sw.h)
            if p._cost < self._cost:
                self._pbest_x = copy(p.x)
                self._pbest_cost = p._cost


    # initialize the swarm global best
    def init_gbest(self, sw):
        if self._cost < sw._gbest_cost:
            # update the new global best
            sw._gbest_x = self.x
            sw._gbest_v = self.v
            sw._gbest_cost = self._cost
            #sw._gbest_b = self.b
            #sw._gbest_nbf = self._nbf

    # def update_gbest(self, sw):
    #     # use init gbest code
    #     if self._cost < sw._gbest_cost:
    #         # update the new global best
    #         sw._gbest_x = self.x
    #         sw._gbest_v = self.v
    #         sw._gbest_cost = self._cost
    #         sw._gbest_cp = self._cp
    #         sw._gbest_b = self.b
    #         sw._gbest_nbf = self._nbf
    #         print(self._nbf)
    #
    #         # reset gbest idle counter
    #         sw._gbest_idle_counter = 0
    #
    #         # Update frequencies of features
    #         for i in range(config.particle_size):
    #             if self.b[i]:
    #                 config.freq[i] += 1

    # update the swarm global best
    def update_gbest(self, sw):
        if self._cost < sw._gbest_cost:
            # push the last global best as a new particle in the heap
            if sw._gbest_cost < 1.:
                p = Particle(sw)
                p.x = copy(sw._gbest_x)
                p._cost = sw._gbest_cost
                p._cp = sw._gbest_cp
                heappush(sw.h, p)

            # update the new global best
            sw._gbest_v = self.v
            sw._gbest_x = self.x
            sw._gbest_cost = self._cost
            #sw._gbest_b = self.b
            #sw._gbest_nbf = self._nbf
            #print(self._nbf)

            # reset gbest idle counter
            sw._gbest_idle_counter = 0

            # Update frequencies of features
            #for i in range(config.particle_size):
            #    if self.b[i]: config.freq[i] += 1

    # comparison operator on cost
    def __lt__(self, other):
        return self._cost < other._cost

    def __le__(self, other):
        return self._cost <= other._cost

    # define the length of the particle
    def __len__(self):
        return len(self.x)

    # print the particle data
    def __str__(self):
        return str('position: {} cost: {}  \n'
                   .format(self.x, self._cp))

