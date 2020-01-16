# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 19:34:00 2018

@author: Hassen Dhrif
"""

import numpy as np

#from swarm import Swarm, Particle
from algorithms.swarm import Swarm, Particle
#import config
from algorithms import config


def combpso():

    sw = Swarm()
    # initialize the swarm
    sw._R = np.random.random_sample(size=(1, config.particle_size))
    p = Particle(sw)
    p.b = np.ones(config.particle_size)
    p._nbf = config.particle_size
    sw._gbest_b = p.b
    sw._gbest_nbf = p._nbf
    sw._gbest_cost, sw._gbest_cp = config.fitness_function(p, sw)

    swarm_size = config.swarm_size
    i, n = 0, 0
    while i < swarm_size:
        p = Particle(sw=sw)
        p.update_pbest(sw=sw)
        p.init_gbest(sw=sw)
        if i < config.swarm_size:
            sw._P.append(p)
        if i >= config.swarm_size + n*50 - 1 and sw._gbest_x == []:
            swarm_size += 50
            n += 1
        i += 1
    # update the swarm
    for t in range(config.max_nbr_iter):
        sw._w = config.w_min + (config.w_max - config.w_min) * 1 \
                / (1 + (t / (config.max_nbr_iter*config.w_a))**config.w_b)
        sw._c1 = config.c_min + (config.c_max - config.c_min) * 1 \
                 / (1 + (t / (config.max_nbr_iter * config.w_a))**config.w_b)
        sw._c2 = config.c_max - (config.c_max - config.c_min) * 1 \
                 / (1 + (t / (config.max_nbr_iter * config.w_a))**config.w_b)
        sw._R = np.random.random_sample(size=(3, config.particle_size))

        for p in sw._P:
            p.update_particle(sw=sw)
            p.update_pbest(sw=sw)
            p.update_gbest(sw=sw)

        # verify if the max number of iterations the swarm can stay idle is reached, if so call partial reinit
        # in all cases increment the gbest counter by 1
        if sw._gbest_idle_counter >= sw._gbest_idle_max_counter:
            sw._reinit_partial_swarm()
        sw._gbest_idle_counter += 1

    print('\n ### Final gbest ### \n')
    print(sw)
    if sw._local_search(): print(sw._final_str())

    return sw