"""
    Defining configuration parameters
"""
import numpy as np
"""
    Main program parameters
"""
epochs = 5
"""
    Swarm parameters
"""
v_min = -5.0              #min boundary of particle's velocity
v_max = 5.0               #max boundary of particle's velocity
x_min = -100.0              #min boundary of particle's position
x_max = 100.0               #max boundary of particle's position
w_min = 0.4               #min boundary for dynamic inertia weight w
w_max = 1.2               #max boundary for dynamic inertia weight w
c_min = 1.7               #min boundary for c coefficient
c_max = 2.1               #max boundary for c coefficient
w_a = .5                  #a parameter in the dynamic inertia weight
w_b = 4                   #b parameter in the dynamic inertia weight

max_nbr_iter = None         #Maximum Number of iterations
gbest_idle_counter = 0      #counts the number of iterations the gbest is idle
gbest_idle_max_counter = 3  #max number of iteration gbest is idle
swarm_reinit_frac = 0.5     #percentage of the swarm size to be renitilized when gbest is idle for too long
feature_init = None

fitness_function = None     #Fitness function used for optimization
