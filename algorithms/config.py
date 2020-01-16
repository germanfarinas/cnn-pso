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
v_min = -1.0            #min boundary of particle's velocity
v_max = 1.0            #max boundary of particle's velocity
x_min = -1.0            #min boundary of particle's position
x_max = 1.0             #max boundary of particle's position
w_min = 0.4               #min boundary for dynamic inertia weight w
w_max = 1.2               #max boundary for dynamic inertia weight w
c_min = 1.7               #min boundary for c coefficient
c_max = 2.1               #max boundary for c coefficient
w_a = .5                  #a parameter in the dynamic inertia weight
w_b = 4                   #b parameter in the dynamic inertia weight
swarm_size = None         #Swarm size in terms of number of particles
particle_size = None      #Particle size
max_nbr_iter = None       #Maximum Number of iterations
gbest_idle_counter = 0      #counts the number of iterations the gbest is idle
gbest_idle_max_counter = 5  #max number of iteration gbest is idle
swarm_reinit_frac = 0.1     #percentage of the swarm size to be renitilized when gbest is idle for too long
feature_init = None

"""
    Feature subset selection parameters
"""
X = []                      #Dataset of gene expressions
y = []                      #class labels
classes = []                #List of classes
genes = []                  #List of genes or probes
folder = None               #Data path
fitness_function = None     #Fitness function used for optimization
clf = None                  #Classifier used in the wrapper method
alpha_1 = 0.8                  #alpha used for the weighted sum formula of the the fitness function
alpha_2 = 0.1
alpha_3 = 0.1
cv = 5                      #cross-validation stratification ratio
filtered_genes = []
"""
    RDC local search parameters
"""
rdc_f = np.sin      #function to use for random projection
rdc_k = 20          #number of random projections to use
rdc_s = 1/6.        #scale parameter
rdc_n = 1           #number of times to compute the RDC and return the median (for stability)
irdc = []           #Non linear corrleation between each feature and the class label
crdc = []          #Non linear correlation between pairs of features
ls_threshold = 15    #Local search threshold
"""
    Subset stability (consistency) parameters
"""
freq = []

