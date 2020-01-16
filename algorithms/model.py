"""
    Model data
    - arg[0]: file name: main.py
    - arg[1]: dataset name: ad_blood1, synthetic1, ...
"""
import sys
from os.path import join
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

#import config
from algorithms import config
#from utilities import subset_accuracy
from algorithms.utilities import subset_accuracy
#import load, filter
from datasets import load, filter

#setup main program parameters
config.epochs = 3

#setup swarm parameters
config.swarm_size = 5        #Swarm size in terms of number of particles
config.max_nbr_iter = 5      #Number of iterations
config.feature_init = 50

#Load the dataset X, the class label y, the genes or probes, the classes, and the data folder
config.X, config.y, config.genes, config.classes, config.folder = load._loads[str(sys.argv[1])]()
config.filtered_genes = filter.ad_filter()
config.X = config.X[:,config.filtered_genes]
config.genes = config.genes[config.filtered_genes]
config.particle_size = len(config.filtered_genes)
config.fitness_function = subset_accuracy
config.clf = ExtraTreesClassifier(n_estimators=20, random_state=0)

#initialize the irdc values and set crdc to 0s
config.rdc_k = 20
config.rdc_n = 20
with open(join(config.folder, str(sys.argv[1]) + '_irdc.txt'), 'r') as fr:
    config.irdc = fr.read().splitlines()
    config.irdc = np.array(config.irdc).astype(float)
    config.irdc = config.irdc[config.filtered_genes]
# config.irdc = np.array([RDC(config.X[:, i], config.y, k=config.rdc_k, n=config.rdc_n)
#                         for i in range(config.particle_size)])
config.crdc = np.zeros((config.particle_size, config.particle_size))  # combined C-association Matrix

#initialize consistency parameters
config.freq = np.zeros(config.particle_size)

