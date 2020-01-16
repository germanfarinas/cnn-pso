
import pickle, sys
from datetime import datetime
from os.path import join
import numpy as np
import pickle
# from joblib import Parallel, delayed
# import multiprocessing

#from combpso import combpso
from algorithms.combpso import combpso
#from swarm import Swarm
from algorithms.swarm import Swarm
#import config, model
from algorithms import config, model


def run_combpso():
    """
    runs the combpso algorithm with the specified arguments and returns a pickle file of the solutions
    INPUT
        - arg[0]: file name: main.py
        - arg[1]: dataset name: ad_blood1, ad_blood2, ...

    OUTPUT
        - pickle file (e.g. ad_blood1.pickle)
    """

    #set environment variables for feature selection examples
    print('Initialize model data ***** \n')
    start = datetime.now()

    SW = []
    for i in range(config.epochs):
        print(f'Processing Run {i} ******************** \n')
        sw = combpso()
        SW.append(sw)

    # Union of all gbests + Local search
    print('\n Processing local search ****** \n')
    gbest_u = np.zeros(config.particle_size).astype(np.int)
    for sw in SW:
        gbest_u = np.bitwise_or(gbest_u, sw._gbest_b)
    sw_u = Swarm()
    sw_u._gbest_b = gbest_u
    sw_u._gbest_nbf = gbest_u.sum()
    sw_u._local_search()
    SW.append(sw_u)
    print(sw_u._final_str())

    print('Time elapsed: {} '.format(datetime.now() - start))

    print('****** Dump results to pickle file ***** \n')
    g = config.genes[sw_u._gbest_b==1]
    with open(join(config.folder, str(sys.argv[1]) + '.pickle'), 'wb') as p_wb:
        pickle.dump(g, p_wb)


if __name__ == "__main__":
    run_combpso()