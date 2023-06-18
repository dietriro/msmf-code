import numpy as np

from msmfcode.core.logging import log
from msmfcode.core.config import *
from msmfcode.execution.optimization import ContGeneticAlgorithm
from msmfcode.models.cann import MSMFGridCAN, MSMFMultiCAN, MSMFSingleCAN
from msmfcode.execution.parallel import ParallelExecutor
from msmfcode.evaluation.plot import plot_fitness
from msmfcode.evaluation.data import load_optimization_state


# Definitions
OPTIMIZE = 0
CONT_OPTIMIZE = 1
PLOT = 2

# Config
log.handlers[LogHandler.STREAM].setLevel(logging.ESSENS)
log.handlers[LogHandler.FILE].setLevel(logging.DEBUG)
task = CONT_OPTIMIZE
net_type = MSMFMultiCAN


def optimize():
    cga = ContGeneticAlgorithm(ParallelExecutor, net_type)
    cga.run(num_gen=3000, save_interval=100, save_state_indiv=False)


def continue_optimization():
    cga = load_optimization_state(ContGeneticAlgorithm, net_type, experiment_num=1)

    # for key in cga.evaluated_entities.keys():
    #     if key[0] == 0.02 and key[1] == -0.06:
    #         print('yeah')
    #         print(cga.evaluated_entities[key].fitness)

    print('asd')

    # cga.run(num_gen=3000, save_interval=2, initialize=False)


def plot_fitness_cga():
    plot_fitness(MSMFSingleCAN, 1, data_fun=np.max)
    plot_fitness(MSMFSingleCAN, 1, data_fun=np.mean)
    # plot_fitness(MSMFSingleCAN, 9, data_fun=np.mean)
    # plot_fitness(MSMFSingleCAN, 10, data_fun=np.mean)



if __name__ == '__main__':
    if task == OPTIMIZE:
        optimize()
    elif task == CONT_OPTIMIZE:
        continue_optimization()
    elif task == PLOT:
        plot_fitness_cga()
