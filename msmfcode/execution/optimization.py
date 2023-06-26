import csv
import numpy as np
from time import time
from datetime import timedelta
from collections import OrderedDict
from typing import Union
from copy import deepcopy, copy

from msmfcode.core.config import *
from msmfcode.core.logging import log
from msmfcode.core.io import load_yaml
from msmfcode.execution.parallel import ParallelExecutor
from msmfcode.evaluation.data import save_optimization_setup, save_optimization_state
from msmfcode.models.cann import Grid, FMSMF, DMSMF, DiffSolverError, EmptyFields


class Entity:
    def __init__(self, params, fitness_fun, index=None, executor_class: ParallelExecutor = None,
                 can_class: Union[Grid, FMSMF, DMSMF] = None):
        self.executor = None
        self.params = copy(params)
        self.id = index
        self.fitness_fun = fitness_fun

        self.fitness = 0

        if executor_class is not None and can_class is not None:
            self.initialize_executor(executor_class, can_class)

    def initialize_executor(self, executor_class: ParallelExecutor,
                            can_class: Union[Grid, FMSMF, DMSMF]):
        self.executor = executor_class('ga', can_class, parameters=self.params)

    def evaluate(self):
        log.essens(f'Evaluating entity [{self.id}] with params: {dict(self.params)}')
        self.executor.init_eval_data()
        try:
            self.executor.run()
            return True
        except (DiffSolverError, EmptyFields):
            return False
        except Exception as e:
            log.error(e)
            raise e

    def calc_fitness(self):
        # Get metrics for fitness calculation
        # Get the last positional error (mean, std)
        if Metric.POS_ERROR_MEAN in self.executor.config[METRICS]:
            pos_error_mean = self.executor.eval_data.get_last_result(Metric.POS_ERROR_MEAN)
        if Metric.POS_ERROR_STD in self.executor.config[METRICS]:
            pos_error_std = self.executor.eval_data.get_last_result(Metric.POS_ERROR_STD)
        # Get field related information
        if Metric.MEAN_FIELD_ACTIVITY in self.executor.config[METRICS]:
            mean_field_activity = self.executor.eval_data.get_last_result(Metric.MEAN_FIELD_ACTIVITY)
        if Metric.PERC_CORRECT_FIELDS in self.executor.config[METRICS]:
            perc_correct_fields = self.executor.eval_data.get_last_result(Metric.PERC_CORRECT_FIELDS)
        # Maximum positional error based on length of environment and number of neurons (best single field model)
        pos_error_max = self.executor.cans[0].p.env_length / self.executor.cans[0].p.num_neurons

        # Calculate fitness based on a decaying exponential function
        try:
            self.fitness = eval(self.fitness_fun)
        except NameError as err:
            log.error('Cannot calculate fitness because a parameter used in the equation is not defined.')
            raise err

        log.info(f'Fitness of entity [{self.id}]: {self.fitness}')

        return self.fitness

    def __lt__(self, obj):
        return self.fitness < obj.fitness

    def __gt__(self, obj):
        return self.fitness > obj.fitness

    def __le__(self, obj):
        return self.fitness <= obj.fitness

    def __ge__(self, obj):
        return self.fitness >= obj.fitness

    def __eq__(self, obj):
        return self.fitness == obj.fitness


class EvaluatedEntity(Entity):
    def __init__(self, entity, last_metrics=None):
        super().__init__(entity.params, entity.fitness_fun, index=entity.id)
        self.fitness = entity.fitness
        self.last_metrics = last_metrics


class ContGeneticAlgorithm:
    def __init__(self, executor_class: type(ParallelExecutor),
                 can_class: type(Union[Grid, FMSMF, DMSMF])):
        self.finished_gen = None
        self.best_fitness = None
        self.data_save_entity = None
        self.fitness_max = None
        self.fitness_mean = None
        self.evaluated_entities = None
        self.experiment_num = None
        self.executor_class = executor_class
        self.can_class = can_class

        self.config: dict = {}
        self.net_params: dict = {}
        self.num_params: int = 0
        self.population: list = None

        self.wrote_data_headers = False

        self.load_params()

    def load_params(self):
        config_all = load_yaml(PATH_CONFIG, FILE_NAME_CONFIG_CGA[self.can_class.__name__])
        self.config = config_all[CONFIGURATION]
        # Add only enabled params for training
        for param_name, param_values in config_all[PARAMETERS].items():
            if param_values[IS_ENABLED]:
                self.net_params[param_name] = param_values
        self.num_params = len(self.net_params)

    def get_random_net_params(self):
        params = OrderedDict()
        for param_name, param in self.net_params.items():
            params[param_name] = self.get_random_net_param(param_name)
        return params

    def get_random_net_param(self, param_name):
        param = self.net_params[param_name]
        if param[IS_ENABLED]:
            num_values = (param[MAX] - param[MIN]) / param[STEP] + 1
            param_value = param[MIN] + np.random.randint(num_values) * param[STEP]
            if ROUND_DECIMAL in param.keys():
                param_value = np.round(param_value, param[ROUND_DECIMAL])
            return param_value
        else:
            log.warning(f'Could not create random value for parameter \"{param_name}\" as it is disabled for training.')
            return None

    def initialize_run(self, num_gen, experiment_num=None):
        self.population = []
        self.evaluated_entities = {}
        self.fitness_mean = np.zeros(num_gen)
        self.fitness_max = np.zeros(num_gen)
        self.experiment_num = experiment_num
        self.finished_gen = -1

        # Generate random values for each entity
        for i_pop in range(self.config[POPULATION_SIZE]):
            params = self.get_random_net_params()
            ent = Entity(params, self.config[FITNESS_FUN], index=i_pop, executor_class=self.executor_class,
                         can_class=self.can_class)
            self.population.append(ent)

        self.data_save_entity = deepcopy(self.population[0])

        self.best_fitness = 0.0

    def run(self, num_gen, save_interval=5, experiment_num=None, initialize=True, save_state_indiv=False):
        if initialize:
            self.initialize_run(num_gen, experiment_num=experiment_num)

            global_metric_names = []
            for metric_name, global_metrics in self.population[0].executor.config[GLOBAL_METRICS].items():
                for global_metric in global_metrics:
                    global_metric_names.append(f'{metric_name}_{global_metric}')

            optimization_data = [['generation', 'entity_id'] + list(self.population[0].params.keys()) +
                                 global_metric_names + ['fitness']]
        else:
            optimization_data = list()

        for i_gen in range(self.finished_gen+1, num_gen):
            fitness_all = np.zeros(len(self.population))
            last_parents = np.zeros(len(self.population))
            time_run_start = time()

            for i_ent in range(len(self.population)):
                # Simulate each parent CAN if it hasn't been simulated yet
                if not self.config[REEVALUATION] and \
                        tuple(self.population[i_ent].params.values()) in self.evaluated_entities.keys():
                    log.essens(f'Reusing entity [{self.population[i_ent].id}] '
                               f'with params: {dict(self.population[i_ent].params)}')
                    # Add fitness from previous evaluation
                    ent_id = self.population[i_ent].id
                    self.population[i_ent] = deepcopy(
                        self.evaluated_entities[tuple(self.population[i_ent].params.values())])
                    self.population[i_ent].id = ent_id
                    fitness_all[self.population[i_ent].id] = self.population[i_ent].fitness

                    last_metrics = self.population[i_ent].last_metrics
                else:
                    # Try to evaluate/run simulation for entity i
                    while not self.population[i_ent].evaluate():
                        # If the evaluation fails, generate a new set of random parameters which has not been
                        # evaluated yet
                        params = self.get_random_net_params()
                        while tuple(self.population[i_ent].params.values()) in self.evaluated_entities.keys():
                            params = self.get_random_net_params()

                        self.population[i_ent] = Entity(params, self.config[FITNESS_FUN], index=i_ent,
                                                        executor_class=self.executor_class, can_class=self.can_class)

                    # Calculate fitness for each parent and sum it up
                    fitness_all[self.population[i_ent].id] = self.population[i_ent].calc_fitness()

                    last_metrics = list()
                    for metric_name, global_metrics in self.population[i_ent].executor.config[GLOBAL_METRICS].items():
                        for global_metric in global_metrics:
                            last_metrics.append(self.population[i_ent].executor.eval_data.get_last_result(metric_name,
                                                                                                          global_metric))

                    self.evaluated_entities[tuple(self.population[i_ent].params.values())] = EvaluatedEntity(
                        self.population[i_ent],
                        last_metrics=last_metrics)

                log.essens(f'Fitness of entity [{self.population[i_ent].id}] = {self.population[i_ent].fitness}')

                param_values = list()
                for param_name, param_value in self.population[i_ent].params.items():
                    if ROUND_DECIMAL in self.net_params[param_name].keys():
                        param_values.append(np.round(param_value, self.net_params[param_name][ROUND_DECIMAL]))
                    else:
                        param_values.append(param_value)

                optimization_data.append([i_gen, self.population[i_ent].id] +
                                         param_values +
                                         list(np.round(last_metrics, EXPERIMENTAL_DATA_DECIMALS)) +
                                         [self.population[i_ent].fitness])

            # Get statistics of fitness
            self.fitness_mean[i_gen] = fitness_all.mean()
            self.fitness_max[i_gen] = fitness_all.max()

            # Get the best fitness of generation and check if it's better than the current maximum
            best_fitness_i_gen = np.max(self.fitness_max[i_gen])
            if best_fitness_i_gen > self.best_fitness:
                self.best_fitness = best_fitness_i_gen

            # Select which entities to keep based on their fitness
            self.natural_selection(fitness_all)
            # Sort population descending
            self.population.sort(reverse=True)
            # Update ids of entities and fitness
            fitness_all = np.zeros(len(self.population))
            for entity_i, entity in enumerate(self.population):
                entity.id = entity_i
                fitness_all[entity_i] = entity.fitness

            # Get normalized fitness of all entities
            fitness_all_norm = fitness_all / np.sum(fitness_all)
            # Set maximum fitness id
            max_fitness_id = fitness_all.argmax()

            # Generate new children by probabilistically selecting a pair of parents for mating
            children = []
            num_children = self.config[POPULATION_SIZE] - len(self.population)
            for i_sel in range(int(np.ceil(num_children / 2))):
                # Check if enough children have already been generated
                if len(self.population) + len(children) == self.config[POPULATION_SIZE]:
                    break
                # Find parent ids based on their fitness
                if self.config[PARENT_SELECTION_SCHEME] == Optimization.ParentSelectionSchemes.FITNESS_WEIGHTING:
                    # Don't modify the fitness.
                    # Choose the weights for roulette-wheel selection based on the actual fitness
                    pass
                elif self.config[PARENT_SELECTION_SCHEME] == Optimization.ParentSelectionSchemes.RANK_WEIGHTING:
                    # ToDo: Implement rank weighting
                    pass
                elif self.config[PARENT_SELECTION_SCHEME] == Optimization.ParentSelectionSchemes.EGREEDY:
                    fitness_all_norm[max_fitness_id] = self.config[PARENT_SELECTION_PARAMETER]
                    fitness_all_norm_mask = np.ma.array(fitness_all_norm, mask=False)
                    fitness_all_norm_mask.mask[max_fitness_id] = True
                    fitness_all_norm_mask *= (1 - self.config[PARENT_SELECTION_PARAMETER]) / fitness_all_norm_mask.sum()
                else:
                    log.error(f'Parent selection scheme \"{self.config[PARENT_SELECTION_SCHEME]}\" is not implemented '
                              f'yet. Please select a valid scheme.')
                    return
                # Perform roulette-wheel selection
                parent_ids = np.random.choice(len(fitness_all_norm), 2, p=fitness_all_norm)

                # Select a random crossover point for the parameters/genes
                # Set lower bound to 1 so that crossover never takes all parameters from one entity
                crossover_id = np.random.randint(1, self.num_params) if self.num_params > 1 else 0
                # Generate two new children using this crossover point
                for i_child in range(2):
                    if self.config[CROSSOVER_ENABLED]:
                        # Generate new params for child based on parent params
                        child_params = self.crossover(self.population[parent_ids[i_child]],
                                                      self.population[parent_ids[1 - i_child]],
                                                      crossover_id=crossover_id)
                    else:
                        child_params = self.population[parent_ids[i_child]].params
                    # Create new child and add it to the list
                    child = Entity(child_params, self.config[FITNESS_FUN],
                                   index=i_sel * 2 + i_child + self.population[-1].id + 1)
                    children.append(child)

            self.population += children

            # Randomly mutate parameters in case mutation_prob is set to a value > 0
            if self.config[MUTATION_PROB] > 0.0:
                for entity_i, entity in enumerate(self.population):
                    # Don't mutate the best entity if parameter is set
                    if not self.config[MUTATE_BEST] and entity_i == 0:
                        continue
                    for param_name in entity.params.keys():
                        if np.random.rand() <= self.config[MUTATION_PROB]:
                            entity.params[param_name] = self.get_random_net_param(param_name)
                            log.debug(
                                f'Entity [{entity_i}]: Mutated parameter [{param_name}] to be {entity.params[param_name]}')

                    entity.initialize_executor(self.executor_class, self.can_class)

            # Print mean fitness for last population
            run_time = int(time() - time_run_start)
            est_time_left = (num_gen - (i_gen + 1)) * run_time

            log.essens('-------------------------------------------------------------------------------------')
            log.essens(f'Finished training of generation {i_gen + 1}')

            log.essens(f'Mean fitness of all entities = {self.fitness_mean[i_gen]}')
            log.essens(f'Best fitness = {self.fitness_max[i_gen]}, by entity {max_fitness_id} '
                       f'with parameters {dict(self.population[max_fitness_id].params)}')
            log.debug(f'Fitness of all parents {fitness_all}')
            log.debug(f'Picked parents {last_parents}')

            log.essens(f'>> Run-time = {timedelta(seconds=run_time)} hours')
            log.essens(f'>> Total time left = {timedelta(seconds=est_time_left)} hours')
            log.essens('-------------------------------------------------------------------------------------')

            # Save data if save interval has been reached
            if (i_gen + 1) % save_interval == 0:
                # Save updated optimization config after each generation
                self.experiment_num = save_optimization_setup(self.data_save_entity.executor.cans[0], self.net_params,
                                                              self.config, experiment_num=self.experiment_num,
                                                              num_generations=i_gen + 1,
                                                              num_eval_iters=
                                                              self.data_save_entity.executor.config[NUM_EVAL_ITERS],
                                                              fitness_max=np.round(self.best_fitness, 5))

                # Save evaluation data to csv file after each generation
                self.finished_gen = i_gen
                self.save_optimization_data(optimization_data)
                if save_state_indiv:
                    save_optimization_state(self, experiment_num=self.experiment_num)
                else:
                    save_optimization_state(self)
                optimization_data = list()

        log.essens('-------------------------------------------------------------------------------------')
        log.essens(f'Mean fitness of all generations: {self.fitness_mean}')
        log.essens(f'Max fitness of all generations: {self.fitness_max}')
        log.essens('-------------------------------------------------------------------------------------')

    def natural_selection(self, fitness_population):
        sorted_population_ids = fitness_population.argsort()
        n_keep = int(np.round(self.config[POPULATION_SIZE] * self.config[SELECTION_RATE]))

        self.population = [self.population[i] for i in sorted_population_ids[n_keep:]]

        return sorted_population_ids[n_keep:][::-1]

    def crossover(self, parent_a, parent_b, crossover_id=None):
        if crossover_id is None:
            crossover_id = np.random.randint(1, self.num_params)

        # Create ordered dict for params of new child
        child_params = OrderedDict()

        # Store crossover information for loop
        param_ranges = [[0, crossover_id], [crossover_id, self.num_params]]
        parents = [parent_a, parent_b]

        # Loop over both parents and add the respective params of each parent to the child params depending on the
        # crossover id
        for i_par in range(len(param_ranges)):
            param_id = 0
            for param_name, param_value in parents[i_par].params.items():
                if param_ranges[i_par][0] <= param_id < param_ranges[i_par][1]:
                    child_params[param_name] = param_value
                param_id += 1

        return child_params

    def save_optimization_data(self, optimization_data):
        opt_file_name = f'{EXPERIMENT_FILE_NAME[ExperimentType.OPTIMIZATION]}_{self.experiment_num:02d}.csv'
        data_path = join(PY_PKG_PATH, EXPERIMENT_FOLDERS[self.can_class.__name__],
                         EXPERIMENT_SUBFOLDERS[FileType.DATA], opt_file_name)

        if self.wrote_data_headers:
            write_mode = 'a'
        else:
            write_mode = 'w'
            self.wrote_data_headers = True

        with open(data_path, write_mode) as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the data rows
            for row in optimization_data:
                csvwriter.writerow(row)
