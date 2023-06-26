import msmfcode
import logging
import numpy as np

from os.path import join, dirname, split, exists
from os import mkdir


PY_PKG_PATH = split(dirname(msmfcode.__file__))[0]


class Metric:
    POS_ERROR = 'pos_error'
    POS_ERROR_MEAN = 'pos_error_mean'
    POS_ERROR_STD = 'pos_error_std'
    POS_ERROR_MIN = 'pos_error_min'
    POS_ERROR_MAX = 'pos_error_max'
    POS_ERROR_NUM_CAT = 'pos_error_num_cat'
    NUM_FIELD_CON_PRUNED = 'num_field_con_pruned'
    NUM_FIELD_CON_TOTAL = 'num_field_con_total'
    MEAN_FIELD_ACTIVITY = 'mean_field_activity'
    PERC_CORRECT_FIELDS = 'perc_correct_fields'
    AVG_NUM_FIELDS_PER_NEURON = 'avg_num_fields_per_neuron'
    AVG_NUM_FIELDS_ACTIVE_PER_NEURON = 'avg_num_fields_active_per_neuron'
    AVG_ACCUM_FIELD_COVERAGE_PER_NEURON = 'avg_accum_field_coverage_per_neuron'
    AVG_ACCUM_ACTIVE_FIELD_COVERAGE_PER_NEURON = 'avg_accum_active_field_coverage_per_neuron'
    NUM_FIELDS_TOTAL = 'num_fields_total'
    NUM_FIELDS_ACTIVE = 'num_fields_total'
    ACTIVITY_FALSE_POSITIVES_NUM = 'activity_false_positives_num'
    ACTIVITY_FALSE_NEGATIVES_NUM = 'activity_false_negatives_num'
    PERC_UNIQUE_FIELD_COMBINATIONS = 'perc_unique_field_combinations'


class Statistics:
    MEAN = 'mean'
    STD = 'std'
    MIN = 'min'
    MAX = 'max'
    MEDIAN = 'median'
    MEDIAN_ADD = ['q1', 'q3', 'whishi', 'whislo']
    NUM_CAT_ERR = 'num_cat_err'


class RandomSeed:
    TIME = 'time'
    INDEX = 'id'


class FileType:
    DATA = 'data'
    FIGURE = 'figure'
    MODEL = 'model'
    OPTIMIZATION = 'optimization'


class ExperimentType:
    EVALUATION = 'evaluation'
    EVALUATION_SINGLE = 'evaluation_single'
    OPTIMIZATION = 'optimization'


class NetworkType:
    MSMF_DYNAMIC = 'DMSMF'
    MSMF_FIXED = 'FMSMF'
    GRID = 'Grid'
    SSSF = 'SSSF'
    GENERAL = 'general'


class Distribution:
    GAMMA = 'gamma'
    NORMAL = 'normal'


class Noise:
    UNIFORM = 'uniform'
    GAUSSIAN = 'gaussian'


class PlotType:
    NEURON_ACTIVITY_STATIC = 'neuron_activity_static'


class PlotConfig:
    FILE_TYPE = 'pdf'


class Optimization:
    class ParentSelectionSchemes:
        RANK_WEIGHTING = 'rank-weighting'
        FITNESS_WEIGHTING = 'fitness-weighting'
        EGREEDY = 'e-greedy'


class LogHandler:
    FILE = 0
    STREAM = 1


# Logging
class Log:
    FILE = join(PY_PKG_PATH, 'data/msmf.log')
    # FORMAT_FILE = "[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName)20s() ] [%(levelname)-8s] %(message)s"
    FORMAT_FILE = "[%(asctime)s] [%(filename)-20s:%(lineno)-4s] [%(levelname)-8s] %(message)s"
    FORMAT_SCREEN = "%(log_color)s%(message)s"
    LEVEL_FILE = logging.DEBUG
    LEVEL_SCREEN = logging.INFO
    DATEFMT = '%d.%m.%Y %H:%M:%S'


STATISTICS_NAMES = [getattr(Statistics, attr) for attr in dir(Statistics)
                    if not callable(getattr(Statistics, attr)) and not attr.startswith("__")
                    and not type(getattr(Statistics, attr)) == list]
EXPERIMENT_FILE_NAME = {ExperimentType.EVALUATION: 'experiment',
                        ExperimentType.EVALUATION_SINGLE: 'experiment',
                        ExperimentType.OPTIMIZATION: 'optimization'}
EXPERIMENT_SETUP_FILE_NAME = {ExperimentType.EVALUATION: 'experiments.csv',
                              ExperimentType.EVALUATION_SINGLE: 'experiments_single.csv',
                              ExperimentType.OPTIMIZATION: 'optimizations.csv'}
EXPERIMENT_FOLDERS = {NetworkType.MSMF_DYNAMIC: join(PY_PKG_PATH, 'data/evaluation/msmf_dynamic'),
                      NetworkType.MSMF_FIXED: join(PY_PKG_PATH, 'data/evaluation/msmf_fixed'),
                      NetworkType.SSSF: join(PY_PKG_PATH, 'data/evaluation/sssf'),
                      NetworkType.GRID: join(PY_PKG_PATH, 'data/evaluation/grid'),
                      NetworkType.GENERAL: join(PY_PKG_PATH, 'data/evaluation/general')}
EXPERIMENT_SUBFOLDERS = {FileType.DATA: 'data', FileType.FIGURE: 'figures', FileType.MODEL: 'models',
                         FileType.OPTIMIZATION: 'optimizations'}
PATH_CONFIG = join(PY_PKG_PATH, 'config')
FILE_NAME_CONFIG_EVAL = {NetworkType.MSMF_DYNAMIC: 'config_eval_DMSMF.yaml',
                         NetworkType.MSMF_FIXED: 'config_eval_FMSMF.yaml',
                         NetworkType.SSSF: 'config_eval_SSSF.yaml',
                         NetworkType.GRID: 'config_eval_Grid.yaml'}
FILE_NAME_CONFIG_CGA = {NetworkType.MSMF_DYNAMIC: 'config_cga_DMSMF.yaml',
                        NetworkType.MSMF_FIXED: 'config_cga_FMSMF.yaml',
                        NetworkType.SSSF: 'config_cga_SSSF.yaml',
                        NetworkType.GRID: 'config_cga_Grid.yaml'}
FILE_NAME_DEFAULT_PARAMS_CAN = {NetworkType.MSMF_DYNAMIC: 'config_default_params_DMSMF.yaml',
                                NetworkType.MSMF_FIXED: 'config_default_params_FMSMF.yaml',
                                NetworkType.SSSF: 'config_default_params_SSSF.yaml',
                                NetworkType.GRID: 'config_default_params_Grid.yaml'}
EXPERIMENTAL_DATA_DECIMALS = 3

# Generate non-existing folders
for model_name in [attr for attr in dir(NetworkType) if not callable(getattr(NetworkType, attr)) and not attr.startswith("__")]:
    model = getattr(NetworkType, model_name)
    for subfolder in EXPERIMENT_SUBFOLDERS.values():
        folder_path = join(EXPERIMENT_FOLDERS[model], subfolder)
        if not exists(folder_path):
            mkdir(folder_path)


## Configuration variables
# Experiment Configuration
DESCRIPTION = 'description'
LABEL = 'label'
NUM_EVAL_ITERS = 'num_eval_iters'
# Network Configuration
NUM_NEURONS = 'num_neurons'
ENV_LENGTH = 'env_length'
INIT_VALUES = 'init_values'
PROB_DEAD_NEURONS = 'prob_dead_neurons'
# Parameter Configuration
PARAM_NAME = 'param_name'
PARAM_EVAL_RANGE_MIN = 'param_eval_range_min'
PARAM_EVAL_RANGE_MAX = 'param_eval_range_max'
PARAM_EVAL_RANGE_STEP = 'param_eval_range_step'
NETWORK_PARAMS = 'network_params'
# Evaluation Configuration
METRICS = 'metrics'
GLOBAL_METRICS = 'global_metrics'
PLOTS = 'plots'
# Evo Opt Configuration
CONFIGURATION = 'configuration'
PARAMETERS = 'parameters'
SELECTION_RATE = 'selection_rate'
POPULATION_SIZE = 'population_size'
MUTATION_PROB = 'mutation_prob'
MUTATE_BEST = 'mutate_best'
REEVALUATION = 'reevaluation'
CROSSOVER_ENABLED = 'crossover_enabled'
PARENT_SELECTION_SCHEME = 'parent_selection_scheme'
PARENT_SELECTION_PARAMETER = 'parent_selection_parameter'
FITNESS_FUN = 'fitness_fun'
IS_ENABLED = 'is_enabled'
MIN = 'min'
MAX = 'max'
STEP = 'step'
ROUND_DECIMAL = 'round_decimal'
GENERATION = 'generation'
NUM_GENERATIONS = 'num_generations'
DECODING_THRESHOLD = 'decoding_threshold'

