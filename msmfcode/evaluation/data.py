import numpy as np
import csv
import datetime
import pickle
from os.path import exists
from matplotlib import cbook
from copy import copy

from msmfcode.core.config import *
from msmfcode.core.logging import log
from msmfcode.core.io import load_yaml
from msmfcode.models.cann import ContinuousAttractorNetwork


class EvaluationMetric:
    name: str = None
    values: list = None
    mean: list = None
    std: list = None
    min: list = None
    max: list = None
    median: list = None
    q1: list = None
    q3: list = None
    whishi: list = None
    whislo: list = None


    def __init__(self, name):
        self.name = name
        self.values = []
        self.mean = []
        self.std = []
        self.min = []
        self.max = []
        self.median = []
        self.q1 = []
        self.q3 = []
        self.whishi = []
        self.whislo = []


    def __str__(self):
        return self.name

    def __getitem__(self, item):
        return getattr(self, item)

    def update_global_metric(self, global_metric):
        if global_metric == Statistics.MEDIAN:
            # Get median/quantile statistics using boxplot library
            stats = cbook.boxplot_stats(self.values)
            # Set value of 'median' to value of 'med' - they use this short, unclear name (med)
            stats[0][Statistics.MEDIAN] = stats[0]['med']
            for stat in Statistics.MEDIAN_ADD + [Statistics.MEDIAN]:
                getattr(self, stat).append(stats[0][stat])
        elif global_metric == Statistics.NUM_CAT_ERR:

            pass
        elif not hasattr(np, global_metric):
            log.warning(f'Cannot calculate statistic {global_metric} as it is not provided by numpy.')
            return
        else:
            metric_fun = getattr(np, global_metric)
            if len(self.values) <=0:
                log.warning(f'Cannot calculate statistic {global_metric} for {self.name} as the values are empty.')
                return
            getattr(self, global_metric).append(metric_fun(self.values))


class EvaluationData:
    def __init__(self, net_type, eval_metrics):
        self.net_type = net_type

        self.pos_error_abs = []
        self.num_runs = 0

        self.eval_metrics = {eval_metric: EvaluationMetric(eval_metric) for eval_metric in eval_metrics}

    def update_step(self, net, estimated_positions, limit: float = 0):
        self.update_pos_error(net, estimated_positions, limit)

        for eval_metric in self.eval_metrics.values():
            if hasattr(net, eval_metric.name) and getattr(net, eval_metric.name) is not None:
                eval_metric.values.append(getattr(net, eval_metric.name))

        self.num_runs += 1

    def update_run(self, reset=True):
        for eval_metric in self.eval_metrics.values():
            for global_metric in STATISTICS_NAMES:
                eval_metric.update_global_metric(global_metric)

        log.trace(f'Mean Error    = {self.eval_metrics[Metric.POS_ERROR_MEAN].mean[-1]}')
        log.trace(f'Std Deviation = {self.eval_metrics[Metric.POS_ERROR_STD].mean[-1]}')

        if reset:
            self.reset()

    def reset(self):
        self.pos_error_abs = []
        self.num_runs = 0

        for eval_metric in self.eval_metrics.values():
            eval_metric.values = []

    def update_pos_error(self, net, estimated_positions, limit: float = 0):
        # calculate abs, mean and std error
        pos_error_abs = np.abs(net.pos - estimated_positions)*net.p.disc_step
        if limit > 0:
            pos_error_abs = pos_error_abs[limit:-limit]

        self.pos_error_abs.append(np.abs(net.pos - estimated_positions)*net.p.disc_step)
        if Metric.POS_ERROR_NUM_CAT in self.eval_metrics.keys():
            # Calculate number of catastrophic errors encountered over whole experiment,
            # -> number of time the error exceeded the threshold cat_error_threshold
            self.eval_metrics[Metric.POS_ERROR_NUM_CAT].values.append(np.sum(pos_error_abs > net.p.cat_error_threshold))
        if Metric.POS_ERROR_MEAN in self.eval_metrics.keys():
            self.eval_metrics[Metric.POS_ERROR_MEAN].values.append(np.mean(pos_error_abs))
        if Metric.POS_ERROR_STD in self.eval_metrics.keys():
            self.eval_metrics[Metric.POS_ERROR_STD].values.append(np.std(pos_error_abs))
        if Metric.POS_ERROR_MIN in self.eval_metrics.keys():
            self.eval_metrics[Metric.POS_ERROR_MIN].values.append(np.min(pos_error_abs))
        if Metric.POS_ERROR_MAX in self.eval_metrics.keys():
            self.eval_metrics[Metric.POS_ERROR_MAX].values.append(np.max(pos_error_abs))

    def get_ratio(self, metric_a, metric_b):
        return list(np.array(self.eval_metrics[metric_a].mean) / np.array(self.eval_metrics[metric_b].mean))

    def get_last_result(self, metric_name, global_metric='mean'):
        metric_values = getattr(self.eval_metrics[metric_name], global_metric)
        if len(metric_values) > 0:
            return metric_values[-1]
        else:
            log.error(f'Eval metric list \'{global_metric}\' for metric \'{metric_name}\' is empty. Cannot return a value.')
            return None

    def save(self, net: ContinuousAttractorNetwork, experiment_id, experiment_num, eval_params, experiment_type):
        eval_file_name = f'{EXPERIMENT_FILE_NAME[ExperimentType.EVALUATION]}_{experiment_id}-{experiment_num:02d}.csv'
        eval_file_path = join(EXPERIMENT_FOLDERS[str(net)], EXPERIMENT_SUBFOLDERS[FileType.DATA],
                              eval_file_name)

        headers = []
        values = []

        # Add config defined metrics
        if experiment_type == ExperimentType.EVALUATION_SINGLE:
            for metric in self.eval_metrics.values():
                headers.append(metric.name)
                values.append(metric.values)

        # Add custom metrics/parameters
        if experiment_type != ExperimentType.EVALUATION_SINGLE:
            for param_name in eval_params:
                headers.append(param_name)
                values.append(eval_params[param_name])

        # writing to csv file
        with open(eval_file_path, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the headers if file is newly created
            csvwriter.writerow(headers)

            # writing the data rows
            for i in range(len(values[0])):
                row = []
                for v in values:
                    if len(v) > i:
                        row.append(round(v[i], EXPERIMENTAL_DATA_DECIMALS))
                csvwriter.writerow(row)


def is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def save_experimental_setup(net: ContinuousAttractorNetwork, experiment_id, experiment_type, experiment_num=None,
                            **kwargs):
    params = [a for a in dir(net.p) if not callable(getattr(net.p, a)) and not a.startswith("__")]

    file_path = join(EXPERIMENT_FOLDERS[str(net)], EXPERIMENT_SETUP_FILE_NAME[experiment_type])

    # add the experiment id and network type
    create_eval_file = not exists(file_path)
    # experiment_num = get_last_experiment_num(net, experiment_id, experiment_type) + 1
    last_experiment_num = get_last_experiment_num(net, experiment_id, experiment_type)
    do_update = False
    if experiment_num is None:
        experiment_num = last_experiment_num + 1
    elif experiment_num == last_experiment_num:
        do_update = True
    elif experiment_num < last_experiment_num:
        log.warning(f'Defined experiment num \'{experiment_num}\' is smaller than \'{last_experiment_num}\'. Aborting '
                    f'save process in order to prevent data loss.')
        return None

    data = {'experiment_id': experiment_id, 'experiment_num': f'{experiment_num:02d}',
            'network_type': str(net), 'time_finished': datetime.datetime.now().strftime('%d.%m.%y - %H:%M')}

    # add all parameters used for creating the cann
    for param_name in params:
        value = getattr(net.p, param_name)
        if value is not None:
            data[param_name] = value

    save_setup(net, data, experiment_type, experiment_num, create_eval_file, do_update, **kwargs)

    return experiment_num


def save_optimization_setup(net: ContinuousAttractorNetwork, optimized_params, optimization_config,
                            experiment_num=None, **kwargs):
    params = [a for a in dir(net.p) if not callable(getattr(net.p, a)) and not a.startswith("__")]

    file_path = join(EXPERIMENT_FOLDERS[str(net)], EXPERIMENT_SETUP_FILE_NAME[ExperimentType.OPTIMIZATION])

    # add the experiment id and network type
    create_eval_file = not exists(file_path)
    last_experiment_num = get_last_optimization_num(type(net))
    do_update = False
    if experiment_num is None:
        experiment_num = last_experiment_num + 1
    elif experiment_num == last_experiment_num:
        do_update = True
    elif experiment_num < last_experiment_num:
        log.warning(f'Defined experiment num \'{experiment_num}\' is smaller than \'{last_experiment_num}\'. Aborting '
                    f'save process in order to prevent data loss.')
        return None

    data = {'experiment_num': f'{experiment_num:02d}', 'network_type': str(net),
            'time_finished': datetime.datetime.now().strftime('%d.%m.%y - %H:%M')}

    # add all parameters used for creating the cann
    for param_name in params:
        if param_name in optimized_params.keys():
            continue
        value = getattr(net.p, param_name)
        if value is not None:
            data[param_name] = value

    # add all optimized params
    for param_name, param_values in optimized_params.items():
        value_range = f'[{param_values[MIN]}, {param_values[MAX]}, {param_values[STEP]}]'
        opt_param_name = f'opt_{param_name}'
        data[opt_param_name] = value_range

    # add optimization config values
    for param_name, param_value in optimization_config.items():
        data[param_name] = param_value

    save_setup(net, data, ExperimentType.OPTIMIZATION, experiment_num, create_eval_file, do_update, **kwargs)

    return experiment_num


def save_setup(net, data, experiment_type, experiment_num, create_eval_file, do_update, **kwargs):
    file_path = join(EXPERIMENT_FOLDERS[str(net)], EXPERIMENT_SETUP_FILE_NAME[experiment_type])

    # add all static parameters defined above for this specific experiment
    for param_name in sorted(kwargs):
        data[param_name] = kwargs[param_name]

    # check if new parameters were added since last time
    do_update_headers = False
    headers = []
    values = []
    lines = []
    if not create_eval_file:
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            lines = list(csv_reader)

        for header in lines[0]:
            headers.append(header)
            if header in data.keys():
                values.append(data[header])
                data.pop(header)
            else:
                values.append('None')

        if len(data.keys()) > 0:
            do_update_headers = True

    for header, value in data.items():
        headers.append(header)
        values.append(value)

    # writing to csv file
    with open(file_path, 'w' if create_eval_file or do_update_headers or do_update else 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the headers if file is newly created
        if create_eval_file or do_update_headers or do_update:
            csvwriter.writerow(headers)

        # update last line if this is only an update
        if do_update:
            experiment_num_col = headers.index('experiment_num')
            for line in lines[1:]:
                if int(line[experiment_num_col]) == experiment_num:
                    line = values
                csvwriter.writerow(line)

        if do_update_headers:
            for line in lines[1:]:
                for header in data.keys():
                    line.append('None')
                csvwriter.writerow(line)

        # writing the data row
        if not do_update:
            csvwriter.writerow(values)

    return experiment_num


def get_last_experiment_num(net, experiment_id, experiment_type) -> int:
    file_path = join(EXPERIMENT_FOLDERS[str(net)], EXPERIMENT_SETUP_FILE_NAME[experiment_type])

    if not exists(file_path):
        return 0

    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        lines = list(csv_reader)

    for line in lines[::-1]:
        if line[0] == experiment_id:
            return int(line[1])

    return 0


def get_last_optimization_num(net) -> int:
    file_path = join(EXPERIMENT_FOLDERS[net.__name__], EXPERIMENT_SETUP_FILE_NAME[ExperimentType.OPTIMIZATION])

    if not exists(file_path):
        return 0

    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        lines = list(csv_reader)

    return int(lines[-1][0])


def eq_lists(l1, l2):
    # here l1 and l2 must be lists
    if len(l1) != len(l2):
        return False
    elif sorted(l1) == sorted(l2):
        return True
    else:
        return False


def save_evaluation_data(net: ContinuousAttractorNetwork, experiment_id, num_runs, **kwargs):
    # experiment_file_path = join(EXPERIMENT_FOLDERS[str(net)], EVALUATION_SETUP_FILE_NAME)
    eval_file_name = f'{EXPERIMENT_FILE_NAME[ExperimentType.EVALUATION]}_{experiment_id:02d}.csv'
    eval_file_path = join(EXPERIMENT_FOLDERS[str(net)], EXPERIMENT_SUBFOLDERS[FileType.DATA], eval_file_name)

    headers = []
    values = []
    for param_name in sorted(kwargs):
        headers.append(param_name)
        values.append(kwargs[param_name])

    # writing to csv file
    with open(eval_file_path, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the headers if file is newly created
        csvwriter.writerow(headers)

        # writing the data rows
        for i in range(num_runs):
            row = []
            for v in values:
                if len(v) > i:
                    row.append(v[i])
            csvwriter.writerow(row)


def load_configuration(config_id, can_type):
    config = load_yaml(PATH_CONFIG, FILE_NAME_CONFIG_EVAL[can_type.__name__])[f'experiment_{config_id}']

    if GLOBAL_METRICS in config.keys():
        for global_metric_name, global_metric_stats in config[GLOBAL_METRICS].items():
            if Statistics.MEDIAN in global_metric_stats:
                global_metric_stats += Statistics.MEDIAN_ADD

    return config


def load_experimental_data(experiment_type, network_type, experiment_num, experiment_id=None, metric=None):
    """
    Loads data points from a CSV file. The first line of the CSV file is expected to contain headers (metrics).
    :param experiment_id: The alphabetic index of the experiment for which data shall be loaded.
    :param experiment_num: The numeric index of the experiment for which data shall be loaded.
    :param metric: Optionally defines the metric which should be returned.
    :return: A list of all metrics or only of the metric given in the parameters of the function.
    """

    experiment_id_num = f'{experiment_id}-{experiment_num:02d}' if experiment_id is not None else f'{experiment_num:02d}'
    experiment_file_name = f'{EXPERIMENT_FILE_NAME[experiment_type]}_{experiment_id_num}.csv'
    file_path = join(EXPERIMENT_FOLDERS[network_type.__name__], EXPERIMENT_SUBFOLDERS[FileType.DATA],
                     experiment_file_name)

    if not exists(file_path):
        return 0

    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        lines = list(csv_reader)

    data = dict()
    headers = lines[0]
    for header in headers:
        data[header] = []

    for line in lines[1:]:
        for header_id, header in enumerate(headers):
            data[header].append(float(line[header_id]))

    if metric is not None:
        return data[metric]
    else:
        return data


def load_experimental_config(experiment_type: ExperimentType, network_type, experiment_num, experiment_id=None):
    """
    Loads the configuration of a previously performed experiment from a CSV file. The first line of the CSV file is
    expected to contain headers (metrics).
    :param experiment_type: The type of the experiment performed.
    :param network_type: The type of the network used for the experiment.
    :param experiment_num: The numeric index of the experiment for which the config shall be loaded.
    :param experiment_id: The alphabetic index of the experiment for which the config shall be loaded.
    :return: A list of all metrics or only of the metric given in the parameters of the function.
    """

    file_path = join(EXPERIMENT_FOLDERS[network_type.__name__], EXPERIMENT_SETUP_FILE_NAME[experiment_type])

    if not exists(file_path):
        return 0

    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        lines = list(csv_reader)

    data = dict()
    headers = lines[0]
    csv_experiment_id = None
    csv_experiment_num = None
    for header_id, header in enumerate(headers):
        data[header] = None
        if header == 'experiment_id':
            csv_experiment_id = header_id
        elif header == 'experiment_num':
            csv_experiment_num = header_id

    data_found = False
    for line in lines[1:]:
        if (experiment_id is None or line[csv_experiment_id] == experiment_id) and \
                int(line[csv_experiment_num]) == experiment_num:
            data_found = True
            for header_id, header in enumerate(headers):
                data[header] = line[header_id]
            break

    if data_found:
        return data
    else:
        return None


def convert_experimental_config(config):
    new_config = copy(config)
    for key in new_config.keys():
        if new_config[key].isnumeric():
            new_config[key] = int(new_config[key])
        elif is_float(new_config[key]):
            new_config[key] = float(new_config[key])
        elif new_config[key] == 'True':
            new_config[key] = True
        elif new_config[key] == 'False':
            new_config[key] = False
        elif new_config[key] == 'None':
            new_config[key] = None
        elif '[' in new_config[key] and ']' in new_config[key]:
            new_config[key] = eval(new_config[key])
    return new_config


def save_model_object(obj: object, network_type, model_description, model_id):
    file_name = f'{type(obj).__name__}_{network_type.__name__}_{model_description}_{model_id}.pkl'
    file_path = join(EXPERIMENT_FOLDERS[network_type.__name__], EXPERIMENT_SUBFOLDERS[FileType.MODEL], file_name)

    with open(file_path, 'wb') as out_file:
        pickle.dump(obj, out_file)


def load_model_object(object_type, network_type, model_description, model_id):
    file_name = f'{object_type.__name__}_{network_type.__name__}_{model_description}_{model_id}.pkl'
    file_path = join(EXPERIMENT_FOLDERS[network_type.__name__], EXPERIMENT_SUBFOLDERS[FileType.MODEL], file_name)

    if not exists(file_path):
        log.error(f'Could not load pickled object from {file_path} because it doesn\'t seem to exist.')
        return None

    with open(file_path, 'rb') as in_file:
        obj = pickle.load(in_file)

    return obj


def save_optimization_state(opt_alg, experiment_num=None):
    """
    Save the state (serialize) of an optimization algorithm to the evaluation folder of the network used in the
    optimization.
    :param opt_alg: The optimization algorithm object to be saved.
    :param experiment_num int: An optional experiment number in order to save state individually.
    :return: None
    """
    if experiment_num is not None:
        file_name = f'{type(opt_alg).__name__}_{opt_alg.can_class.__name__}_{experiment_num:02d}.pkl'
    else:
        file_name = f'{type(opt_alg).__name__}_{opt_alg.can_class.__name__}.pkl'
    file_path = join(EXPERIMENT_FOLDERS[opt_alg.can_class.__name__], EXPERIMENT_SUBFOLDERS[FileType.OPTIMIZATION],
                     file_name)

    with open(file_path, 'wb') as out_file:
        pickle.dump(opt_alg, out_file)


def load_optimization_state(opt_alg_type, can_type, experiment_num=None):
    """
    Load the state (serialize) of an optimization algorithm from the evaluation folder of the network used in the
    optimization.
    :param opt_alg_type: The type of the optimization algorithm object to be loaded.
    :param can_type: The type of the CAN used in the optimization algorithm.
    :param experiment_num int: An optional experiment number in order to lead individual state.
    :return:
    """
    if experiment_num is not None:
        file_name = f'{opt_alg_type.__name__}_{can_type.__name__}_{experiment_num:02d}.pkl'
    else:
        file_name = f'{opt_alg_type.__name__}_{can_type.__name__}.pkl'
    file_path = join(EXPERIMENT_FOLDERS[can_type.__name__], EXPERIMENT_SUBFOLDERS[FileType.OPTIMIZATION], file_name)

    if not exists(file_path):
        log.error(f'Could not load pickled object from {file_path} because it doesn\'t seem to exist.')
        return None

    with open(file_path, 'rb') as in_file:
        obj = pickle.load(in_file)

    return obj
