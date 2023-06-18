import numpy as np
import pandas as pd
from copy import copy

from msmfcode.core.logging import log
from msmfcode.core.config import *
from msmfcode.models.cann import MSMFGridCAN, MSMFMultiCAN, MSMFSingleCAN, DiffSolverError, EmptyFields
from msmfcode.execution.parallel import ParallelExecutor
from msmfcode.evaluation.data import load_experimental_data, load_experimental_config, convert_experimental_config

import warnings
warnings.filterwarnings("error")

# Config
experiment_type = ExperimentType.OPTIMIZATION
network_type = MSMFMultiCAN
experiment_nums = [1]
verification_metrics = {Metric.POS_ERROR_MEAN + '_mean': 0.3, Metric.AVG_NUM_FIELDS_PER_NEURON + '_mean': 0.1}
perc_samples = None
num_samples = 100

log.handlers[LogHandler.STREAM].setLevel(logging.ESSENS)
log.handlers[LogHandler.FILE].setLevel(logging.ESSENS)


def run_optimization_verification(experiment_type, network_type, experiment_num, verification_metrics,
                                  perc_samples=None, num_samples=None, experiment_id=None):

    # Load experimental data and configurations
    data = load_experimental_data(experiment_type, network_type, experiment_num, experiment_id=experiment_id)
    config = load_experimental_config(experiment_type, network_type, experiment_num)
    # Convert experimental configuration data to respective (correct) types
    config = convert_experimental_config(config)

    data_pd = pd.DataFrame(data)

    opt_param_start_id = data_pd.columns.get_loc('entity_id') + 1
    opt_param_end_id = data_pd.columns.get_loc('pos_error_mean_mean')
    opt_param_names = data_pd.columns[opt_param_start_id:opt_param_end_id].to_list()

    data_pd.drop_duplicates(subset=opt_param_names, inplace=True)
    data_pd.reset_index(inplace=True)

    num_samples_total = data_pd.shape[0]

    if num_samples is None:
        if perc_samples is None:
            log.error('Either perc_samples or num_samples has to be set.')
        num_samples = int(perc_samples * num_samples_total)

    samples_rand = np.random.choice(num_samples_total, num_samples, replace=False)
    verification_values = list()
    opt_param_values_all = list()
    for i_sample, i_sample_rand in enumerate(samples_rand):
        verification_values.append(list())
        opt_param_values = dict()
        for param_name in opt_param_names:
            opt_param_values[param_name] = type(config[f'opt_{param_name}'][0])(data_pd[param_name].loc[i_sample_rand])
        opt_param_values_all.append(copy(opt_param_values))

        log.essens(f'# [{i_sample+1}/{len(samples_rand)}] Verifying the correctness for params: {opt_param_values}\n')

        add_params = {**config, **opt_param_values}

        try:
            pe = ParallelExecutor(f'ga', network_type, parameters=add_params)
            pe.run()
        except (DiffSolverError, EmptyFields):
            log.warning(f'Could not evaluate network due to error: {DiffSolverError.__name__}. Skipping and trying '
                        f'next network.')
            continue
        except RuntimeWarning:
            pass
        except Exception as e:
            print(e)

        for v_metric_name, v_metric_threshold in verification_metrics.items():
            verification_metric_split = v_metric_name.rsplit('_', maxsplit=1)
            verification_value = pe.eval_data.get_last_result(verification_metric_split[0],
                                                              global_metric=verification_metric_split[1])
            verification_values[i_sample].append(verification_value)

            diff = np.abs(verification_value-data_pd[v_metric_name].loc[i_sample_rand])
            diff_ratio = diff / data_pd[v_metric_name].loc[i_sample_rand]

            log.essens(f'## Metric: {v_metric_name}')
            log.essens(f'### Original = {data_pd[v_metric_name].loc[i_sample_rand]:.3f}')
            log.essens(f'### New = {verification_value:.3f}')
            log.essens(f'### Difference = {diff:.3f}')
            log.essens(f'### Ratio = {diff_ratio:.3f}')

            if diff_ratio > v_metric_threshold:
                log.warning(f'### Difference ratio exceeded threshold {v_metric_threshold}!')

            log.essens('')

        log.essens('##############################################################################\n')


if __name__ == '__main__':
    for exp_num in experiment_nums:
        run_optimization_verification(experiment_type, network_type, exp_num, verification_metrics,
                                      num_samples=num_samples, perc_samples=perc_samples)
