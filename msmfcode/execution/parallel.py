import numpy as np
from time import time
import multiprocessing as mp
import traceback
from datetime import timedelta

from msmfcode.core.config import *
from msmfcode.core.logging import log
from msmfcode.evaluation.decoding import pop_decoding
from msmfcode.evaluation.plot import plot_error, plot_neuron_activity_static
from msmfcode.evaluation.data import load_configuration, EvaluationData, get_last_experiment_num, \
    save_experimental_setup, save_model_object
from msmfcode.models.cann import DMSMF, DiffSolverError


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class Executor:
    def __init__(self, experiment_id, can_type, parameters=None):
        # Load configuration from yaml file
        self.experiment_id = experiment_id
        self.config = load_configuration(self.experiment_id, can_type)
        self.can_type = can_type
        self.network_parameters = parameters if parameters is not None else dict()
        # Merge dicts with config as basis and network_parameters as update
        self.network_parameters = {**self.config[NETWORK_PARAMS], **self.network_parameters}
        self.config[METRICS] = list(self.config[GLOBAL_METRICS].keys())

        # Create NUM_EVAL_ITERS can models
        self.cans = list()
        for i in range(self.config[NUM_EVAL_ITERS]):
            can = can_type(index=i, random_seed=RandomSeed.TIME, collected_metrics=self.config[METRICS],
                           add_params=self.network_parameters)
            self.cans.append(can)

        # Create new evaluation data object for monitoring mean/total values of parameters throughout the runs
        self.eval_data = None
        self.init_eval_data()

        # Get new experiment number
        experiment_type = ExperimentType.EVALUATION_SINGLE if type(self) is ParallelExecutor \
            else ExperimentType.EVALUATION
        self.experiment_num = 1 + get_last_experiment_num(self.cans[0], self.experiment_id, experiment_type)

    def init_eval_data(self):
        self.eval_data = EvaluationData(str(self.cans[0]), self.config[METRICS])

    def print_metrics(self):
        log.info(f'>> Metrics after {self.config[NUM_EVAL_ITERS]} runs:')
        for metric_name, metric_data in self.eval_data.eval_metrics.items():
            if len(metric_data.mean) <= 0:
                continue
            log.info(f'>> {metric_name} = {metric_data.mean[-1]:.2f}')

    def print_global_metrics(self):
        log.info(f'>> Metrics after {self.config[NUM_EVAL_ITERS]} runs:')
        for metric_name, global_metrics in self.config[GLOBAL_METRICS].items():
            for global_metric in global_metrics:
                metric_data = self.eval_data.eval_metrics[metric_name][global_metric]
                if len(metric_data) <= 0:
                    continue
                log.info(f'>> {metric_name}_{global_metric} = {metric_data[-1]:.2f}')
        # for metric_name, metric_data in self.eval_data.eval_metrics.items():
        #     if len(metric_data.mean) <= 0:
        #         continue
        #     log.info(f'>> {metric_name} = {metric_data.mean[-1]:.2f}')

    @staticmethod
    def parallel_exec_can(can_local, queue_cans, reset_fields=True, reset_weights=True, decoding_threshold=None):
        try:
            # initialize seed, fields and weights
            if reset_fields or reset_weights:
                can_local.init_random_state(RandomSeed.TIME)
                log.debug(f'Initialized random state for CAN {can_local.id}')
            if reset_fields:
                can_local.init_fields()
                log.debug(f'Initialized fields for CAN {can_local.id}')
            if reset_weights:
                can_local.init_weights()
                log.debug(f'Initialized weights for CAN {can_local.id}')

            can_local.run()
            log.debug(f'Finished run of CAN {can_local.id}')

            estimated_positions = pop_decoding(can_local, False, decoding_threshold=decoding_threshold)
            can_local.pos_estimated = estimated_positions
        except Exception as e:
            queue_cans.put(can_local)
            raise e

        queue_cans.put(can_local)

        log.debug(f'Finished execution of CAN {can_local.id}')

    def run_batch(self, reset_fields, reset_weights, can_range=None, remove_finished_nets=False):
        if can_range is None:
            can_range = range(self.config[NUM_EVAL_ITERS])

        q_cans = mp.Queue()

        # Create and start NUM_EVAL_ITERS processes
        processes = []
        decoding_threshold = self.config[DECODING_THRESHOLD] if DECODING_THRESHOLD in self.config.keys() else None
        for i, i_eval_iter in enumerate(can_range):
            log.debug(f'>>> Starting evaluation iteration {i_eval_iter}')
            processes.append(Process(target=self.parallel_exec_can, args=(self.cans[i_eval_iter], q_cans,
                                                                          reset_fields, reset_weights,
                                                                          decoding_threshold)))
            processes[i].start()

        log.info(f'>> Started all CAN simulations')

        # Retrieve results from each process in form of a can in the queue
        num_received_cans = 0
        while num_received_cans < len(can_range):
            log.debug(f'Waiting for CAN [{num_received_cans + 1}/{len(can_range)}]')
            can = q_cans.get()
            self.cans[can.id] = can
            num_received_cans += 1

        # Wait for all processes to stop
        for i in range(len(can_range)):
            processes[i].join()

            # Check if an exception occurred in the sub-process, then raise this exception
            if processes[i].exception:
                exc, trc = processes[i].exception
                print(trc)
                raise exc

        log.info(f'>> Finished all CAN simulations')

        # Perform further processing with the retrieved data
        # Todo: Maybe change the limit here to 1
        for i_eval_iter in can_range:
            self.eval_data.update_step(self.cans[i_eval_iter], self.cans[i_eval_iter].pos_estimated, limit=2)
            if remove_finished_nets:
                del self.cans[i_eval_iter]
                self.cans = [None] + self.cans
            log.debug(f'>>> Finished evaluation iteration {i_eval_iter}')

    def run_all_batches(self, reset_fields, reset_weights, remove_finished_nets, cpu_cores_max=None):
        # Calculate number of batch runs based on number of total eval iters and available CPU cores
        cpu_cores = mp.cpu_count() if cpu_cores_max is None else cpu_cores_max
        num_batch_runs = int(np.ceil(self.config[NUM_EVAL_ITERS] / cpu_cores))
        for i_batch_runs in range(num_batch_runs):
            log.info(f'>> Starting batch run [{i_batch_runs + 1}/{num_batch_runs}]')

            can_range = range(i_batch_runs * cpu_cores,
                              min((i_batch_runs + 1) * cpu_cores, self.config[NUM_EVAL_ITERS]))
            self.run_batch(reset_fields, reset_weights, can_range=can_range, remove_finished_nets=remove_finished_nets)

    def save_data(self, experiment_type, show_plots=False, save_plots=True, save_model=False, eval_params=None,
                  override_data=False, **kwargs):
        # Save parameters and evaluation data
        eval_params = eval_params if eval_params is not None else dict()

        # Add global metrics from normal metrics to eval_params
        for metric_name, global_metrics in self.config[GLOBAL_METRICS].items():
            for global_metric in global_metrics:
                if experiment_type == ExperimentType.EVALUATION:
                    eval_params[f'{metric_name}_{global_metric}'] = getattr(self.eval_data.eval_metrics[metric_name],
                                                                            global_metric)
                else:
                    eval_params[f'{metric_name}_{global_metric}'] = np.round(
                        getattr(self.eval_data.eval_metrics[metric_name],
                                global_metric)[0], 3)

        if experiment_type == ExperimentType.EVALUATION_SINGLE:
            kwargs = {**kwargs, **eval_params}

        self.experiment_num = save_experimental_setup(self.cans[0], self.experiment_id, experiment_type,
                                                      num_eval_iters=self.config[NUM_EVAL_ITERS],
                                                      experiment_num=self.experiment_num if override_data else None,
                                                      **kwargs)

        self.eval_data.save(self.cans[0], self.experiment_id, self.experiment_num, eval_params, experiment_type)

        if save_model:
            save_model_object(self, self.can_type,
                              model_description=f'{EXPERIMENT_FILE_NAME[ExperimentType.EVALUATION]}',
                              model_id=f'{self.experiment_id}-{self.experiment_num:02d}')


class ParallelExecutor(Executor):
    def __init__(self, experiment_id, can_type, parameters=None):
        super().__init__(experiment_id, can_type, parameters)

    def run(self, reset_fields=True, reset_weights=True, remove_finished_nets=False, cpu_cores_max=None):

        log.info(f'> Starting experiment {self.experiment_id}-{self.experiment_num:02d}: {self.config[DESCRIPTION]}')

        # Setup timing
        time_start = time()

        log.info(f'>> Starting evaluation run')

        self.run_all_batches(reset_fields, reset_weights, remove_finished_nets, cpu_cores_max=cpu_cores_max)

        # Finish up the run
        self.eval_data.update_run(reset=False)

        log.info('-------------------------------------------------------------------------------------')
        if GLOBAL_METRICS in self.config.keys():
            self.print_global_metrics()
        else:
            self.print_metrics()
        log.info('-------------------------------------------------------------------------------------')

        # Finish timing
        time_end = time()
        log.info(f'> Finished experiment {self.experiment_id}-{self.experiment_num:02d} in '
                 f'{time_end - time_start:.2f} seconds.')

    def save_data(self, show_plots=False, save_plots=True, save_model=False, eval_params=None, override_data=False,
                  **kwargs):
        eval_params = eval_params if eval_params is not None else dict()

        super().save_data(ExperimentType.EVALUATION_SINGLE, show_plots=show_plots, save_plots=save_plots,
                          save_model=save_model, eval_params=eval_params, override_data=override_data,
                          description=self.config[DESCRIPTION], label=self.config[LABEL])

        # Save plots
        if PlotType.NEURON_ACTIVITY_STATIC in self.config[PLOTS].keys() and show_plots:
            if self.config[PLOTS][PlotType.NEURON_ACTIVITY_STATIC]:
                plot_neuron_activity_static(self.cans[0], experiment_num=self.experiment_num,
                                            fig_title=self.config[DESCRIPTION],
                                            show_plot=show_plots, save_plot=save_plots)


class ParallelEvaluationExecutor(Executor):
    def __init__(self, experiment_id, can_type, parameters=None):
        super().__init__(experiment_id, can_type, parameters)

        # Define local parameters from self.config
        self.run_time = None
        self.param_eval_range = np.round(np.arange(self.config[PARAM_EVAL_RANGE_MIN],
                                                   self.config[PARAM_EVAL_RANGE_MAX],
                                                   self.config[PARAM_EVAL_RANGE_STEP]), 6)
        self.param_eval_range_params = (
            self.config[PARAM_EVAL_RANGE_MIN], self.config[PARAM_EVAL_RANGE_MAX], self.config[PARAM_EVAL_RANGE_STEP])

    def run(self, reset_fields=True, reset_weights=True, remove_finished_nets=False, cpu_cores_max=None):

        log.info(f'> Starting experiment {self.experiment_id}-{self.experiment_num:02d}: {self.config[DESCRIPTION]}')
        log.info(f'> Evaluated parameter: {self.config[PARAM_NAME]} in  range{self.param_eval_range_params}')

        # Setup timing
        time_start = time()

        i_eval_run = 1
        for param_i, param in enumerate(self.param_eval_range):
            time_run_start = time()
            # Reset evaluation data
            self.eval_data.reset()
            # Reset all networks with the new parameter
            self.network_parameters[self.config[PARAM_NAME]] = param
            for i in range(self.config[NUM_EVAL_ITERS]):
                self.cans[i].__init__(index=i, random_seed=RandomSeed.TIME, collected_metrics=self.config[METRICS],
                                      add_params=self.network_parameters)

            log.info(f'>> Starting evaluation run {i_eval_run} with ')
            log.info(f'>> {self.config[PARAM_NAME]} = {param}')

            self.run_all_batches(reset_fields if param_i > 0 else True, reset_weights if param_i > 0 else True,
                                 remove_finished_nets=remove_finished_nets, cpu_cores_max=cpu_cores_max)

            # Finish up the run
            self.eval_data.update_run()
            i_eval_run += 1

            run_time = int(time() - time_run_start)
            est_time_left = (len(self.param_eval_range) - (param_i + 1)) * run_time

            log.info('-------------------------------------------------------------------------------------')
            self.print_metrics()
            log.info(f'>> Run-time = {timedelta(seconds=run_time)} seconds')
            log.info(f'>> Total time left = {timedelta(seconds=est_time_left)} seconds')
            log.info('-------------------------------------------------------------------------------------')

        # Finish timing
        time_end = time()
        self.run_time = int(time_end - time_start)
        log.info(f'> Finished experiment {self.experiment_id}-{self.experiment_num:02d} in '
                 f'{self.run_time} seconds.')

    def save_data(self, show_plots=False, save_plots=True, save_model=False, eval_params=None, eval_metrics=None,
                  override_data=False, **kwargs):
        eval_params = eval_params if eval_params is not None else dict()

        # Add evaluation range to eval_params
        eval_params[self.config[PARAM_NAME]] = self.param_eval_range
        # Add manually calculated metrics to eval_params
        if self.can_type is DMSMF and (self.cans[0].p.field_ratio_threshold is not None or
                                               self.cans[0].p.field_connection_prob is not None) and \
                'num_field_con_pruned' in self.eval_data.eval_metrics.keys() and \
                'num_field_con_total' in self.eval_data.eval_metrics.keys():
            eval_params['ignored_field_connections_ratio'] = self.eval_data.get_ratio('num_field_con_pruned',
                                                                                      'num_field_con_total')

        super().save_data(ExperimentType.EVALUATION, show_plots=show_plots, save_plots=save_plots,
                          save_model=save_model,
                          eval_params=eval_params,
                          override_data=override_data,
                          evaluated_parameter=self.config[PARAM_NAME],
                          evaluated_range=self.param_eval_range_params,
                          description=self.config[DESCRIPTION],
                          label=self.config[LABEL],
                          run_time=self.run_time)

        # ToDo: Check if this also makes sense for a single run (ParallelExecutor) or could be used after manually
        #  running the algorithm several times
        # Save plots and show them if enabled
        for plot_type, plot_enabled in self.config['plots'].items():
            if plot_enabled:
                plot_error(type(self.cans[0]), self.param_eval_range, self.eval_data.eval_metrics[plot_type].mean,
                           self.experiment_id,
                           self.experiment_num,
                           x_label=self.config['param_name'], y_label=plot_type, img_name=plot_type,
                           save_plot=save_plots, show_plot=show_plots)
