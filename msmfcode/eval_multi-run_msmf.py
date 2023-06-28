import logging

import msmfcode_src
from msmfcode.core.logging import log
from msmfcode.core.config import *
from msmfcode.execution.parallel import ParallelExecutor, ParallelEvaluationExecutor
from msmfcode.models.cann import DMSMF, FMSMF, Grid, SSSF
from msmfcode.evaluation.plot import plot_weight_distribution, plot_neuron_activity_fixed_attractors, \
    plot_neuron_activity_variable_attractors


# Configuration
PLOT_FILE_TYPE = 'pdf'
experiment_id = '2-4-a'
log.handlers[LogHandler.STREAM].setLevel(logging.INFO)
log.handlers[LogHandler.FILE].setLevel(logging.DETAIL)


def main():
    pe = ParallelEvaluationExecutor(experiment_id, FMSMF)

    pe.run(reset_fields=True, cpu_cores_max=None)
    pe.save_data(show_plots=False, save_plots=False)
    # pe.cans[0].plot()
    # plot_neuron_activity_variable_attractors_static(pe.cans[0], neuron_range=[0, 1, 2, 3, 4])


if __name__ == '__main__':
    main()
