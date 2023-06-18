from msmfcode.core.config import *
from msmfcode.evaluation.plot import plot_optimization_results_3d, plot_optimization_results_3d_sequential
from msmfcode.models.cann import MSMFSingleCAN, MSMFMultiCAN

import plotly.io as pio
pio.renderers.default = 'browser'


# Config
plot_sequentially = False
net = MSMFMultiCAN
experiment_nums = [15]

param_ranges = dict()
# param_ranges['max_field_sizes'] = [10, 50]
# param_ranges['theta'] = [0, 0.05]
param_ranges[Metric.POS_ERROR_MEAN + '_mean'] = [0, 1.0]

plot_params = list()
# plot_params.append(Metric.AVG_NUM_FIELDS_PER_NEURON + '_mean')
# plot_params.append('max_field_sizes')
# plot_params.append('alpha')
# plot_params.append('theta')

# evaluation_metric = Metric.AVG_NUM_FIELDS_PER_NEURON + '_mean'
# evaluation_metric = Metric.POS_ERROR_MEAN + '_mean'
# evaluation_metric = 'max_field_sizes'
evaluation_metric = 'theta'

# color_param = Metric.AVG_NUM_FIELDS_PER_NEURON + '_mean'
color_param = Metric.POS_ERROR_MEAN + '_mean'
# color_param = 'max_field_sizes'

add_params = list()
# add_params.append('max_field_sizes')
# add_params.append(Metric.POS_ERROR_MEAN + '_std')
# add_params.append('theta')

# Plotting
plot_optimization_results_3d(net, experiment_nums, param_ranges,
                             plot_params, evaluation_metric, color_param, add_params, plot_sequentially, theme='dark')
