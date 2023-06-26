import matplotlib.pyplot as plt
from msmfcode.core.config import *
from msmfcode.evaluation.plot import plot_error
from msmfcode.evaluation.data import load_experimental_data, load_experimental_config
from msmfcode.models.cann import DMSMF, FMSMF, Grid

ERROR_LABELS = {Metric.POS_ERROR_MEAN: 'Mean positional error (m)',
                Metric.POS_ERROR_STD: 'Std. deviation of positional error (m)',
                Metric.POS_ERROR: 'Positional error (m)',
                Metric.MEAN_FIELD_ACTIVITY: 'Mean field activity',
                Metric.PERC_CORRECT_FIELDS: 'Proportion of correct fields (%)',
                Metric.AVG_NUM_FIELDS_PER_NEURON: 'Avg. number of fields (per neuron)'}

# Config
# network_type = DMSMF
# network_type = FMSMF
network_type = Grid
# plt.rcParams['figure.dpi'] = 300
PLOT_FILE_TYPE = 'pdf'
experiment_id = '2-4-a'
experiment_range = [1]
num_exp = len(experiment_range)
second_axis = True
save_plot = False
show_legend = False
plot_size = (12, 9)
max_polys = [6, 3]
plot_min_y = [False, False]
# labels = ['V-MSMF [N=$num_neurons$, L=$env_length$]']
labels = ['V-MSMF (mean error) [N=$num_neurons$, L=$env_length$]',
          'V-MSMF (correct fields) [N=$num_neurons$, L=$env_length$]']

# labels = ['500 total neurons', '1000 total neurons']
# labels = ['1D Variable-MSMF (200 m)', '1D Variable-MSMF (400 m)']
# labels = ['1D Fixed-MSMF (mean)', '1D Fixed-MSMF (std. dev.)']
# plot_types = [Metric.POS_ERROR_MEAN+'_median', Metric.PERC_CORRECT_FIELDS+'_mean']
plot_types = [Metric.POS_ERROR_MEAN+'_mean', Metric.AVG_NUM_FIELDS_PER_NEURON+'_mean']
x_label = 'Max. total coverage of all fields per neuron (m)'
# x_label = 'Percentage of neurons per attractor - f'
# x_label = 'Total number of neurons - N'

# Load experimental data
data = []
ax = None
for i_plot, plot_type in enumerate(plot_types):
    for i_exp, experiment_num in enumerate(experiment_range):
        curr_data = load_experimental_data(ExperimentType.EVALUATION, network_type, experiment_num,
                                           experiment_id=experiment_id)
        data.append(curr_data)
        curr_config = load_experimental_config(ExperimentType.EVALUATION, network_type, experiment_num,
                                               experiment_id=experiment_id)

        x_data = curr_data[curr_config['evaluated_parameter']]
        y_data = curr_data[plot_type]
        print(y_data)
        # y_label = ERROR_LABELS[plot_type if i_plot == 0 else Metric.POS_ERROR]
        y_label = ERROR_LABELS[plot_type[::-1].split('_', 1)[1][::-1]]
        last_exp = i_exp == num_exp-1 and plot_type == plot_types[-1]
        i_total = i_plot*len(experiment_range)+i_exp

        labels[i_total] = labels[i_total].replace('$num_neurons$', curr_config[NUM_NEURONS])
        labels[i_total] = labels[i_total].replace('$env_length$', curr_config[ENV_LENGTH])

        plot_error(network_type, x_data[1:], y_data[1:], experiment_id, experiment_num, x_label=x_label, y_label=y_label,
                   img_name='_'.join(plot_types), plot_type='circles', label=labels[i_total], color_id=i_total,
                   plot_fitted_poly=False, max_poly=max_polys[i_total], plot_min_y=plot_min_y[i_total],
                   save_plot=(last_exp and save_plot), use_all_plot_names=True, second_plot=last_exp and second_axis,
                   show_legend=show_legend, show_plot=last_exp, clear_plot=False, plot_size=plot_size)

plt.show()









