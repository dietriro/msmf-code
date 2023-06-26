import os
import csv
import webbrowser

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.offline as offline

from msmfcode.core.config import *
from msmfcode.core.logging import log
from msmfcode.evaluation.data import load_experimental_config, load_experimental_data, convert_experimental_config
from msmfcode.models.cann import DMSMF, FMSMF, ContinuousAttractorNetwork, VariableCAN

plot_names = set()

line_styles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5, 1, 5))]


def plot_error(net, x_values, y_values, experiment_id, experiment_num, err_type='mean', img_name='', x_label='',
               y_label=None, plot_type='line', label=None, color_id=0, line_style_id=0, plot_fitted_poly=False,
               max_poly=10, plot_min_y=False, show_grid=True, save_plot=False, use_all_plot_names=False,
               second_plot=False, show_legend=False, show_plot=False, clear_plot=True, plot_size=None, new_figure=True,
               legend_loc='right'):
    global plot_names

    if y_label is None:
        y_label = f'{err_type.capitalize()} error (m)'

    if img_name == '':
        img_name = err_type
    else:
        img_name = img_name.replace('_', '-')

    plot_names.add(f'{experiment_id}-{experiment_num:02d}')

    if use_all_plot_names:
        plot_name = '-'.join(plot_names)
    else:
        plot_name = f'{experiment_id}-{experiment_num:02d}'

    img_file_path = join(EXPERIMENT_FOLDERS[net.__name__], EXPERIMENT_SUBFOLDERS[FileType.FIGURE],
                         f'{net.__name__}_{plot_name}_{img_name}.{PlotConfig.FILE_TYPE}')

    plt.rc('grid', linestyle=':')

    if second_plot:
        plt.twinx()
    else:
        if new_figure:
            plt.figure(figsize=plot_size)

    if 'line' in plot_type:
        plt.plot(x_values, y_values, label=label, color=f'C{color_id}', linestyle=line_styles[line_style_id])
    if 'circles' in plot_type:
        plt.scatter(x_values, y_values, label=None if 'line' in plot_type else label, facecolors='None',
                    edgecolors=f'C{color_id}')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(show_grid)
    # plt.ylim(0, 115)
    # plt.ylim(0, 125)
    # plt.xlim(-5, 140)

    plt.tight_layout()

    if plot_fitted_poly:
        polyline, data_fit, poly = get_fitted_data(x_values, y_values, max_poly=max_poly)
        log.info(f'Found suitable polynomial function with degree {poly}')
        plt.plot(polyline, data_fit, label=f'{label} ({poly}. deg. poly. fun.)', color=f'C{color_id}')

        if plot_min_y:
            xax_min, xax_max = plt.xlim()
            yax_min, yax_max = plt.ylim()
            xax_rng = xax_max - xax_min
            yax_rng = yax_max - yax_min

            min_y = np.min(data_fit)
            min_x = polyline[np.argmin(data_fit)]
            plt.axvline(x=min_x, ymax=min_y / yax_rng, color=f'C{color_id}', linestyle='--')
            plt.axhline(y=min_y, xmax=min_x / xax_rng, color=f'C{color_id}', linestyle='--')

            log.info(f'Fitted min point: ({min_x:.2f}, {min_y:.2f})')
            log.info(f'Real min point: ({x_values[np.argmin(y_values)]:.2f}, {np.min(y_values):.2f})')

        # plt.text(min_x-18+color_id*21, 60, f'min={np.min(data_fit):.2f}')

    if label is not None and show_legend:
        if second_plot:
            plt.legend(loc='upper left')
        else:
            plt.legend(loc=f'upper {legend_loc}')
        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    if save_plot:
        plt.savefig(img_file_path)
    if show_plot:
        plt.show()
    if clear_plot:
        plt.clf()


def gen_plot_error_box(network_types, experiment_id, experiment_numbers, metric, metric_type=Statistics.MEAN,
                       add_metric=None, fig_size=None, fig_title='', fig_name='error_box-plot', fig_xlabels=None,
                       y_label_first=None, y_label_second=None, save_fig=False, show_fig=True, plot_type='box',
                       positions=None):
    # Data
    data_add = []
    num_cols = 4 if metric_type == Statistics.MEAN else 5
    num_experiments = np.sum([len(l) for l in experiment_numbers])
    data = np.zeros((num_experiments, num_cols))
    data_full = []

    if len(network_types) == 1:
        network_type_name = network_types[0].__name__
    else:
        network_type_name = NetworkType.GENERAL

    i_total = 0
    for i_net_type, network_type in enumerate(network_types):
        for i_exp in experiment_numbers[i_net_type]:
            config = load_experimental_config(ExperimentType.EVALUATION_SINGLE, network_type, i_exp,
                                              experiment_id=experiment_id)

            if config is None:
                log.warning(f'Cannot load data of type {ExperimentType.EVALUATION_SINGLE} for network '
                            f'{network_type.__name__} and experiment number {i_exp}')
                continue

            if plot_type == 'violin':
                data_full.append(np.array(load_experimental_data(ExperimentType.EVALUATION_SINGLE, network_type, i_exp,
                                                                 experiment_id=experiment_id, metric=metric),
                                          dtype=float))

            if metric_type == Statistics.MEAN:
                global_metrics = [Statistics.MEAN, Statistics.STD, Statistics.MIN, Statistics.MAX]
            elif metric_type == Statistics.MEDIAN:
                global_metrics = [metric_type] + Statistics.MEDIAN_ADD
            else:
                log.error(f'Parameter "metric_type" has to be set to either Statistics.MEAN or Statistics.MEDIAN. '
                          f'Cannot continue with current value: {metric_type}')
                return

            for i_metric, global_metric in enumerate(global_metrics):
                data[i_total, i_metric] = float(config[f'{metric}_{global_metric}'])

            if add_metric is not None:
                data_add.append(float(config[add_metric]))

            i_total += 1

    if i_total == 0:
        log.warning('Could not plot anything because no data was loaded.')
        return

    if plot_type == 'box':
        plot_error_box(data, network_type_name, metric_type, fig_size=fig_size, fig_title=fig_title,
                       fig_name=fig_name, fig_xlabels=fig_xlabels, y_label=y_label_first,
                       save_fig=False, show_fig=show_fig)
    else:
        plot_error_violin(data_full, network_type_name, metric_type, fig_size=fig_size, fig_title=fig_title,
                          fig_name=fig_name, fig_xlabels=fig_xlabels, y_label=y_label_first,
                          save_fig=False, show_fig=show_fig, positions=positions)

    if len(data_add) > 0:
        plot_metric_scatter(data_add, second_plot=True, y_label=y_label_second)

    if save_fig:
        plt.tight_layout()
        plt.savefig(join(EXPERIMENT_FOLDERS[network_type_name], EXPERIMENT_SUBFOLDERS[FileType.FIGURE],
                         f'{network_type_name}_{fig_name}.{PlotConfig.FILE_TYPE}'))


def plot_error_violin(data_full, network_type_name, metric_type, fig_size=None, fig_title='',
                   fig_name='error_vio-plot', fig_xlabels=None, y_label='Mean Positional Error (m)',
                   save_fig=False, show_fig=True, positions=None):
    if fig_size is None:
        # width, height
        fig_size = [10, 8]

    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["figure.autolayout"] = True

    if positions is None:
        positions = [i+1 for i in range(len(data_full))]

    _, ax = plt.subplots()
    violin_parts = ax.violinplot(data_full, positions=positions, widths=0.8,
                                 showmeans=True, showextrema=True, showmedians=True)

    for partname in ['cmedians']:
        vp = violin_parts[partname]
        vp.set_edgecolor('C1')
        # vp.set_linewidth(1)

    if fig_xlabels is not None:
        ax.set_xticks(positions, labels=fig_xlabels)

    plt.suptitle(fig_title, fontsize=14)
    plt.xlabel('Models (Parameters)')
    plt.ylabel(y_label)

    if save_fig:
        plt.tight_layout()
        plt.savefig(join(EXPERIMENT_FOLDERS[network_type_name], EXPERIMENT_SUBFOLDERS[FileType.FIGURE],
                         f'{network_type_name}_{fig_name}.{PlotConfig.FILE_TYPE}'))
    if show_fig:
        plt.show()


def plot_error_box(data, network_type_name, metric_type, fig_size=None, fig_title='',
                   fig_name='error_box-plot', fig_xlabels=None, y_label='Mean Positional Error (m)',
                   save_fig=False, show_fig=True):
    if fig_size is None:
        # width, height
        fig_size = [10, 8]

    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["figure.autolayout"] = True

    stats = []
    for i_plot in range(data.shape[0]):
        if metric_type == Statistics.MEAN:
            stats.append({'med': data[i_plot, 0],
                          'q1': data[i_plot, 0] - data[i_plot, 1],
                          'q3': data[i_plot, 0] + data[i_plot, 1],
                          'whislo': data[i_plot, 2],
                          'whishi': data[i_plot, 3]})
        elif metric_type == Statistics.MEDIAN:
            stats.append({'med': data[i_plot, 0],
                          'q1': data[i_plot, 1],
                          'q3': data[i_plot, 2],
                          'whislo': data[i_plot, 3],
                          'whishi': data[i_plot, 4]})
        else:
            log.error(f'Parameter "metric_type" has to be set to either Statistics.MEAN or Statistics.MEDIAN. '
                      f'Cannot continue with current value: {metric_type}')
            return

    _, ax = plt.subplots()
    ax.bxp(stats, showfliers=False)
    if fig_xlabels is not None:
        ax.set_xticklabels(fig_xlabels)

    plt.suptitle(fig_title, fontsize=14)
    plt.xlabel('Models')
    plt.ylabel(y_label)

    if save_fig:
        plt.tight_layout()
        plt.savefig(join(EXPERIMENT_FOLDERS[network_type_name], EXPERIMENT_SUBFOLDERS[FileType.FIGURE],
                         f'{fig_name}.{PlotConfig.FILE_TYPE}'))
    if show_fig:
        plt.show()


def plot_metric_scatter(data_y, data_x=None, second_plot=False, y_label=None, label=None, color_id=0):
    if data_x is None:
        data_x = range(1, len(data_y) + 1)

    if second_plot:
        plt.twinx()

    plt.ylabel(y_label)

    plt.scatter(data_x, data_y, label=label, facecolors='None', edgecolors=f'C{color_id}')


def get_fitted_data(data_x, data_y, num_values=1000, max_poly=4, print_results=True):
    mean_error_best = None
    mean_error_id = None
    for i_poly in range(1, max_poly + 1):
        model = np.poly1d(np.polyfit(data_x, data_y, i_poly))
        data_fit = model(data_x)
        mean_error = np.mean(np.abs(data_fit - data_y))
        if print_results:
            log.info(f'Polynomial function of degree {i_poly} has mean error = {mean_error}')
        if mean_error_best is None or mean_error < mean_error_best:
            mean_error_best = mean_error
            mean_error_id = i_poly

    polyline = np.linspace(np.min(data_x), np.max(data_x), 1000)
    model = np.poly1d(np.polyfit(data_x, data_y, mean_error_id))
    return polyline, model(polyline), mean_error_id


def plot_weight_distribution(weights: np.ndarray):
    plt.hist(weights.ravel(), bins=np.linspace(np.min(weights), np.max(weights), num=50))
    plt.xlabel('Weights')
    plt.ylabel(f'Number of weight occurrences')
    plt.show()


def plot_neuron_activity_fixed_attractors(net: FMSMF, neuron_range=None, pause_time=3.0):
    # Plot single neuron activity profile

    if neuron_range is None:
        neuron_range = range(net.ind.shape[0])

    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(2, 1, figsize=(12, 8))

    for neuron_id in neuron_range:

        # For Sine Function
        positions = net.pos * net.p.disc_step
        axis[0].plot(positions, net.m[neuron_id, :])
        axis[0].set_xlim(0, net.p.env_length)

        for k in range(net.ind.shape[1]):
            neuron_ind = np.argwhere(net.ind[:, k] == neuron_id).flatten()
            if len(neuron_ind) == 0:
                continue
            field_loc = net.th[neuron_ind, k]

            x_a = (field_loc - net.lam[k]) * net.p.disc_step
            x_b = (field_loc + net.lam[k]) * net.p.disc_step

            axis[1].plot([x_a, x_b], [1, 1], linewidth=3)
            axis[1].set_xlim(0, net.p.env_length)

        axis[1].set_xlabel("Position of the fields (m)")
        # Draw, wait and clear the previous data from the plot
        plt.draw()
        plt.pause(pause_time)
        axis[0].clear()
        axis[1].clear()


def plot_neuron_activity_variable_attractors(net: DMSMF, neuron_range=None, pause_time=3.0):
    # Plot single neuron activity profile

    if neuron_range is None:
        neuron_range = range(net.field_locs_bins.shape[0])

    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(2, 1, figsize=(12, 8))

    for neuron_id in neuron_range:

        # For Sine Function
        positions = net.pos * net.p.disc_step
        axis[0].plot(positions, net.m[neuron_id, :])
        axis[0].set_xlim(0, net.p.env_length)

        for k in range(net.field_locs_bins.shape[1]):
            if net.field_locs_bins[neuron_id, k] == 0:
                continue

            x_a = net.field_locs_bins[neuron_id, k] * net.p.disc_step - net.field_sizes[neuron_id, k] / 2
            x_b = net.field_locs_bins[neuron_id, k] * net.p.disc_step + net.field_sizes[neuron_id, k] / 2

            axis[1].plot([x_a, x_b], [1, 1], linewidth=3)
            axis[1].set_xlim(0, net.p.env_length)

        axis[1].set_xlabel("Field position in bins (0.5m)")
        # Draw, wait and clear the previous data from the plot
        plt.draw()
        plt.pause(pause_time)
        axis[0].clear()
        axis[1].clear()


def plot_neuron_activity_static(net: ContinuousAttractorNetwork, neuron_range=None, experiment_num=0, fig_title='',
                                fig_type_name='neuron-activity', show_plot=True, save_plot=False, ):
    if neuron_range is None:
        neuron_range = range(5)

    # Plot single neuron activity profile

    # Sort fields, so that the colors are representing the id of the fields
    field_locs_sorted, field_sizes_sorted = net.get_field_locs_sizes_sorted()

    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(len(neuron_range) * 2, 1, figsize=(12, 8))

    # Setup labels for both y-axes
    figure.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Activity", labelpad=32)
    plt.yticks([])
    tmp_axis_right = plt.twinx()
    tmp_axis_right.set_yticks([])
    tmp_axis_right.set_xticks([])
    tmp_axis_right.set_ylabel("Neuron ID", labelpad=20)

    for i, neuron_id in enumerate(neuron_range):
        axis_id = i * 2

        # For Sine Function
        positions = net.pos * net.p.disc_step
        axis[axis_id].plot(positions, net.m[neuron_id, :])
        axis[axis_id].set_xlim(0, net.p.env_length)
        axis[axis_id].set_ylim(-np.max(net.m) / 10, np.max(net.m))
        axis[axis_id].tick_params(axis="x", direction="in")
        axis[axis_id].set(xticklabels=[])

        axis[axis_id + 1].set_yticks([])
        axis_right = axis[axis_id + 1].twinx()
        for k in range(field_locs_sorted.shape[1]):
            if field_sizes_sorted[neuron_id, k] == 0:
                continue

            x_a = field_locs_sorted[neuron_id, k] * net.p.disc_step - field_sizes_sorted[neuron_id, k] / 2
            x_b = field_locs_sorted[neuron_id, k] * net.p.disc_step + field_sizes_sorted[neuron_id, k] / 2

            axis_right.plot([x_a, x_b], [neuron_id, neuron_id], linewidth=3)

        axis[axis_id + 1].set_xlim(0, net.p.env_length)
        axis_right.set_yticks([neuron_id])

        if axis_id < (len(neuron_range) - 1) * 2:
            axis[axis_id + 1].tick_params(axis="x", direction="in")
            axis[axis_id + 1].set(xticklabels=[])
        else:
            axis[axis_id + 1].set_xlabel("Position (m)")

    # Draw, wait and clear the previous data from the plot
    plt.subplots_adjust(hspace=0.0)
    if save_plot:
        plt.suptitle(fig_title, fontsize=14)
        plt.tight_layout()
        plot_name = f'single-{experiment_num:02d}'
        fig_name = f'{net}_{EXPERIMENT_FILE_NAME[ExperimentType.EVALUATION_SINGLE]}_' \
                   f'{plot_name}_{fig_type_name}.{PlotConfig.FILE_TYPE}'
        fig_path = join(EXPERIMENT_FOLDERS[str(net)], EXPERIMENT_SUBFOLDERS[FileType.FIGURE], fig_name)
        plt.savefig(join(EXPERIMENT_FOLDERS[str(net)], EXPERIMENT_SUBFOLDERS[FileType.FIGURE], fig_name))
    if show_plot:
        plt.show()


# Plotting for Evolutionary Optimization results
def plot_fitness(net_type, experiment_num, data_fun=np.mean):
    exp_setup = load_experimental_config(ExperimentType.OPTIMIZATION, net_type, experiment_num)
    num_generations = int(exp_setup['num_generations'])
    population_size = int(exp_setup['population_size'])

    fitness_all = np.zeros((num_generations, population_size))

    experiment_file_name = f'{EXPERIMENT_FILE_NAME[ExperimentType.OPTIMIZATION]}_{experiment_num:02d}.csv'
    with open(join(PY_PKG_PATH, EXPERIMENT_FOLDERS[net_type.__name__], EXPERIMENT_SUBFOLDERS[FileType.DATA],
                   experiment_file_name), 'r') as f:
        csv_reader = csv.reader(f)

        lines = list(csv_reader)

        headers = lines[0]
        id_entity_id = headers.index('entity_id')
        id_generation = headers.index('generation')
        id_fitness = headers.index('fitness')

        for line_id, line in enumerate(lines[1:]):
            generation = int(line[id_generation])
            entity_id = int(line[id_entity_id])
            fitness = float(line[id_fitness])

            fitness_all[generation, entity_id] = fitness

    data_x = range(num_generations)
    data_y = data_fun(fitness_all, 1)

    plt.plot(data_x, data_y)
    plt.show()


def save_plot(title):
    plt.suptitle(title, fontsize=14)

    plt.tight_layout()

    plt.savefig('test.pdf')


def convert_fig_to_html(fig, background_color='#111111'):
    plot_div = offline.plot(fig, output_type='div')

    template = """
    <head>
    <body style="background-color:{background_color:s};">
    </head>
    <body>
    {plot_div:s}
    </body>""".format(plot_div=plot_div, background_color=background_color)

    return template


def plot_optimization_results_3d(net: ContinuousAttractorNetwork, experiment_nums, param_ranges: dict,
                                 plot_params, eval_param, color_param, add_params, plot_sequentially, view, fig_title,
                                 fig_name, axis_titles=None, theme='dark', save_plot=False):
    data = load_experimental_data(ExperimentType.OPTIMIZATION, net, experiment_nums[0])
    for i_exp in range(1, len(experiment_nums)):
        data_tmp = load_experimental_data(ExperimentType.OPTIMIZATION, net, experiment_nums[i_exp])
        for key in data.keys():
            data[key] += data_tmp[key]

    config = load_experimental_config(ExperimentType.OPTIMIZATION, net, experiment_nums[0])
    config = convert_experimental_config(config)

    # Loop over all entries from experimental data
    data_np = np.zeros((len(data[GENERATION]), len(plot_params) + 2 + len(add_params)))
    num_points = 0
    for i in range(len(data[GENERATION])):
        num_params_fit = 0
        for param_name, param_range in param_ranges.items():
            # if fixed_param_min_value <= data[fixed_param_name][i] < fixed_param_max_value:
            if param_range[0] <= data[param_name][i] < param_range[1]:
                num_params_fit += 1

        if num_params_fit == len(param_ranges):
            for i_plot_param, plot_param in enumerate(plot_params + [eval_param, color_param] + add_params):
                if plot_param in data.keys():
                    data_np[num_points, i_plot_param] = data[plot_param][i]
                elif plot_param == 'n_att':
                    data_np[num_points, i_plot_param] = data['nmin'][i] + data['nmed'][i] + data['nmax'][i]
        num_points += 1

    # Remove all unfilled rows from data
    data_np = data_np[~((data_np == 0).all(1)), :]
    # Remove duplicates from the data
    data_np = np.unique(data_np, axis=0)

    add_param_labels = ''
    for i_add_param, add_param in enumerate(add_params):
        add_param_labels += '<b>%{customdata[' + str(4 + i_add_param) + ']}:</b> %{customdata[' + \
            str(4 + i_add_param + len(add_params)) + ']:.3f}<br>'
    add_param_data = np.array([data_np[:, i] for i in range(4, 4 + len(add_params))]).transpose()
    main_param_data = [np.full(data_np.shape[0], param_name) for param_name in plot_params + [eval_param, color_param] + add_params]
    main_param_data = np.array(main_param_data).transpose()

    # Generate additional text based on param ranges
    add_texts = list()
    add_text_tmp = '<b>Visualized parameter ranges:</b><br>'
    for param_name, param_range in param_ranges.items():
        add_text_tmp += f'{param_name} = {param_range} <br>'
    add_texts.append(add_text_tmp)

    # Generate additional text based on optimized parameters
    add_text_tmp = '<b>Optimized parameters ranges:</b><br>'
    for param_name in config.keys():
        if 'opt_' in param_name and config[param_name] is not None:
            add_text_tmp += f'{param_name.replace("opt_", "")} = {config[param_name]} <br>'
    add_texts.append(add_text_tmp)

    if add_param_data.size != 0:
        custom_data = np.concatenate((main_param_data, add_param_data), 1)
    else:
        custom_data = main_param_data

    if axis_titles is None:
        axis_titles = plot_params + [eval_param]

    if plot_sequentially:
        fig = plot_optimization_results_3d_sequential(data_np, custom_data, plot_params, eval_param, color_param,
                                                      add_param_labels, config)
    else:
        fig = plot_optimization_results_3d_full(data_np, custom_data, plot_params, eval_param, color_param,
                                                add_param_labels, fig_title, axis_titles, net.__name__, fig_name, view,
                                                save_plot=save_plot)

    if add_texts is not None:
        for i_text, add_text in enumerate(add_texts):
            fig.add_annotation(text=add_text,
                               align='left',
                               showarrow=False,
                               xref='paper',
                               yref='paper',
                               x=0.0+i_text*0.17,
                               y=1.07,
                               borderwidth=1,
                               font={'size': 18})

    if not save_plot:
        # Update layout according to parameter
        fig.update_layout(template=f'plotly_{theme}')
        if fig.layout.template.layout.paper_bgcolor == 'white':
            background_color = '#000000'
        elif fig.layout.template.layout.paper_bgcolor == 'black':
            background_color = '#111111'
        else:
            background_color = '#111111'

        # Convert figure to html, adding custom CSS (background color)
        fig_html = convert_fig_to_html(fig, background_color=background_color)

        html_file_path = os.path.join(PY_PKG_PATH, 'data', 'tmp', 'optimization_result.html')
        with open(html_file_path, 'w') as html_file:
            html_file.write(fig_html)

        webbrowser.open('file://' + os.path.realpath(html_file_path))

    return fig, data_np, plot_params + [eval_param, color_param] + add_params


def plot_optimization_results_3d_full(data_plot, custom_data, plot_params, eval_param, color_param, add_param_labels,
                                      fig_title, axis_titles, network_type_name, fig_name, view, save_plot=False):
    if save_plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # For each set of style and range settings, plot n random points in the box
        # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
        im = ax.scatter(data_plot[:, 0], data_plot[:, 1], data_plot[:, 2], c=data_plot[:, 3],  marker='o', alpha=1.0)
        ax.view_init(view[0], view[1])

        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label(axis_titles[3], rotation=270, labelpad=30)

        ax.set_xlabel(axis_titles[0], labelpad=10)
        ax.set_ylabel(axis_titles[1], labelpad=10)
        ax.set_zlabel(axis_titles[2], labelpad=10)

        ax.set_title(fig_title)

        plt.tight_layout()
        plt.savefig(join(EXPERIMENT_FOLDERS[network_type_name], EXPERIMENT_SUBFOLDERS[FileType.FIGURE],
                         f'{network_type_name}_{fig_name}.{PlotConfig.FILE_TYPE}'))

        plt.show()
    else:
        marker_data = go.Scatter3d(
            x=data_plot[:, 0],
            y=data_plot[:, 1],
            z=data_plot[:, 2],
            marker=dict(
                size=3,
                color=data_plot[:, 3],  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                opacity=1.0,
                colorbar=dict(thickness=20, title=color_param)
            ),
            opacity=1.0,
            mode='markers',
            customdata=custom_data,
            hovertemplate=
            '<b>%{customdata[0]}:</b> %{x:.3f}<br>' +
            '<b>%{customdata[1]}:</b> %{y:.3f}<br>' +
            '<b>%{customdata[2]}:</b> %{z:.3f}<br>' +
            '<b>%{customdata[3]}:</b> %{marker.color:.3f}<br>' +
            add_param_labels +
            '<extra></extra>'
        )
        layout = go.Layout(
            showlegend=False,
            scene=go.layout.Scene(
                xaxis=go.layout.scene.XAxis(title=axis_titles[0]),
                yaxis=go.layout.scene.YAxis(title=axis_titles[1]),
                zaxis=go.layout.scene.ZAxis(title=axis_titles[2])
            )
        )
        fig = go.Figure(data=marker_data, layout=layout)

    return fig


def plot_optimization_results_3d_sequential(data_plot, custom_data, plot_params, eval_param, color_param,
                                            add_param_labels, config):
    fig = go.Figure()

    # Add traces, one for each slider step
    for i_gen in np.arange(1, config[NUM_GENERATIONS] + 1):
        i_sample_start = (i_gen - 1) * config[POPULATION_SIZE]
        i_sample = i_gen * config[POPULATION_SIZE]
        fig.add_trace(
            go.Scatter3d(
                visible=False,
                x=data_plot[i_sample_start:i_sample, 0],
                y=data_plot[i_sample_start:i_sample, 1],
                z=data_plot[i_sample_start:i_sample, 2],
                marker=dict(
                    size=3,
                    color=data_plot[:i_sample, 3],  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.8,
                    colorbar=dict(thickness=20, title=color_param)
                ),
                opacity=0.8,
                mode='markers',
                customdata=custom_data,
                hovertemplate=
                '<b>%{customdata[0]}:</b> %{x:.3f}<br>' +
                '<b>%{customdata[1]}:</b> %{y:.3f}<br>' +
                '<b>%{customdata[2]}:</b> %{z:.3f}<br>' +
                '<b>%{customdata[3]}:</b> %{marker.color:.3f}<br>' +
                add_param_labels +
                '<extra></extra>'
            )
        )
        log.info(f'Created traces for generation {i_gen}')

    # Make 1st trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Showing data until generation " + str(i + 1)}],  # layout attribute
            label=str(i + 1)
        )
        step["args"][0]["visible"][:i + 1] = [True] * i  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Last visible generation: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    return fig


def plot_field_size_distribution(net: DMSMF):
    a = net.field_sizes.flatten()
    a = a[~(a == 0)]
    plt.hist(a)
    plt.title("histogram")
    plt.show()


def plot_distribution(values, ax=None, title=None, nbins=None, x_lim=None, y_lim=None):
    a = values.flatten()
    a = a[~(a <= 0)]

    if ax is None:
        fig = plt.figure(1)
        ax = fig.add_subplot(111)

    ax.hist(a, bins=nbins)
    ax.set_title(title)

    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])

    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    return ax
