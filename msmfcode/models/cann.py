import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from numpy.random import MT19937, RandomState, SeedSequence
from abc import abstractmethod, ABC
from time import time_ns
from functools import reduce

from msmfcode.core.config import *
from msmfcode.core.logging import log
from msmfcode.core.io import load_yaml
from msmfcode.core.helper import mod

# Specify matlab visulization parameter only if not 'headless'
if os.environ.get('DISPLAY') is not None:
    matplotlib.use("TkAgg")


class DiffSolverError(Exception):
    pass


class EmptyFields(Exception):
    pass


class Parameters:
    def __init__(self, can, params=None):
        ## Network configuration
        # type of background noise added to neurons
        self.noise_type = None
        # (mean) of background input/noise
        self.I = None
        # standard deviation of background input/noise
        self.I_std = None
        # maximum excitatory connection between neurons
        self.J1 = None
        # maximum inhibitory connection between neurons
        self.J0 = None
        # amplitude of a place-specific input
        self.Iloc = None
        # time constant
        self.tau = None

        ## Experiment configuration
        # length of the 1D environment in meters
        self.env_length = None
        # step size of the discretization
        self.disc_step = None
        # total number of neurons
        self.num_neurons = None
        # initial value for neurons
        self.init_value = None
        # total time in seconds
        self.T = None
        # speed of movement in bins/second
        self.vel = None
        # timestep for ode
        self.dt = None
        # expected mean activity for identifying a field
        self.expected_mean_field_activity = None
        # whether to remove potential background activity before calculating mean field activity
        self.mean_field_activity_remove_background = None
        # Errors above this threshold (m) is counted as a catastrophic error - 5% of environment size in Eliav et al. 2022
        self.cat_error_threshold = None

        ## Noise/uncertainty configuration
        # probability of deactivating a neuron (lesions)
        self.prob_dead_neurons = None
        # the exact id's of the dead neurons
        self.dead_neuron_ids = None
        # random seed type for generation of randomized numbers
        self.random_seed_type = None
        # random seed for generation of randomized numbers
        self.random_seed = None
        # standard deviation of positional noise
        self.pos_std = None
        # Start of positional input suppression (m)
        self.pos_input_suppression_start = None
        # End of positional input suppression (m)
        self.pos_input_suppression_end = None

        ## Variable MSMF specific parameters
        # the type of distribution used for generating field sizes (e.g. gamma, normal)
        self.field_size_distribution = None
        # the mean of the normal distribution used for field size generation
        self.fs_normal_mean = None
        # the std of the normal distribution used for field size generation
        self.fs_normal_std = None
        # scaling factor as measuered in the experiments
        self.scaling_factor = None
        # shape parameter for gamma distribution
        self.alpha = None
        # scale parameter for gamma distribution
        self.theta = None
        # the maximum sum of all field sizes of one neuron
        self.max_field_sizes = None
        # maximum number of sampled locations per neuron
        self.s = None
        # whether to generate fields based on bin locs or completely free
        self.gen_fields_at_bin_locs = None
        # whether fields are allowed to overlap or not
        self.allow_field_overlap = None
        # the maximum ratio between field sizes in order to still be in the same attractor
        self.field_ratio_threshold = None
        # probability of connecting different fields
        self.field_connection_prob = None
        # maximum number of tries
        self.max_num_tries = None

        ## Fixed MSMF specific parameters
        # number of maximum sized attractors
        self.nmax = None
        # number of medium sized attractors
        self.nmed = None
        # number of minimum sized attractors
        self.nmin = None
        # fraction of neurons per attractor
        self.f = None
        # a fixed offset for the starting location of the first neuron in each attractor (in bins)
        self.field_loc_offset = None
        # whether to dynamically calculate and set the field offset based on the field size within each attractor
        self.set_field_loc_offset_dyn = None
        # total number of all attractors
        self.nf = None
        # interaction length parameter for neurons within each attractor
        self.interaction_length = None
        # a factor with which the interaction length is multiplied for weight generation and input calculation
        self.interaction_length_factor = None
        # sample neurons for attractor with or without replacement
        self.replace_neuron_attractor = None

        ## Grid MSMF specific parameters
        # number of total modules
        self.num_modules = None
        # number of neurons per module
        self.num_neurons_per_module = None
        # scale of all modules
        self.module_scale = None
        # the scale of the smallest module
        self.module_scale_min = None
        # scale of the individual modules
        self.module_scales = None

        # Load default parameters
        self.load_default_params(can)

        # Set specific parameters loaded from individual configuration
        if params is not None:
            for name, value in params.items():
                setattr(self, name, value)

        if type(can) is Grid:
            self.num_neurons = int(self.num_modules * self.num_neurons_per_module)

    def load_default_params(self, can):
        default_params = load_yaml(PATH_CONFIG, FILE_NAME_DEFAULT_PARAMS_CAN[str(can)])

        for name, value in default_params.items():
            if getattr(self, name) is None:
                setattr(self, name, value)


class ContinuousAttractorNetwork(ABC):
    def __init__(self, index=None, random_seed=None, collected_metrics=None, add_params=None):

        self.p = Parameters(self, add_params)

        self.collected_metrics = collected_metrics if collected_metrics is not None else []
        self.id = index

        self.random_state: RandomState = None
        self.init_random_state(random_seed)

        # Parameters for simulation time/velocity
        self.p.vel = (self.p.env_length / self.p.disc_step) / self.p.T  # speed of movement in bins/second
        self.p.dt = 1 / self.p.vel  # timestep for ode
        self.num_bins = int(self.p.env_length / self.p.disc_step)

        # Create arrays for data storage
        self.pos_estimated = None
        self.pos = None
        self.m = None
        self.t = None
        self.fields_per_pos = None

        self.field_locs = None  # field locations of PCs
        self.field_locs_bins = None  # field location bins of PCs
        self.field_sizes = None  # field sizes of PCs
        self.field_mask = None  # mask, where actual fields are in the arrays above
        self.alive_neuron_ids = None

        # Create array with initial potential for all neurons
        # ToDo: Update random initial potentials to use random_state
        self.init_values = self.p.init_value * self.random_state.random(self.p.num_neurons)

        # Evaluation metrics
        self.mean_field_activities = None
        self.mean_field_activity = None
        self.perc_correct_fields = None
        self.avg_num_fields_per_neuron = None
        self.avg_num_fields_active_per_neuron = None
        self.avg_accum_field_coverage_per_neuron = None
        self.avg_accum_active_field_coverage_per_neuron = None
        self.num_fields_total = None
        self.num_fields_active = None
        self.perc_unique_field_combinations = None
        self.activity_false_positives_num = None
        self.activity_false_negatives_num = None
        self.weighted_avg_activity = None

    def init_random_state(self, random_seed):
        self.p.random_seed_type = 'int' if type(random_seed) is int else random_seed

        if random_seed == RandomSeed.INDEX:
            random_seed = self.id
        elif type(random_seed) is int:
            pass
        elif random_seed == RandomSeed.TIME:
            random_seed = time_ns()
        else:
            random_seed = time_ns()

        self.p.random_seed = random_seed

        log.debug(f'CAN [{self.id}] Random Seed = {random_seed}')

        # noinspection PyTypeChecker
        self.random_state = RandomState(MT19937(SeedSequence(random_seed)))

    def run(self):
        sol = solve_ivp(self.run_cont, [0, self.p.T], self.init_values,
                        t_eval=np.arange(self.p.dt / 2, self.p.T, self.p.dt), method='RK45', args=[self.p])

        if not sol.success:
            log.error(f'Could not find a solution for Network {self.id}')
            raise DiffSolverError(f'Could not find a solution for Network {self.id}')

        self.m = np.clip(sol.y, 0, None)
        self.pos = self.p.vel * sol.t

        # Update ground truth position based on suppression of positional input
        if self.p.pos_input_suppression_start is not None:
            pos_sup = np.argwhere((self.p.pos_input_suppression_start <= self.pos) &
                                  (self.pos <= self.p.pos_input_suppression_end)).flatten()
            if pos_sup[0] > 0:
                self.pos[pos_sup] = self.pos[pos_sup[0] - 1]
            else:
                log.error('Could not execute positional input suppression because the start value is <= 0.')

        self.t = sol.t

        # Calculate metrics
        self.avg_num_fields_per_neuron = self.num_fields_total / self.p.num_neurons

        num_field_bins = np.sum(self.fields_per_pos)
        self.activity_false_positives_num = len(np.where((self.m > 0) & np.logical_not(self.fields_per_pos))[0])
        self.activity_false_negatives_num = (num_field_bins - len(np.where((self.m > 0) & self.fields_per_pos)[0]))

        num_unique_field_combinations = len(np.unique(reduce(lambda a, b: 2 * a + b, self.fields_per_pos)))
        self.perc_unique_field_combinations = num_unique_field_combinations / self.num_bins

        self.weighted_avg_activity = np.mean(self.m) * self.p.num_neurons * self.p.env_length

        if Metric.MEAN_FIELD_ACTIVITY in self.collected_metrics:
            self.mean_field_activity = self.get_mean_field_activity()
            log.debug(f'>>>> Mean field activity = {self.mean_field_activity}')

        if Metric.PERC_CORRECT_FIELDS in self.collected_metrics:
            self.perc_correct_fields = self.get_perc_correct_fields()
            log.debug(f'>>>> Percentage of correct fields = {self.perc_correct_fields}')

    def set_dead_neurons(self, prob_dead_neurons=None):
        # kill neurons
        # - set all weights to zero for specified number of dead neurons
        # - set init potential of these neurons to zero
        if prob_dead_neurons is not None:
            self.p.prob_dead_neurons = prob_dead_neurons

        if self.p.prob_dead_neurons is None or self.p.prob_dead_neurons <= 0:
            self.p.dead_neuron_ids = []
            return

        num_dead_neurons = int(self.p.prob_dead_neurons * self.p.num_neurons)
        dead_neuron_ids = np.transpose(self.random_state.choice(self.p.num_neurons, num_dead_neurons, replace=False))
        self.W[dead_neuron_ids, :] = 0
        self.init_values[dead_neuron_ids] = 0
        self.p.dead_neuron_ids = dead_neuron_ids
        self.alive_neuron_ids = np.arange(self.p.num_neurons)[np.in1d(np.arange(self.p.num_neurons),
                                                                      self.p.dead_neuron_ids).__invert__()]

    @abstractmethod
    def run_cont(self, t, v, p):
        pass

    @abstractmethod
    def init_fields(self):
        pass

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    def get_field_start_bin(self, field_loc, field_size):
        """
        Returns the first bin for the given field with location field_loc and size field_size.
        :param field_loc: The location of a single field or an array of field locations.
        :param field_size: The size of a single field or an array of field sizes.
        :return: Either an array of starting bins or a single starting bin.
        """
        field_start_bin = np.floor(np.round((field_loc - field_size / 2) / self.p.disc_step, 8))
        if type(field_start_bin) is np.ndarray:
            field_start_bin[field_start_bin < 0] = 0
            return np.asarray(field_start_bin, int)
        else:
            if field_start_bin < 0:
                return 0
            else:
                return int(field_start_bin)

    def get_field_end_bin(self, field_loc, field_size):
        """
        Returns the last bin for the given field with location field_loc and size field_size.
        :param field_loc: The location of a single field or an array of field locations.
        :param field_size: The size of a single field or an array of field sizes.
        :return: Either an array of starting bins or a single starting bin.
        """
        field_end_bin = (field_loc + field_size / 2)
        if type(field_end_bin) is np.ndarray:
            field_end_bin = np.where(mod(field_end_bin, self.p.disc_step) == 0,
                                     field_end_bin / self.p.disc_step - 1,
                                     field_end_bin / self.p.disc_step)
            field_end_bin = np.floor(np.round(field_end_bin, 8))
            field_end_bin[field_end_bin < 0] = 0
            field_end_bin[field_end_bin > self.num_bins] = self.num_bins
            return np.asarray(field_end_bin, int)
        else:
            field_end_bin /= self.p.disc_step
            if mod(field_end_bin, 1) == 0:
                field_end_bin -= 1
            field_end_bin = np.floor(np.round(field_end_bin, 8))
            if field_end_bin < 0:
                return 0
            elif field_end_bin > self.num_bins:
                return self.num_bins
            else:
                return int(field_end_bin)

    def get_perc_correct_fields(self):
        if self.p.expected_mean_field_activity is None:
            log.warn('Cannot calculate percentage of correct fields because \'expected_mean_field_activity\' is None.')
            return None

        if self.mean_field_activities is None or len(self.mean_field_activities) <= 0:
            log.warn(
                'Cannot calculate percentage of correct fields because \'mean_field_activities\' is None or empty.')
            return None

        num_total_fields = len(self.mean_field_activities)
        num_correct_fields = np.sum(self.mean_field_activities >= self.p.expected_mean_field_activity)

        if num_total_fields > 0:
            return num_correct_fields / num_total_fields
        else:
            return 0

    def get_activity_removed_background_noise(self):
        # Invert fields_per_pos to get info where there is no field
        no_fields = (self.fields_per_pos ^ (self.fields_per_pos & 1 == self.fields_per_pos))
        # Calculate number of positions where no field occurs
        num_no_fields = np.count_nonzero(no_fields, axis=1)
        # Set number of no-field-positions to 1 where it is zero to prevent division by zero
        num_no_fields[num_no_fields == 0] = 1
        # Calculate background activity
        background_activity = np.sum(self.m * no_fields, axis=1)
        # Calculate mean background activity
        mean_background_activity = background_activity / num_no_fields
        # Return new overall activity
        return self.m - mean_background_activity.reshape((mean_background_activity.shape[0], 1))

    def get_field_locs_sizes_sorted(self, bins=True):
        # Sort fields, so that the colors are representing the id of the fields
        sorted_field_ids = np.argsort(self.field_locs_bins, axis=1)
        if bins:
            field_locs_sorted = np.take_along_axis(self.field_locs_bins, sorted_field_ids, axis=1)
        else:
            field_locs_sorted = np.take_along_axis(self.field_locs, sorted_field_ids, axis=1)
        field_sizes_sorted = np.take_along_axis(self.field_sizes, sorted_field_ids, axis=1)

        return field_locs_sorted, field_sizes_sorted

    def get_mean_field_activity(self):
        self.mean_field_activities = []

        if self.p.mean_field_activity_remove_background:
            activity = self.get_activity_removed_background_noise()
        else:
            activity = self.m

        for i_neuron in range(self.p.num_neurons):
            if self.p.dead_neuron_ids is not None and i_neuron in self.p.dead_neuron_ids:
                log.debug(f'Skipped dead neuron {i_neuron}')
                continue
            for i_field in range(len(self.field_locs_bins[i_neuron])):
                field_size = self.field_sizes[i_neuron, i_field]
                if field_size <= 0:
                    continue
                field_loc = self.field_locs[i_neuron, i_field]
                field_start_bin = self.get_field_start_bin(field_loc, field_size)
                field_end_bin = self.get_field_end_bin(field_loc, field_size)

                mean_field_activity = np.mean(activity[i_neuron, field_start_bin:field_end_bin + 1])

                self.mean_field_activities.append(mean_field_activity)

        self.mean_field_activities = np.array(self.mean_field_activities)

        if len(self.mean_field_activities) > 0:
            return self.mean_field_activities.mean()
        else:
            return 0

    def calc_fields_per_neuron(self):
        """
        Creates a truth matrix (num_neurons, num_bins) for the field locations of all neurons. A 1 indicates, that the
        neuron has a field a that position, a 0 indicates it doesn't.
        :return:
        """
        self.fields_per_pos = np.zeros((self.p.num_neurons, self.num_bins), dtype=int)

        for i_n in range(self.p.num_neurons):
            field_sizes_i = self.field_sizes[i_n, self.field_sizes[i_n, :] > 0]
            field_locs_i = self.field_locs[i_n, self.field_sizes[i_n, :] > 0]

            field_start_bin = self.get_field_start_bin(field_locs_i, field_sizes_i)
            field_end_bin = self.get_field_end_bin(field_locs_i, field_sizes_i)

            for i_field in range(field_start_bin.size):
                self.fields_per_pos[i_n, field_start_bin[i_field]:field_end_bin[i_field] + 1] = 1

    def calc_num_active_fields(self):
        if len(self.p.dead_neuron_ids) == 0:
            self.num_fields_active = self.num_fields_total
        else:
            self.num_fields_active = np.sum(self.field_sizes[self.alive_neuron_ids, :] > 0)
        self.avg_num_fields_active_per_neuron = self.num_fields_active / self.p.num_neurons

    def calc_accum_field_coverage(self):
        # Calculate avg accum field coverage for all neurons
        num_fields = np.sum(self.field_sizes > 0, 1)
        self.avg_accum_field_coverage_per_neuron = np.mean(np.sum(self.field_sizes, 1))
        # Calculate avg accum field coverage only for active neurons (non-dead)
        num_fields_active = np.sum(self.field_sizes[self.alive_neuron_ids] > 0, 1)
        self.avg_accum_active_field_coverage_per_neuron = np.mean(np.sum(self.field_sizes[self.alive_neuron_ids], 1))

        # ToDo: Maybe implement calculation of average field size per neuron
        # Calculate avg field size only for active neurons (non-dead)
        # num_fields_active = np.sum(self.field_sizes[self.alive_neuron_ids] > 0, 1)
        # self.avg_accum_active_field_coverage_per_neuron = np.mean(np.sum(self.field_sizes[self.alive_neuron_ids], 1) /
        #                                                           np.where(num_fields_active > 0, num_fields_active, 1))

    def __str__(self):
        return type(self).__name__


class FMSMF(ContinuousAttractorNetwork):
    def __init__(self, index=None, random_seed=None, collected_metrics=None, add_params=None):
        super().__init__(index=index, random_seed=random_seed, collected_metrics=collected_metrics,
                         add_params=add_params)

        # The number of bins, discretizing the environment
        # Lbins = int(self.p.env_length / self.p.disc_step)
        # The length of all CANs (in bins)
        self.can_lengths = np.zeros(self.p.nmin + self.p.nmed + self.p.nmax)
        if self.p.nmin > 0:
            self.can_lengths[0: self.p.nmin] = (self.p.nmin > 0) * self.num_bins / self.p.nmin
        if self.p.nmed > 0:
            self.can_lengths[self.p.nmin: self.p.nmin + self.p.nmed] = (self.p.nmed > 0) * self.num_bins / self.p.nmed
        if self.p.nmax > 0:
            self.can_lengths[self.p.nmin + self.p.nmed: self.p.nmin + self.p.nmed + self.p.nmax] = \
                (self.p.nmax > 0) * self.num_bins / self.p.nmax
        # Interaction length (one direction, half of field size) between neurons (in bins)
        self.lam = self.p.interaction_length * self.can_lengths

        # Calculate field location offset if enabled
        if self.p.set_field_loc_offset_dyn:
            self.p.field_loc_offset = []
            for n_i in [self.p.nmin, self.p.nmed, self.p.nmax]:
                if n_i > 0:
                    self.p.field_loc_offset.append(self.p.interaction_length * (self.num_bins / n_i))
                else:
                    self.p.field_loc_offset.append(0)

        # Total number of attractors
        self.p.nf = self.p.nmin + self.p.nmed + self.p.nmax

        # The threshold or location of the individual fields
        self.th = np.zeros((int(self.p.num_neurons * self.p.f) + 1, self.p.nf))
        # The index - indicating for each field which neuron is associated with it
        self.ind = np.zeros((int(self.p.num_neurons * self.p.f), self.p.nf),
                            dtype=int)  # indices of neurons per attractor(p.nf)
        # Weight matrix for connections between all neurons
        self.W = None
        # Array for storing the time steps t for all positions
        self.t = None

    def init_fields(self):
        field_loc_offset = self.p.field_loc_offset[0]
        for i in range(self.p.nmin):
            self.ind[:, i] = self.random_state.choice(self.p.num_neurons, size=int(self.p.num_neurons * self.p.f),
                                                      replace=self.p.replace_neuron_attractor)
            self.th[:, i] = np.linspace(field_loc_offset, field_loc_offset + self.can_lengths[i],
                                        num=int(self.p.num_neurons * self.p.f) + 1).transpose()
            field_loc_offset = field_loc_offset + self.can_lengths[i]

        field_loc_offset = self.p.field_loc_offset[1]
        for i in range(self.p.nmin, self.p.nmin + self.p.nmed):
            self.ind[:, i] = self.random_state.choice(self.p.num_neurons, size=int(self.p.num_neurons * self.p.f),
                                                      replace=self.p.replace_neuron_attractor)
            self.th[:, i] = np.linspace(field_loc_offset, field_loc_offset + self.can_lengths[i],
                                        num=int(self.p.num_neurons * self.p.f) + 1).transpose()
            field_loc_offset = field_loc_offset + self.can_lengths[i]

        field_loc_offset = self.p.field_loc_offset[2]
        for i in range(self.p.nmin + self.p.nmed, self.p.nmin + self.p.nmed + self.p.nmax):
            self.ind[:, i] = self.random_state.choice(self.p.num_neurons, size=int(self.p.num_neurons * self.p.f),
                                                      replace=self.p.replace_neuron_attractor)
            self.th[:, i] = np.linspace(field_loc_offset, field_loc_offset + self.can_lengths[i],
                                        num=int(self.p.num_neurons * self.p.f) + 1).transpose()
            field_loc_offset = field_loc_offset + self.can_lengths[i]

        # ToDo: Check if this is still needed
        self.th = self.th[:-1, :]

        self.num_fields_total = self.p.nf * self.p.f * self.p.num_neurons

        # calculate field locs/sizes in generic format
        self.calc_field_locs_sizes_generic()

        # Pre-calculate all fields for each neuron once (needed for decoding and mean/perc num field calculation)
        self.calc_fields_per_neuron()

        # Calculate accumulated field coverage per neuron
        self.calc_accum_field_coverage()

    def init_weights(self):

        self.W = np.zeros((self.p.num_neurons, self.p.num_neurons))

        if self.p.J0 == 0 and self.p.J1 == 0:
            # set dead neurons for lesion simulation
            self.set_dead_neurons()
            # calculate number of active fields
            self.calc_num_active_fields()
            return

        for i in range(self.p.nf):
            locations = [self.th[:, i] for _ in range(int(self.p.num_neurons * self.p.f))]
            dis = np.stack(locations, axis=1) - np.stack(locations, axis=0)
            d = np.abs(dis)
            tmp = np.exp(-d / (self.lam[i] * self.p.interaction_length_factor))
            self.W[np.ix_(self.ind[:, i], self.ind[:, i])] += self.can_lengths[i] / \
                                                              (self.lam[i] * self.p.interaction_length_factor) * \
                                                              (self.p.J1 * tmp + self.p.J0)

        # set dead neurons for lesion simulation
        self.set_dead_neurons()

        # calculate number of active fields
        self.calc_num_active_fields()

        # ToDo: check if this is still necessary
        self.W = self.W / (self.p.num_neurons * self.p.f)

        np.fill_diagonal(self.W, 0)  # zero diagonal

    def run_cont(self, t, v, p):
        Iext = np.zeros(p.num_neurons)
        pos = self.p.vel * t

        if self.p.noise_type == Noise.GAUSSIAN:
            I = np.random.normal(self.p.I, self.p.I_std, Iext.shape)
        else:
            I = self.p.I

        # Calculate positional input only if input suppression is not set or current position is not in specified range
        if self.p.pos_input_suppression_start is None or \
                (self.p.pos_input_suppression_start is not None and
                 not (self.p.pos_input_suppression_start <= pos <= self.p.pos_input_suppression_end)):
            # Add noise to position
            if self.p.pos_std is not None:
                pos = np.abs(self.random_state.normal(pos, self.p.pos_std))

            # Calculate external (positional input)
            d = np.abs(self.th - pos)
            values = self.p.Iloc * np.exp(-d / (self.lam * self.p.interaction_length_factor))
            Iext = np.bincount(self.ind.flatten(), values.flatten())
            Iext.resize(p.num_neurons, refcheck=False)

        currt = np.dot(self.W, np.clip(v, 0, None).transpose()) + I + Iext
        dvdt = (-v + currt) / self.p.tau
        dvdt[p.dead_neuron_ids] = 0
        return dvdt

    def plot(self, n_att=None, pos_range=None, plot_type='circles'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlim(0, self.t.size)
        plt.ylim(0, np.max(self.m))

        pos_range = range(self.t.size - 1) if pos_range is None else pos_range
        n_att = range(self.p.nf) if n_att is None else n_att

        # pos = current position of the agent
        for pos in pos_range:
            # i = attractor index
            for i in n_att:
                # Plot all fields of a neuron as circles or lines according to their size
                if plot_type == 'circles':
                    ax.scatter(self.th[:, i], self.m[self.ind[:, i], pos], c='None', edgecolors=f'C{i}')

            # Plot a line indicating the current position of the agent
            ax.plot([self.pos[pos], self.pos[pos]], [0, np.max(self.m)], label=f'Pos: {self.pos[pos]}', color='red')

            plt.legend(loc='upper right')

            # Draw, wait and clear the previous data from the plot
            plt.draw()
            plt.pause(0.5)
            ax.clear()

    def calc_field_locs_sizes_generic(self):
        self.field_locs = np.zeros((self.p.num_neurons, 1000))
        self.field_locs_bins = np.zeros((self.p.num_neurons, 1000))
        self.field_sizes = np.zeros((self.p.num_neurons, 1000))

        for i_neuron in range(self.p.num_neurons):
            neuron_ind = np.argwhere(self.ind == i_neuron)
            if self.p.dead_neuron_ids is not None and neuron_ind in self.p.dead_neuron_ids:
                log.debug(f'Skipped dead neuron {neuron_ind}')
                continue

            self.field_locs_bins[i_neuron, :neuron_ind.shape[0]] = self.th[neuron_ind[:, 0], neuron_ind[:, 1]]
            # multiply size with 2 because original size is only interaction length (
            self.field_sizes[i_neuron, :neuron_ind.shape[0]] = self.lam[neuron_ind[:, 1]] * 2 * self.p.disc_step

        # remove zero columns in field locs and field sizes
        self.field_locs_bins = self.field_locs_bins[:, ~((self.field_sizes == 0).all(0))]
        self.field_sizes = self.field_sizes[:, ~((self.field_sizes == 0).all(0))]

        self.field_locs = self.field_locs_bins * self.p.disc_step
        self.field_mask = self.field_sizes > 0


class SSSF(FMSMF):
    def __init__(self, index=None, random_seed=None, collected_metrics=None, add_params=None):
        super().__init__(index=index, random_seed=random_seed, collected_metrics=collected_metrics,
                         add_params=add_params)


class VariableCAN(ContinuousAttractorNetwork, ABC):
    def __init__(self, index=None, random_seed=None, collected_metrics=None, add_params=None):
        super().__init__(index=index, random_seed=random_seed, collected_metrics=collected_metrics,
                         add_params=add_params)

        self.mean_field_activities = None
        self.W = None
        self.t = None

    def run_cont(self, t, v, p):
        Iext = np.zeros(p.num_neurons)
        pos = self.p.vel * t

        if self.p.noise_type == Noise.GAUSSIAN:
            I = np.random.normal(self.p.I, self.p.I_std, Iext.shape)
        else:
            I = self.p.I

        # Calculate positional input only if input suppression is not set or current position is not in specified range
        if self.p.pos_input_suppression_start is None or \
                (self.p.pos_input_suppression_start is not None and
                 not (self.p.pos_input_suppression_start <= pos <= self.p.pos_input_suppression_end)):
            # Add noise to position
            if self.p.pos_std is not None:
                pos = np.abs(self.random_state.normal(pos, self.p.pos_std))

            # Calculate external (positional) input
            distances_tmp = abs(self.field_locs - (pos * self.p.disc_step))
            distances_tmp[self.field_sizes <= 0] = 0
            Iext = self.p.Iloc * np.sum(np.where(self.field_sizes <= 0, 0,
                                                 np.exp(-distances_tmp / np.where(self.field_mask,
                                                                                  (self.field_sizes / 2),
                                                                                  1))), 1)

        currt = np.dot(self.W, np.clip(v, 0, None).transpose()) + I + Iext
        dvdt = (-v + currt) / self.p.tau
        dvdt[p.dead_neuron_ids] = 0
        return dvdt

    def plot(self, pos_range=None, plot_type='circles'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlim(0, self.t.size)
        plt.ylim(0, np.max(self.m))

        pos_range = range(self.t.size - 1) if pos_range is None else pos_range

        # pos = current position of the agent
        for pos in pos_range:
            # i = neuron index
            for i in range(self.p.num_neurons):
                # k = field index for neuron i
                for k in range(self.field_locs.shape[1]):
                    # Check if a field k with size > 0 exists for neuron i
                    if self.field_sizes[i, k] == 0:
                        continue
                    # Plot all fields of a neuron as circles or lines according to their size
                    if plot_type == 'circles':
                        ax.scatter(self.field_locs[i, k], self.m[i, pos], c='None',
                                   edgecolors=f'C{i}')
                    elif plot_type == 'fields':
                        x_a = self.field_locs[i, k] - self.field_sizes[i, k] / 2
                        x_b = self.field_locs[i, k] + self.field_sizes[i, k] / 2
                        ax.plot([x_a, x_b], [self.m[i, pos], self.m[i, pos]], c=f'C{i}', linewidth=3)
            # Plot a line indicating the current position of the agent
            ax.plot([self.pos[pos], self.pos[pos]], [0, np.max(self.m)], label=f'Pos: {self.pos[pos]}', color='red')

            plt.legend(loc='upper right')

            # Draw, wait and clear the previous data from the plot
            plt.draw()
            plt.pause(0.5)
            ax.clear()


class DMSMF(VariableCAN):
    def __init__(self, index=None, random_seed=None, collected_metrics=None, add_params=None):
        super().__init__(index=index, random_seed=random_seed, collected_metrics=collected_metrics,
                         add_params=add_params)

        if self.p.field_size_distribution == Distribution.GAMMA:
            # Calculate a scale parameter for the gamma distribution if not set
            if self.p.theta is None:
                self.p.theta = 1.8 * np.power((self.p.env_length / 200), self.p.scaling_factor)
            self.p.fs_normal_mean = None
            self.p.fs_normal_std = None
        else:
            self.p.theta = None
            self.p.alpha = None

        # Calculate the maximum sum of all field sizes of one neuron if not set
        if self.p.max_field_sizes is None:
            self.p.max_field_sizes = 0.15 * self.p.env_length * np.power((200 / self.p.env_length),
                                                                         self.p.scaling_factor)

        self.num_field_con_pruned = 0
        self.num_field_con_total = 0
        self.ignored_field_connections_ratio = 0

    def init_fields(self):

        self.field_locs_bins = np.zeros((self.p.num_neurons, self.p.s), dtype=int)  # field locations for PCs
        self.field_locs = np.zeros((self.p.num_neurons, self.p.s))  # field locations for PCs
        self.field_sizes = np.zeros((self.p.num_neurons, self.p.s))  # field sizes of PCs

        # generate multiple randomly sized fields with random sizes for each neuron,
        # according to the theoretical analysis for scheme 6
        num_field_num_limit_reached = 0
        for i in range(self.p.num_neurons):
            k = 0
            num_tries = 0

            # generate a new field for neuron i if overall field sum doesn't exceed threshold
            while num_tries < self.p.max_num_tries:
                # Generate field size based on defined distribution
                if self.p.field_size_distribution == Distribution.GAMMA:
                    field_size = self.random_state.gamma(self.p.alpha, self.p.theta)
                elif self.p.field_size_distribution == Distribution.NORMAL:
                    field_size = self.random_state.normal(self.p.fs_normal_mean, self.p.fs_normal_std)
                    while field_size <= 0:
                        field_size = self.random_state.normal(self.p.fs_normal_mean, self.p.fs_normal_std)
                        num_tries += 1
                else:
                    log.error(f'Cannot generate field size. Field size distribution "{self.p.field_size_distribution}" '
                              f'not implemented yet!')

                if self.p.gen_fields_at_bin_locs:
                    field_loc_id = self.random_state.randint(self.p.env_length / self.p.disc_step) + 0.5
                    field_loc = field_loc_id * self.p.disc_step
                else:
                    field_loc = self.random_state.random() * (self.p.env_length - field_size) + field_size / 2
                    field_loc_id = field_loc / self.p.disc_step
                fields_overlap = False

                # check if threshold was reached, if so: continue loop for another try
                if np.sum(self.field_sizes[i, :]) + field_size >= self.p.max_field_sizes:
                    num_tries += 1
                    continue

                # check whether fields are overlapping in case the parameter is set to False
                if not self.p.allow_field_overlap:
                    for l in range(self.field_locs_bins.shape[1]):
                        # if the field size is 0 then the end of the existing fields has been reached
                        if self.field_sizes[i, l] == 0:
                            break

                        tmp_field_size = self.field_sizes[i, l]
                        tmp_field_loc = self.field_locs_bins[i, l] * self.p.disc_step

                        # check if fields overlap, if they do then increase number of tries and leave
                        if not (field_loc - field_size / 2 > tmp_field_loc + tmp_field_size / 2 or
                                field_loc + field_size / 2 < tmp_field_loc - tmp_field_size / 2):
                            num_tries += 1
                            fields_overlap = True
                            break

                # add new field to the neurons fields if it doesn't overlap with its other fields
                if not fields_overlap and k < self.p.s:
                    self.field_locs_bins[i, k] = field_loc_id
                    self.field_locs[i, k] = field_loc
                    self.field_sizes[i, k] = field_size
                    k = k + 1
                elif k >= self.p.s:
                    num_field_num_limit_reached += 1
                    break

        if num_field_num_limit_reached > 0:
            log.warning(f'Reached field number limit for {num_field_num_limit_reached} neurons.')

        # check if fields were generated, throw error if not
        if k == 0:
            raise EmptyFields('Could not generate any fields.')

        # remove zero columns in field locs and field sizes
        self.field_locs_bins = self.field_locs_bins[:, ~((self.field_sizes == 0).all(0))]
        self.field_locs = self.field_locs[:, ~((self.field_sizes == 0).all(0))]
        self.field_sizes = self.field_sizes[:, ~((self.field_sizes == 0).all(0))]

        self.field_mask = self.field_sizes > 0

        self.num_fields_total = np.sum(self.field_sizes > 0)

        # Pre-calculate all fields for each neuron once (needed for decoding and mean/perc num field calculation)
        self.calc_fields_per_neuron()

        # Calculate accumulated field coverage per neuron
        self.calc_accum_field_coverage()

    def init_weights(self):

        self.W = np.zeros((self.p.num_neurons, self.p.num_neurons))
        num_fields = np.shape(self.field_locs_bins)[1]
        self.num_field_con_pruned = 0
        self.num_field_con_total = 0

        if self.p.J0 == 0 and self.p.J1 == 0:
            # set dead neurons for lesion simulation
            self.set_dead_neurons()
            # calculate number of active fields
            self.calc_num_active_fields()
            return

        if self.p.field_ratio_threshold is None and self.p.field_connection_prob is None:
            log.debug('Neither the \'field_ratio_threshold\' nor the \'field_connection_prob\' are set. '
                      'All connections will be created.')

        for i in range(self.p.num_neurons):
            w_ij = 0
            for j in range(self.p.num_neurons):
                # ToDo: Check how to deal with self-recurrent weights, whether they are needed or not
                if i == j:
                    #             self.W(i, j) = self.p.J1 + self.p.J0
                    continue
                for k in range(num_fields):
                    # continue with next neuron (j+1) if end of self.fields reached
                    if self.field_locs_bins[i, k] == 0:
                        break
                    for l in range(num_fields):
                        # continue with next field (k+1) if end of fields reached
                        if self.field_locs_bins[j, l] == 0:
                            break
                        self.num_field_con_total += 1
                        # check if field sizes are approximately the same
                        # if not, set weight to 0
                        if self.p.field_ratio_threshold is not None:
                            if min([self.field_sizes[i, k], self.field_sizes[j, l]]) / max(
                                    [self.field_sizes[i, k], self.field_sizes[j, l]]) < self.p.field_ratio_threshold:
                                self.num_field_con_pruned += 1
                                continue
                        elif self.p.field_connection_prob is not None:
                            if self.random_state.rand() < self.p.field_connection_prob:
                                self.num_field_con_pruned += 1
                                continue
                        distance = abs(self.field_locs_bins[i, k] - self.field_locs_bins[j, l]) * self.p.disc_step
                        int_radius = self.field_sizes[i, k] / 2
                        w_ij = w_ij + (self.p.J1 * np.exp(-distance / int_radius) + self.p.J0)
                self.W[i, j] = w_ij

        # Calculate field connection statistics
        if self.num_field_con_total > 0:
            self.ignored_field_connections_ratio = self.num_field_con_pruned / self.num_field_con_total

        # set dead neurons for lesion simulation
        self.set_dead_neurons()

        # calculate number of active fields
        self.calc_num_active_fields()

        # ToDo: check if this is still necessary
        self.W = self.W / self.p.num_neurons

        np.fill_diagonal(self.W, 0)  # zero diagonal


class Grid(VariableCAN):
    def __init__(self, index=None, random_seed=None, collected_metrics=None, add_params=None):
        super().__init__(index=index, random_seed=random_seed, collected_metrics=collected_metrics,
                         add_params=add_params)

        # Set module scales automatically of they are not set manually
        if self.p.module_scales is None:
            if self.p.module_scale is None or self.p.module_scale_min is None:
                log.error('Cannot calculate module scales because the module scale is not set.')
                return
            self.module_scales = [self.p.module_scale_min]
            for i_module in range(self.p.num_modules - 1):
                self.module_scales.append(self.module_scales[i_module] * self.p.module_scale)
            log.debug(f'Automatically calculated module scales = {self.module_scales}')
        else:
            self.module_scales = self.p.module_scales

    def init_fields(self):

        # sort scales descending
        self.module_scales = np.sort(self.module_scales)
        field_locs = []
        i_n = 0

        # ToDo: Check if performance here can be improved by using np matrix operations and e.g. linspace
        # Set location of fields based on the number of neurons per module and their scales
        for i_mod in range(self.p.num_modules):
            scale = self.module_scales[i_mod]

            for i_nm in range(self.p.num_neurons_per_module):
                field_locs_n = []
                start_loc = 0.5 * scale + i_nm * scale
                for field_loc in np.arange(start_loc, self.p.env_length, scale * self.p.num_neurons_per_module):
                    # add id of field location within array (actual location / discretization step)
                    field_locs_n.append(field_loc)

                # Check if current number of fields is smaller than the largest number of fields (neuron 0)
                if len(field_locs) > 0:
                    while len(field_locs_n) < len(field_locs[0]):
                        # Fill the rest with -1
                        field_locs_n.append(-1)

                field_locs.append(field_locs_n)
                i_n += 1

        self.field_locs = np.array(field_locs)  # field locations for PCs
        self.field_locs_bins = np.where(self.field_locs == -1, -1, self.field_locs / self.p.disc_step).astype(int)
        self.field_sizes = np.zeros(self.field_locs_bins.shape)  # field sizes of PCs

        # Set field sizes for all fields of a neurons in a module
        for i_mod in range(self.p.num_modules):
            start_neuron = i_mod * self.p.num_neurons_per_module
            end_neuron = i_mod * self.p.num_neurons_per_module + self.p.num_neurons_per_module
            self.field_sizes[start_neuron:end_neuron] = self.module_scales[i_mod]

        # set field sizes to 0, where field loc is -1 (i.e. no field)
        self.field_sizes[self.field_locs_bins == -1] = 0
        # Set field mask where fields exist
        self.field_mask = self.field_sizes > 0

        self.num_fields_total = np.sum(self.field_sizes > 0)

        # Pre-calculate all fields for each neuron once (needed for decoding and mean/perc num field calculation)
        self.calc_fields_per_neuron()

        # Calculate accumulated field coverage per neuron
        self.calc_accum_field_coverage()

    def init_weights(self):

        self.W = np.zeros((self.p.num_neurons, self.p.num_neurons))

        if self.p.J0 == 0 and self.p.J1 == 0:
            # set dead neurons for lesion simulation
            self.set_dead_neurons()
            # calculate number of active fields
            self.calc_num_active_fields()
            return

        # connect all neurons within a module
        for i_mod in range(self.p.num_modules):
            offset = i_mod * self.p.num_neurons_per_module
            for i_nm in range(offset, offset + self.p.num_neurons_per_module):
                for j_nm in range(offset, offset + self.p.num_neurons_per_module):
                    if i_nm == j_nm:
                        continue
                    # ToDo: Check if only one connection should be modeled (shortest over edge) or both (in both dir.)
                    # calculate shortest distance between neurons (over edge)
                    distance = self.module_scales[i_mod] * \
                               (min(abs(i_nm - j_nm),
                                    abs(min(i_nm, j_nm) - (max(i_nm, j_nm) - self.p.num_neurons_per_module))))
                    # ToDo: Maybe add a maximum distance for weight calculation between two neurons
                    int_radius = self.module_scales[i_mod] * 1.0
                    self.W[i_nm, j_nm] += self.p.J1 * np.exp(-distance / int_radius) + self.p.J0

        # set dead neurons for lesion simulation
        self.set_dead_neurons()

        # calculate number of active fields
        self.calc_num_active_fields()

        # ToDo: check if this is still necessary
        # self.W = self.W / self.p.num_neurons

        np.fill_diagonal(self.W, 0)  # zero diagonal
