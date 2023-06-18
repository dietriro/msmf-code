import numpy as np
import matplotlib.pyplot as plt

from msmfcode.core.logging import log
from msmfcode.models.cann import ContinuousAttractorNetwork, MSMFMultiCAN, VariableCAN

ROW = 0
COL = 1


def pop_decoding(net: ContinuousAttractorNetwork, plot_result: bool, decoding_threshold=None):
    """

    :param decoding_threshold:
    :param net:
    :param plot_result:
    :return:
    """
    estimated_positions = np.zeros(net.pos.size)
    m = np.copy(net.m)
    if decoding_threshold is not None:
        m[m < decoding_threshold] = 0

    for pos in range(net.pos.size):
        activity = np.dot(m[:, pos], net.fields_per_pos)

        activity_max = np.max(activity)
        if activity_max > 0:
            pos_estimate = np.mean(net.pos[activity == activity_max])
        else:
            pos_estimate = pos + net.pos.size / 2
        estimated_positions[pos] = pos_estimate

    return estimated_positions


# Plot positional error for place cell models
def plot_loc_error(net, estimated_positions, eval_data, plot_decay=0, data_id=-1):
    if len(eval_data.error_abs) <= 0:
        return

    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(3, 1, figsize=(12, 8))

    # For Sine Function
    axis[0].plot(net.pos, net.pos)
    axis[0].set_ylabel("Position (m)")

    # For Cosine Function
    axis[1].plot(net.pos, estimated_positions)
    axis[1].set_ylabel("Decoded Position (m)")

    # For Tangent Function
    axis[2].plot(net.pos, eval_data.error_abs[data_id])
    axis[2].set_ylabel("Abs. Error (m)")

    axis[2].set_xlabel("Position of the agent (m)")

    # Combine all the operations and display
    if plot_decay > 0:
        plt.draw()
        plt.pause(plot_decay)
    else:
        plt.show()
