from msmfcode.core.logging import log
from msmfcode.core.config import *
from msmfcode.execution.parallel import ParallelExecutor, ParallelEvaluationExecutor
from msmfcode.models.cann import MSMFSingleCAN, MSMFMultiCAN, MSMFGridCAN, SSSFCAN
from msmfcode.evaluation.plot import plot_weight_distribution, plot_neuron_activity_fixed_attractors, \
    plot_neuron_activity_variable_attractors, plot_neuron_activity_static, save_plot
import matplotlib.pyplot as plt

# Configuration
PLOT_FILE_TYPE = 'png'
log.handlers[LogHandler.STREAM].setLevel(logging.INFO)
log.handlers[LogHandler.FILE].setLevel(logging.DETAIL)
experiment_id = 'single'

pe = ParallelExecutor(experiment_id, MSMFSingleCAN)

# for i_can in range(len(pe.cans)):
#     pe.cans[i_can].p.prob_dead_neurons = 0.3

pe.run()

# pe.experiment_num = 1
# pe.save_data(save_model=True)

# plot_neuron_activity_static(pe.cans[0], neuron_range=[0,1,2,3,4])
# plot_neuron_activity_static(pe.cans[0], neuron_range=range(pe.cans[0].p.num_neurons))

# pe.cans[0].m_cont = np.array(pe.cans[0].m_cont)
# pe.cans[0].Iext_cont = np.array(pe.cans[0].Iext_cont)
# pe.cans[0].pos_cont = np.array(pe.cans[0].pos_cont)

# plt.plot(pe.cans[0].pos_cont, pe.cans[0].m_cont, 'x')
# plt.show()

pass
# plot_neuron_activity_fixed_attractors_static(pe.cans[0], neuron_range=[0,1,2,3,4])