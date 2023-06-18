from scipy.integrate import solve_ivp
from scipy.io import loadmat
import numpy as np

from time import time


class P:
    field_locs = None
    field_sizes = None
    disc_step = None
    Iloc = None
    W = None
    I = None
    v = None
    tau = None
    T = None
    dt = None
    n = None


def run_cont(t, v, p):
    Iext = np.zeros(p.n)
    pos = p.v * t

    for i in range(p.n):
        field_locs_i = p.field_locs_bins[i, p.field_locs_bins[i, :] != 0]
        field_sizes_i = p.field_sizes[i, p.field_sizes[i, :] != 0]
        distances = abs(field_locs_i - pos) * p.disc_step
        Iext[i] += p.Iloc * np.sum(np.exp(-distances / (field_sizes_i / 2)))

    currt = np.dot(p.W, np.clip(v, 0, None).transpose()) + p.I + Iext
    dvdt = (-v + currt) / p.tau
    return dvdt
    # dvdt[p.dead_neuron_ids] = 0


mat = loadmat('../matlab/msmf-models/eval_data/ode_test.mat')
# init_values = np.random.rand(1, p.n) * 0.01
init_values = mat['init_values']
init_values = init_values.reshape(init_values.shape[1])

p = P()
members = [attr for attr in dir(p) if not callable(getattr(p, attr)) and not attr.startswith("__")]

for member in members:
    setattr(p, member, mat[member])

p.n = int(p.n)

start = time()

sol = solve_ivp(run_cont, [0, p.T], init_values, t_eval=np.arange(0, p.T, p.dt), args=[p])

end = time()

print(f'This took {end-start} seconds')

# print(sol.t)
# print(sol.y[])


