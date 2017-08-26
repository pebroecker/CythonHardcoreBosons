import h5py
import numpy as np
from copy import copy

class observable:
    name = ""
    count = 0
    timeseries = np.zeros((1))
    average = 0.
    binning_error = 0.


def add_value(obs, val):
    if len(obs.timeseries) <= obs.count:
        new_timeseries = np.zeros((2 * len(obs.timeseries)))
        new_timeseries[0:obs.count] = obs.timeseries[:]
        obs.timeseries = copy(new_timeseries)

    obs.timeseries[obs.count] = val
    obs.count += 1


def obs2hdf5(filename, obs, timeseries=False):
    h5file = h5py.File(filename, "a")
    if not "simulation" in h5file.keys():
        h5file.create_group("simulation")

    if not "results" in h5file["simulation"].keys():
        h5file["simulation"].create_group("results")

    h5file["simulation/results/" + obs.name + "/count"] = obs.count
    h5file["simulation/results/" + obs.name + "/mean"] = np.mean(obs.timeseries[0:obs.count - 1])
    if timeseries == True:
        h5file["simulation/results/" + obs.name + "/timeseries"] = obs.timeseries[0:obs.count - 1]

    h5file.close()
