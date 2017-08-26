#!python
#cython: language_level=2, boundscheck=False, nonecheck=False, wraparound=False, initializedcheck=False, cdivision=True
#
#overflowcheck.fold=False, wraparound=False, initializedcheck=False, initializedcheck=False
cimport cython
import observable as obs
import xml_parameters
import sys
import os
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport floor
import h5py
from copy import copy

import numpy as np
import numpy.random as npr
cimport numpy as np


cdef inline double rnd():
    return float(rand()) / RAND_MAX


cdef:
    unsigned long L
    unsigned long vol
    unsigned long n_bonds
    double beta
    unsigned long expansion_order
    unsigned long curr_expansion_order
    double Delta
    double h
    double eps
    double H_c
    double fe_up_fe_up
    double fe_up_od
    double up_af_af
    double up_af_od
    double up_od_od
    double od_fe_up
    double fe_dn_fe_dn
    double fe_dn_od
    double dn_af_af
    double dn_af_od
    double dn_od_od
    double od_fe_dn
    long[:, ::1] operator_string
    long[:, :, ::1] operator_string_obs
    long[::1] windings_x
    long[::1] windings_y
    long[::1] worldline_sites
    long[::1] spins
    long[::1] leg_spins
    long[::1] legs
    double[:, ::1] even_odd
    long exit_op
    long exit_leg
    int curr_conf
    long[:, :, ::1] loop_instructions
    long[:, :, ::1] loop_ops
    long[:, :, ::1] spm_corr
    long[:, :, ::1] sz_corr
    long[:, :, :, ::1] loop_img
    long[:, :, :, ::1] single_loop_img
    long[:, ::1] op_stats
    long[:, ::1] spin_imgs
    long[:, :, ::1] windings_x_obs
    long[:, :, ::1] windings_y_obs
    float[:, :, ::1] norm_windings_x_obs
    float[:, :, ::1] norm_windings_y_obs
    long[:, :, :, ::1] worldline_imgs
    float[:, :, :, ::1] worldline_wind_imgs
    float[:, :, :, ::1] windings_x_imgs
    float[:, :, :, ::1] windings_y_imgs
    long[::1] initial_sites
    long[::1] rnd_bonds
    long[::1] rnd_a
    long[::1] rnd_b

cpdef void main(argv):
    global L, vol, n_bonds, beta, expansion_order, curr_expansion_order, Delta, h, eps, H_c
    global fe_up_fe_up, fe_up_od, up_af_af, up_af_od, up_od_od, od_fe_up, fe_dn_fe_dn, fe_dn_od, dn_af_af, dn_af_od, dn_od_od, od_fe_dn
    global spins, operator_string, operator_string_obs, leg_spins, legs, windings_x, windings_y, worldline_sites
    global even_odd, windings_x_obs, windings_y_obs, norm_windings_x_obs, norm_windings_y_obs, windings_x_imgs, windings_y_imgs
    global curr_conf, loop_instructions, loop_ops, sz_corr, spm_corr, loop_img, worldline_imgs, worldline_wind_imgs, single_loop_img, spin_imgs, worldine_imgs, op_stats, initial_sites, rnd_bonds, rnd_a, rnd_b

    cdef long thermalizations, sweeps, sweeps_per_meas
    cdef int rethermalize
    cdef long i, j, k, l, idx
    cdef long fix_expansion_order, old_expansion_order
    cdef long n_od_ops
    cdef double h_fe_up, h_fe_down, h_od, h_af, up_up_norm, dn_dn_norm

    cdef long[:] signs = np.zeros([128], dtype=np.int64)
    cdef long[:, ::1] old_op_string
    cdef long[::1] old_legs
    cdef long[::1] old_leg_spins

    prefix = str(argv[1])
    idx = int(argv[2])
    output_filename = prefix + ".task{i}.out.h5".format(i=idx)

    if os.path.exists(output_filename):
        print("HDF5 file", output_filename, "exists")
        raise

    p = {}
    params = xml_parameters.xml2parameters(p, prefix, idx)
    xml_parameters.parameters2hdf5(params, output_filename)

    srand(np.int64(params["SEED"]))
    print("Initial random value ", rnd())

    L = np.uint(params["L"])
    vol = L * L
    n_bonds = 2 * vol
    beta = float(params["BETA"])
    thermalizations = np.uint(params["THERMALIZATION"])
    sweeps = np.uint(params["SWEEPS"])
    sweeps_per_meas = 1
    spins = npr.choice([np.int64(-1), np.int64(1)], size=(vol))
    expansion_order = L * L
    curr_expansion_order = 0
    even_odd = np.ones((L, L), dtype=np.float)

    for i in np.arange(0, L, 2, dtype=int):
        for j in np.arange(0, L, 2, dtype=int):
            even_odd[i, j] *= -1

    for i in np.arange(1, L, 2, dtype=int):
        for j in np.arange(1, L, 2, dtype=int):
            even_odd[i, j] *= -1

    if "EXPANSION_ORDER" in params.keys():
        expansion_order = np.uint(params["EXPANSION_ORDER"])
        fix_expansion_order = 1
    else:
        fix_expansion_order = 0

    if "RETHERMALIZE" in params.keys():
        rethermalize = 1
        print("Rethermalization activated")
    else:
        print("Could not find rethermalization")
        rethermalize = 0
        
    windings_x = np.zeros((vol), dtype=np.int64)
    windings_y = np.zeros((vol), dtype=np.int64)
    windings_x_obs = np.zeros((128, L, L), dtype=np.int64)
    windings_y_obs = np.zeros((128, L, L), dtype=np.int64)
    norm_windings_x_obs = np.zeros((128, L, L), dtype=np.float32)
    norm_windings_y_obs = np.zeros((128, L, L), dtype=np.float32)
    worldline_sites = np.zeros((vol), dtype=np.int64)
    loop_instructions = np.zeros((128, 1000, 3), dtype=np.int64)
    loop_ops = np.zeros((128, 1000, 7), dtype=np.int64)
    operator_string = np.zeros((expansion_order, 4), dtype=np.int64)
    operator_string_obs = np.zeros((128, 1000, 3), dtype=np.int64)
    legs = np.zeros((2 * vol + 4 * expansion_order), dtype=np.int64)
    leg_spins = np.zeros((2 * vol + 4 * expansion_order), dtype=np.int64)
    sz_corr = np.zeros((128, vol, vol), dtype=np.int64)
    spm_corr = np.zeros((128, vol, vol), dtype=np.int64)
    loop_img = np.zeros((128, L, L, 200), dtype=np.int64)
    single_loop_img = np.zeros((128, L, L, 200), dtype=np.int64)
    op_stats = np.zeros((3, 1), dtype=np.int64)
    spin_imgs = np.zeros((128, vol), dtype=np.int64)
    worldline_imgs = np.zeros((128, 512, L, L), dtype=np.int64)
    worldline_wind_imgs = np.zeros((128, 512, L, L), dtype=np.float32)
    windings_x_imgs = np.zeros((128, 512, L, L), dtype=np.float32)
    windings_y_imgs = np.zeros((128, 512, L, L), dtype=np.float32)
    single_worldline_imgs = np.zeros((128, 512, L, L), dtype=np.int64)
    initial_sites = np.zeros((vol), dtype=np.int64)
    rnd_a = np.zeros((vol), dtype=np.int64)
    rnd_b = np.zeros((vol), dtype=np.int64)
    rnd_bonds = np.zeros((2 * vol), dtype=np.int64)

    for i in range(vol):
        leg_spins[i * 2] = spins[i]
        leg_spins[i * 2 + 1] = spins[i]


    Delta = float(params["Delta"])
    h = float(params["h"])
    eps = float(params["EPS"])

    if h > 0:
        H_c = eps + Delta / 4. + h / 4.
    else:
        H_c = eps + Delta / 4. - h / 4.

    h_fe_up = H_c - Delta / 4 - h / 4.
    h_fe_dn = H_c - Delta / 4. + h / 4.
    h_af = H_c + Delta / 4

    h_od = 1.0

    up_up_norm = h_fe_up + h_af + h_od
    dn_dn_norm = h_fe_dn + h_af + h_od

    fe_up_fe_up = h_fe_up / up_up_norm
    fe_up_od = (h_fe_up + h_af) / up_up_norm
    up_af_af = h_af / up_up_norm
    up_af_od = (h_af + h_fe_up) / up_up_norm
    up_od_od = h_od / up_up_norm
    od_fe_up = (h_od + h_af) / up_up_norm

    fe_dn_fe_dn = h_fe_dn / dn_dn_norm
    fe_dn_od = (h_fe_dn + h_af) / dn_dn_norm
    dn_af_af = h_af / dn_dn_norm
    dn_af_od = (h_af + h_fe_dn) / dn_dn_norm
    dn_od_od = h_od / dn_dn_norm
    od_fe_dn = (h_od + h_af) / dn_dn_norm

    print(fe_up_fe_up, fe_up_od, up_af_af, up_af_od, up_od_od, od_fe_up)
    print(fe_dn_fe_dn, fe_dn_od, dn_af_af, dn_af_od, dn_od_od, od_fe_dn)
    print("mu is ", h + 2 * Delta)
    
    obs_exp_order = obs.observable()
    obs_exp_order.name = "Expansion Order"
    obs_energy, obs_energy_2, obs_magnetization, obs_magnetization_2 = obs.observable(), obs.observable(), obs.observable(), obs.observable()
    obs_staggered_magnetization = obs.observable()
    obs_staggered_magnetization_2 = obs.observable()
    obs_density = obs.observable()
    obs_energy.name = "Energy"
    obs_energy_2.name = "Energy2"
    obs_magnetization.name = "Magnetization"
    obs_magnetization_2.name = "Magnetization2"
    obs_staggered_magnetization.name = "Staggered Magnetization"
    obs_staggered_magnetization_2.name = "Staggered Magnetization2"
    obs_density.name = "Density"

    print("Let's go", thermalizations, sweeps)
    print(sz_corr.shape)

    curr_conf = 0

    for i in range(thermalizations):
        diagonal_update()
        off_diagonal_update()

        if fix_expansion_order == True: continue

        old_expansion_order = expansion_order

        if curr_expansion_order >= expansion_order - 1:
            print("Growing expansion order by 10 percent ", int(np.ceil(expansion_order * 1.1)))
            old_op_string = operator_string.copy()
            old_legs = legs.copy()
            old_leg_spins = leg_spins.copy()

            expansion_order = int(np.ceil(expansion_order * 1.1))
            operator_string = np.zeros((expansion_order, 4), dtype=np.int64)
            legs = np.zeros((2 * vol + 4 * expansion_order), dtype=np.int64)
            leg_spins = np.zeros((2 * vol + 4 * expansion_order), dtype=np.int64)

            for k in range(old_expansion_order):
                for l in [0, 1, 2, 3]:
                    operator_string[k, l] = old_op_string[k, l]

            legs[:len(old_legs)] = old_legs[:]
            leg_spins[:len(old_legs)] = old_leg_spins[:]

    print("Let's measure", thermalizations, sweeps)
    sz_corr[...] = 0
    spm_corr[...] = 0
    spin_imgs[...] = 0
    worldline_imgs[...] = 0
    worldline_wind_imgs[...] = 0
    windings_x_imgs[...] = 0
    windings_y_imgs[...] = 0
    loop_img[...] = 0
    single_loop_img[...] = 0
    op_stats[...] = 0
    operator_string_obs[...] = 0
    loop_instructions[...] = 0
    loop_ops[...] = 0

    print(sz_corr.shape)

    for i in range(sweeps):
        diagonal_update()
        off_diagonal_update()

        if i % sweeps_per_meas == 0:
            sign = 1

            n_od_ops = 0

            for j in range(expansion_order):
                if operator_string[j, 0] == 3:
                    n_od_ops += 1

            signs[curr_conf] = 0 if n_od_ops % 2 == 0 else 1
            if signs[curr_conf] == -1: print("This model does not have a sign problem"); raise

            curr_conf += 1

            energy, magnetization, staggered_magnetization, density = measurements()
            obs.add_value(obs_energy, energy)
            obs.add_value(obs_energy_2, energy**2)
            obs.add_value(obs_magnetization, magnetization)
            obs.add_value(obs_magnetization_2, magnetization**2)
            obs.add_value(obs_staggered_magnetization, staggered_magnetization)
            obs.add_value(obs_staggered_magnetization_2, staggered_magnetization**2)
            obs.add_value(obs_density, density)
            obs.add_value(obs_exp_order, curr_expansion_order)

        if curr_conf == 128:
            print("Dumping data")
            h5f = h5py.File(output_filename, "a")
            if not "simulation" in h5f.keys():
                h5f.create_group("simulation")

            if not "results" in h5f["simulation"].keys():
                h5f["simulation"].create_group("results")

            if not "sz_corr" in h5f["simulation/results"].keys():
                h5f["simulation/results"].create_group("sz_corr")

            if not "spm_corr" in h5f["simulation/results"].keys():
                h5f["simulation/results"].create_group("spm_corr")

            if not "loop_img" in h5f["simulation/results"].keys():
                h5f["simulation/results"].create_group("loop_img")

            if not "single_loop_img" in h5f["simulation/results"].keys():
                h5f["simulation/results"].create_group("single_loop_img")

            if not "spin_imgs" in h5f["simulation/results"].keys():
                h5f["simulation/results"].create_group("spin_imgs")

            if not "worldline_imgs" in h5f["simulation/results"].keys():
                h5f["simulation/results"].create_group("worldline_imgs")

            key = 1
            test_k = "{k}".format(k=key)
            while test_k in h5f["simulation/results/spin_imgs"].keys():
                key += 1
                test_k = "{k}".format(k=key)
            new_key = "{k}".format(k=key)

            dset_loop_instructions = h5f.create_dataset("simulation/results/loop_instructions/" + new_key, (128, 1000, 3), compression="gzip", compression_opts=9, data=np.asarray(loop_instructions))
            dset_loop_ops = h5f.create_dataset("simulation/results/loop_ops/" + new_key, (128, 1000, 7), compression="gzip", compression_opts=9, data=np.asarray(loop_ops))
            dset_operator_string = h5f.create_dataset("simulation/results/operator_string/" + new_key, (128, 1000, 3), compression="gzip", compression_opts=9, data=np.asarray(operator_string_obs))
            # print("Dumping sz_corr")
            dset_sz = h5f.create_dataset("simulation/results/sz_corr/" + new_key, (128, vol, vol), compression="gzip", compression_opts=9, data=np.asarray(sz_corr))
            # dset_sz = np.asarray(sz_corr)[:]
            # print("Dumping spm_corr")
            dset_spm = h5f.create_dataset("simulation/results/spm_corr/" + new_key, (128, vol, vol), compression="gzip", compression_opts=9, data=np.asarray(spm_corr))
            # dset_spm = np.asarray(spm_corr)[:]
            # print("Dumping spin_imgs")
            dset_loop = h5f.create_dataset("simulation/results/loop_img/" + new_key, (128, L, L, 200), compression="gzip", compression_opts=9, data=np.asarray(loop_img))
            dset_single_loop = h5f.create_dataset("simulation/results/single_loop_img/" + new_key, (128, L, L, 200), compression="gzip", compression_opts=9, data=np.asarray(single_loop_img))
            print("Dumping op_stats")
            dset_op_stats = h5f.create_dataset("simulation/results/op_stats/" + new_key, (3, 1), compression="gzip", compression_opts=9, data=np.asarray(op_stats))
            print("Dumping op_stats done")
            dset_s = h5f.create_dataset("simulation/results/spin_imgs/" + new_key, (128, vol), compression="gzip", compression_opts=9, data=np.asarray(spin_imgs))
            dset_w = h5f.create_dataset("simulation/results/worldline_imgs/" + new_key, (128, 512, L, L), compression="gzip", compression_opts=9, data=np.asarray(worldline_imgs))
            dset_ww = h5f.create_dataset("simulation/results/worldline_wind_imgs/" + new_key, (128, 512, L, L), compression="gzip", compression_opts=9, data=np.asarray(worldline_wind_imgs))
            dset_wix = h5f.create_dataset("simulation/results/windings_x_imgs/" + new_key, (128, 512, L, L), compression="gzip", compression_opts=9, data=np.asarray(worldline_wind_imgs))
            dset_wiy = h5f.create_dataset("simulation/results/windings_y_imgs/" + new_key, (128, 512, L, L), compression="gzip", compression_opts=9, data=np.asarray(worldline_wind_imgs))
            dset_wx = h5f.create_dataset("simulation/results/windings_x/" + new_key, (128, L, L), compression="gzip", compression_opts=9, data=np.asarray(windings_x_obs))
            dset_wy = h5f.create_dataset("simulation/results/windings_y/" + new_key, (128, L, L), compression="gzip", compression_opts=9, data=np.asarray(windings_y_obs))
            dset_nwx = h5f.create_dataset("simulation/results/norm_windings_x/" + new_key, (128, L, L), compression="gzip", compression_opts=9, data=np.asarray(norm_windings_x_obs))
            dset_nwy = h5f.create_dataset("simulation/results/norm_windings_y/" + new_key, (128, L, L), compression="gzip", compression_opts=9, data=np.asarray(norm_windings_y_obs))

            # dset_s = np.asarray(spin_imgs)[:]
            h5f.close()
            # print("Dumping", new_key, "was a success")
            # print(np.asarray(spin_imgs)[:])
            # print(type(np.asarray(spin_imgs)))
            # sys.exit(0)
            curr_conf = 0
            sz_corr[...] = 0
            spm_corr[...] = 0
            spin_imgs[...] = 0
            worldline_imgs[...] = 0
            worldline_wind_imgs[...] = 0
            windings_x_imgs[...] = 0
            windings_y_imgs[...] = 0
            windings_x[...] = 0
            windings_y[...] = 0
            windings_x_obs[...] = 0
            windings_y_obs[...] = 0
            norm_windings_x_obs[...] = 0
            norm_windings_y_obs[...] = 0
            loop_img[...] = 0            
            single_loop_img[...] = 0
            op_stats[...] = 0
            operator_string_obs[...] = 0
            loop_instructions[...] = 0
            loop_ops[...] = 0

            if rethermalize == 1:
                print("Rethermalizing")
                curr_expansion_order = 0
                # leg_spins[:] = 0
                legs[:] = 0
                operator_string[:] = 0
                operator_string_obs[:] = 0
                loop_ops[:] = 0
                print("Sweeping")
                for u in range(thermalizations):
                    # print(u)
                    diagonal_update()
                    # print("Diagonal update done")
                    # for i in range(expansion_order):
                    #     print(operator_string[i, 0], operator_string[i, 1], operator_string[i, 2])
                        
                    off_diagonal_update()
                    # print("Off diagonal update done")


    obs.obs2hdf5(output_filename, obs_energy)
    obs.obs2hdf5(output_filename, obs_energy_2)
    obs.obs2hdf5(output_filename, obs_magnetization)
    obs.obs2hdf5(output_filename, obs_magnetization_2)
    obs.obs2hdf5(output_filename, obs_staggered_magnetization)
    obs.obs2hdf5(output_filename, obs_staggered_magnetization_2)
    obs.obs2hdf5(output_filename, obs_density)
    obs.obs2hdf5(output_filename, obs_exp_order)
    h5f = h5py.File(output_filename, "a")

    print("initial sites")
    print(np.asarray(initial_sites))
    print("rnd_bonds")
    print(np.asarray(rnd_bonds))
    print("rnd_a")
    print(np.asarray(rnd_a))
    print("rnd_b")
    print(np.asarray(rnd_b))

    dset_initial_sites = h5f.create_dataset("simulation/results/initial_sites", vol, compression="gzip", compression_opts=9, data=np.asarray(initial_sites))
    print("rnd bonds")
    dset_rnd_bonds = h5f.create_dataset("simulation/results/rnd_bonds", 2 * vol, compression="gzip", compression_opts=9, data=np.asarray(rnd_bonds))
    dset_rnd_a = h5f.create_dataset("simulation/results/rnd_a", vol, compression="gzip", compression_opts=9, data=np.asarray(rnd_a))
    dset_rnd_b = h5f.create_dataset("simulation/results/rnd_b", vol, compression="gzip", compression_opts=9, data=np.asarray(rnd_b))
    h5f.close()
    # specific_heat = beta^2 * vol * (mean(obs_energy.timeseries)^2 - mean(obs_energy_2.timeseries))
    # magnetic_susceptibility = -beta * vol^2 * (mean(obs_magnetization.timeseries)^2 - mean(obs_magnetization_2.timeseries))
    # HDF5.h5write(output_file, "simulation/results/Specific Heat", specific_heat)
    # HDF5.h5write(output_file, "simulation/results/Magnetic Susceptibility", magnetic_susceptibility)

############################################################################
############################################################################
############################################################################
cdef double weight(int i, int j):
    global L, vol, n_bonds, beta, expansion_order, curr_expansion_order, Delta, h, eps, H_c

    w = H_c
    w += -Delta * 0.25 * i * j
    w += -h / 4. * 0.5 * (i + j)
    return w


############################################################################
############################################################################
############################################################################
# cdef long vertex_to_op(int exp_order):
    # global operator_string
    #
    # cdef long op
    #
    # s1 = 4 * (exp_order - 1) + 2 * vol + 1
    # s2 = 4 * (exp_order - 1) + 2 * vol + 2
    # s3 = 4 * (exp_order - 1) + 2 * vol + 3
    # s4 = 4 * (exp_order - 1) + 2 * vol + 4
    #
    # print(leg_spins[s2], "\t", leg_spins[s4])
    # print("========    ", operator_string[exp_order, 0], "     ====")
    # print(leg_spins[s1], "\t", leg_spins[s3], "\n")
    # if (leg_spins[s1] == leg_spins[s3]) and (leg_spins[s1] == leg_spins[s2]) and (leg_spins[s3] == leg_spins[s4]): # Ferro
    #     op = 1 if leg_spins[s1] == 1 else 2
    # elif (leg_spins[s1] != leg_spins[s3]) and (leg_spins[s2] != leg_spins[s4]) and (leg_spins[s1] == leg_spins[s2]): # AF
    #     op = 3
    # elif (leg_spins[s1] != leg_spins[s2]) and (leg_spins[s3] != leg_spins[s4 ]) and (leg_spins[s1] == leg_spins[s4]): # OffD
    #     op = 4
    #
    # return op

############################################################################
############################################################################
############################################################################
cdef void find_exit_leg(int exp_order, int leg):
    global vol
    global fe_up_fe_up, fe_up_od, up_af_af, up_af_od, up_od_od, od_fe_up, fe_dn_fe_dn, fe_dn_od, dn_af_af, dn_af_od, dn_od_od, od_fe_dn
    global leg_spins, operator_string
    global exit_leg, exit_op

    cdef long op_type = operator_string[exp_order, 0]
    cdef double r = rnd()

    # print("Input to np.outer", leg_spins[4 * exp_order + 2 * vol + leg], leg, op_type, exp_order)

    if leg_spins[4 * exp_order + 2 * vol + leg] == 1:
        if leg == 0:
            if op_type == 1:
                if r < fe_up_fe_up:
                    exit_op = 1
                    exit_leg = 1
                elif r > fe_up_od:
                    exit_op = 3
                    exit_leg = 4
                else:
                    exit_op = 2
                    exit_leg = 2
            elif op_type == 2:
                if r < dn_af_af:
                    exit_op = 2
                    exit_leg = 1
                elif r > dn_af_od:
                    exit_op = 3
                    exit_leg = 3
                else:
                    exit_op = 1
                    exit_leg = 2
            elif op_type == 3:
                if r < dn_od_od:
                    exit_op = 3
                    exit_leg = 1
                elif r > od_fe_dn:
                    exit_op = 1
                    exit_leg = 4
                else:
                    exit_op = 2
                    exit_leg = 3
        elif leg == 1:
            if op_type == 1:
                if r < fe_up_fe_up:
                    exit_op = 1
                    exit_leg = 2
                elif r > fe_up_od:
                    exit_op = 3
                    exit_leg = 3
                else :
                    exit_op = 2
                    exit_leg = 1
            elif op_type == 2:
                if r < dn_af_af:
                    exit_op = 2
                    exit_leg = 2
                elif r > dn_af_od:
                    exit_op = 3
                    exit_leg = 4
                else:
                    exit_op = 1
                    exit_leg = 1
            elif op_type == 3:
                if r < dn_od_od:
                    exit_op = 3
                    exit_leg = 2
                elif r > od_fe_dn:
                    exit_op = 1
                    exit_leg = 3
                else:
                    exit_op = 2
                    exit_leg = 4
        elif leg == 2:
            if op_type == 1:
                if r < fe_up_fe_up:
                    exit_op = 1
                    exit_leg = 3
                elif r > fe_up_od:
                    exit_op = 3
                    exit_leg = 2
                else:
                    exit_op = 2
                    exit_leg = 4
            elif op_type == 2:
                if r < dn_af_af:
                    exit_op = 2
                    exit_leg = 3
                elif r > dn_af_od:
                    exit_op = 3
                    exit_leg = 1
                else:
                    exit_op = 1
                    exit_leg = 4
            elif op_type == 3:
                if r < dn_od_od:
                    exit_op = 3
                    exit_leg = 3
                elif r > od_fe_dn:
                    exit_op = 1
                    exit_leg = 2
                else:
                    exit_op = 2
                    exit_leg = 1
        elif leg == 3:
            if op_type == 1:
                if r < fe_up_fe_up:
                    exit_op = 1
                    exit_leg = 4
                elif r > fe_up_od:
                    exit_op = 3
                    exit_leg = 1
                else:
                    exit_op = 2
                    exit_leg = 3
            elif op_type == 2:
                if r < dn_af_af:
                    exit_op = 2
                    exit_leg = 4
                elif r > dn_af_od:
                    exit_op = 3
                    exit_leg = 2
                else:
                    exit_op = 1
                    exit_leg = 3
            elif op_type == 3:
                if r < dn_od_od:
                    exit_op = 3
                    exit_leg = 4
                elif r > od_fe_dn:
                    exit_op = 1
                    exit_leg = 1
                else:
                    exit_op = 2
                    exit_leg = 2

    else:
        if leg == 0:
            if op_type == 1:
                if r < fe_dn_fe_dn:
                    exit_op = 1
                    exit_leg = 1
                elif r > fe_dn_od:
                    exit_op = 3
                    exit_leg = 4
                else:
                    exit_op = 2
                    exit_leg = 2
            elif op_type == 2:
                if r < up_af_af:
                    exit_op = 2
                    exit_leg = 1
                elif r > up_af_od:
                    exit_op = 3
                    exit_leg = 3
                else:
                    exit_op = 1
                    exit_leg = 2
            elif op_type == 3:
                if r < up_od_od:
                    exit_op = 3
                    exit_leg = 1
                elif r > od_fe_up:
                    exit_op = 1
                    exit_leg = 4
                else:
                    exit_op = 2
                    exit_leg = 3
        elif leg == 1:
            if op_type == 1:
                if r < fe_dn_fe_dn:
                    exit_op = 1
                    exit_leg = 2
                elif r > fe_dn_od:
                    exit_op = 3
                    exit_leg = 3
                else:
                    exit_op = 2
                    exit_leg = 1
            elif op_type == 2:
                if r < up_af_af:
                    exit_op = 2
                    exit_leg = 2
                elif r > up_af_od:
                    exit_op = 3
                    exit_leg = 4
                else:
                    exit_op = 1
                    exit_leg = 1
            elif op_type == 3:
                if r < up_od_od:
                    exit_op = 3
                    exit_leg = 2
                elif r > od_fe_up:
                    exit_op = 1
                    exit_leg = 3
                else:
                    exit_op = 2
                    exit_leg = 4
        elif leg == 2:
            if op_type == 1:
                if r < fe_dn_fe_dn:
                    exit_op = 1
                    exit_leg = 3
                elif r > fe_dn_od:
                    exit_op = 3
                    exit_leg = 2
                else:
                    exit_op = 2
                    exit_leg = 4
            elif op_type == 2:
                if r < up_af_af:
                    exit_op = 2
                    exit_leg = 3
                elif r > up_af_od:
                    exit_op = 3
                    exit_leg = 1
                else:
                    exit_op = 1
                    exit_leg = 4
            elif op_type == 3:
                if r < up_od_od:
                    exit_op = 3
                    exit_leg = 3
                elif r > od_fe_up:
                    exit_op = 1
                    exit_leg = 2
                else:
                    exit_op = 2
                    exit_leg = 1
        elif leg == 3:
            if op_type == 1:
                if r < fe_dn_fe_dn:
                    exit_op = 1
                    exit_leg = 4
                elif r > fe_dn_od:
                    exit_op = 3
                    exit_leg = 1
                else:
                    exit_op = 2
                    exit_leg = 3
            elif op_type == 2:
                if r < up_af_af:
                    exit_op = 2
                    exit_leg = 4
                elif r > up_af_od:
                    exit_op = 3
                    exit_leg = 2
                else:
                    exit_op = 1
                    exit_leg = 3
            elif op_type == 3:
                if r < up_od_od:
                    exit_op = 3
                    exit_leg = 4
                elif r > od_fe_up:
                    exit_op = 1
                    exit_leg = 1
                else:
                    exit_op = 2
                    exit_leg = 2
    exit_leg -= 1

# The idea is that on the 0th time slice, all sites have a one site
# operator that even in the absence of any other operators ensures periodic
# boundary conditions in operator time. These operators occupy the first
# 2 * vol entries (up and down leg) of the legs
cdef void check_operator(int exp_order):
    global vol, operator_string, leg_spins

    if operator_string[exp_order, 0] == 0: return

    s1 = 4 * exp_order + 2 * vol + 0
    s2 = 4 * exp_order + 2 * vol + 1
    s3 = 4 * exp_order + 2 * vol + 2
    s4 = 4 * exp_order + 2 * vol + 3

    if operator_string[exp_order, 0] == 1:
        if (leg_spins[s1] != leg_spins[s3]) or \
            (leg_spins[s2] != leg_spins[s4]) or \
            (leg_spins[s1] != leg_spins[s2]):
            print("Diag Ferromagnetic operator {exp_order} has wrong spins".format(exp_order=exp_order))
            print(leg_spins[s2], " ", leg_spins[s4])
            print(leg_spins[s1], " ", leg_spins[s3])

    elif operator_string[exp_order, 0] == 2:
        if (leg_spins[s1] == leg_spins[s3]) or \
            (leg_spins[s2] == leg_spins[s4]) or \
            (leg_spins[s1] != leg_spins[s2]):
            print("Diag Antiferromagnetic operator {exp_order} has wrong spins".format(exp_order=exp_order))
            print(leg_spins[s2], " ", leg_spins[s4])
            print(leg_spins[s1], " ", leg_spins[s3])

    elif operator_string[exp_order, 0] == 3:
        if (leg_spins[s1] == leg_spins[s2]) or\
            (leg_spins[s3] == leg_spins[s4]) or\
            (leg_spins[s1] != leg_spins[s4]):
            print("Offd Antiferromagnetic operator {exp_order}  has wrong spins".format(exp_order=exp_order))
            print(leg_spins[s2], " ", leg_spins[s4])
            print(leg_spins[s1], " ", leg_spins[s3])

# here diagonal
cdef diagonal_update():
    global L, vol, expansion_order, curr_expansion_order, curr_conf, worldline_imgs, worldline_wind_imgs, curr_conf
    global spins, legs, leg_spins, operator_string, operator_string_obs, initial_sites, rnd_bonds, rnd_a, rnd_b, op_stats
    global windings_x, windings_y, worldline_sites, windings_x_obs, windings_y_obs, windings_x_imgs, windings_y_imgs
    global norm_windings_x_obs, norm_windings_y_obs
    
    # print("Diagonal update")
    cdef long[:] check_spins = leg_spins[0:vol]
    cdef long[:] num_od_ops = np.zeros(vol, dtype=np.int64)
    cdef long i, j, o, a, b, d, op_type, site_a, site_b, last_leg, wldl_a, wldl_b
    cdef long rnd_i, rnd_dir, rnd_j, total_winding_x, total_winding_y
    cdef float max_wind_x
    cdef float max_wind_y
    cdef float max_wind

    check_leg_integrity()
    spins = leg_spins[0:vol]

    windings_x[...] = 0
    windings_y[...] = 0

    for i in range(vol): worldline_sites[i] = i

    total_winding_x = 0
    total_winding_y = 0
    max_wind_x = 0
    max_wind_y = 0
    max_wind = 0

    # print("Windings")
    for i in range(expansion_order):
        if operator_string[i, 0] == 3:
            site_a = operator_string[i, 1]
            site_b = operator_string[i, 2]
            worldline_sites[site_a], worldline_sites[site_b] = worldline_sites[site_b], worldline_sites[site_a]

            if operator_string[i, 3] == 0:
                if spins[worldline_sites[site_a]] == 1:
                    windings_x[worldline_sites[site_a]] += 1
                    # windings_x[worldline_sites[site_b]] -= 1
                    
                if spins[worldline_sites[site_b]] == 1:
                    # windings_x[worldline_sites[site_a]] -= 1
                    windings_x[worldline_sites[site_b]] -= 1

                if abs(windings_x[worldline_sites[site_a]]) > max_wind_x: max_wind_x = abs(windings_x[worldline_sites[site_a]])
                if abs(windings_x[worldline_sites[site_b]]) > max_wind_x: max_wind_x = abs(windings_x[worldline_sites[site_b]])


                site_a = i * 4 + 2 * vol
                if leg_spins[site_a] == 1:
                    total_winding_x += 1
                else:
                    total_winding_x += -1
            else:
                if spins[worldline_sites[site_a]] == 1:
                    windings_y[worldline_sites[site_a]] += 1
                    # windings_y[worldline_sites[site_b]] -= 1
                    
                if spins[worldline_sites[site_b]] == 1:
                    # windings_y[worldline_sites[site_a]] -= 1
                    windings_y[worldline_sites[site_b]] -= 1

                if abs(windings_y[worldline_sites[site_a]]) > max_wind_y: max_wind_y = abs(windings_y[worldline_sites[site_a]])
                if abs(windings_y[worldline_sites[site_b]]) > max_wind_y: max_wind_y = abs(windings_y[worldline_sites[site_b]])

                site_a = i * 4 + 2 * vol
                if leg_spins[site_a] == 1:
                    total_winding_y += 1
                else:
                    total_winding_y += -1

    max_wind = max_wind_x * max_wind_x + max_wind_y * max_wind_y
    # for i in range(vol):
    #     print worldline_sites[i],
    # print("After sweep ")
    # print("Total windings are ", total_winding_x, total_winding_y)
    # print(np.sum(windings_x), np.sum(windings_y))
    
    for i in range(vol): worldline_sites[i] = i

    # print("winding imgs")
    if max_wind == 0: max_wind = 1
    if max_wind_x == 0: max_wind_x = 1
    if max_wind_y == 0: max_wind_y = 1

    for i in range(expansion_order):
        
        if operator_string[i, 0] == 3:

            site_a = operator_string[i, 1]
            site_b = operator_string[i, 2]
            worldline_sites[site_a], worldline_sites[site_b] = worldline_sites[site_b], worldline_sites[site_a]

        if i < 1000:
            operator_string_obs[curr_conf, i, 0] = operator_string[i, 0]
            operator_string_obs[curr_conf, i, 1] = operator_string[i, 1]
            operator_string_obs[curr_conf, i, 2] = operator_string[i, 2]
            
        if i < 512:
            for a in range(L):
                for b in range(L):
                    worldline_imgs[curr_conf, i, a, b] = np.reshape(spins, (L, L))[a, b]
                    site_a = a + L * b
                    worldline_wind_imgs[curr_conf, i, a, b] = worldline_imgs[curr_conf, i, a, b] * (1 + windings_x[worldline_sites[site_a]] * windings_x[worldline_sites[site_a]] + windings_y[worldline_sites[site_a]] * windings_y[worldline_sites[site_a]]) / max_wind
                    windings_x_imgs[curr_conf, i, a, b] = windings_x[worldline_sites[site_a]] / max_wind_x
                    windings_y_imgs[curr_conf, i, a, b] = windings_y[worldline_sites[site_a]] / max_wind_y

                    if i == 0:
                        windings_x_obs[curr_conf, a, b] = windings_x[site_a]
                        windings_y_obs[curr_conf, a, b] = windings_y[site_a]
                        norm_windings_x_obs[curr_conf, a, b] = windings_x[site_a] / max_wind_x
                        norm_windings_y_obs[curr_conf, a, b] = windings_y[site_a] / max_wind_y

        op_type = operator_string[i, 0]
        if op_type == 0: continue

        site_a = operator_string[i, 1]
        site_b = operator_string[i, 2]

        check_operator(i)
        last_leg = 2 * vol + i * 4

        if leg_spins[last_leg + 0] != spins[site_a] or\
            leg_spins[last_leg + 2   ] != spins[site_b]:
            print("Operator error")
            print(leg_spins[last_leg + 1], "\t", leg_spins[last_leg + 3])
            print("=============\t", op_type)
            print(leg_spins[last_leg + 0], "\t", leg_spins[last_leg + 2])
            print("Spins are unequal for op_type ", op_type, " - ", i, " - ", \
                spins[site_a], " | ", spins[site_b])
            raise

        if op_type == 3:
            spins[site_a] = leg_spins[last_leg + 1]
            spins[site_b] = leg_spins[last_leg + 3]                       

    # if np.max(nabs(check_spins - spins)) > 0:
    #     print(check_spins)
    #     print(spins)
    #     print("Spins were incorrectly propagated")
    #     raise

    # print("Check done for sweep ...")

    legs[:] = 0
    cdef long[::1] dangling_entry = np.zeros((vol), dtype=np.int64)

    for i in range(vol):
        dangling_entry[i] = i + vol

    for i in range(expansion_order):            
        op_type = operator_string[i, 0]
        site_a =  operator_string[i, 1]
        site_b =  operator_string[i, 2]

        if op_type != 0:
            op_stats[op_type -1, 0] += 1

        if op_type == 3:
            num_od_ops[site_a] = num_od_ops[site_a] + 1
            num_od_ops[site_b] = num_od_ops[site_b] + 1

        last_leg = i * 4 + 2 *  vol

        if op_type == 0:
            rnd_i = int(floor(rnd() * L))
            rnd_j = int(floor(rnd() * L))
            rnd_dir = int(floor(rnd() * 2))

            site_a = rnd_i + L * rnd_j

            if rnd_dir == 0:
                rnd_i = (rnd_i + 1) % L
            elif rnd_dir == 1:
                rnd_j = (rnd_j + 1) % L
            site_b = rnd_i + L * rnd_j

            w = weight(spins[site_a], spins[site_b])
            if w < 0: print("Negative weights"); raise

            # print(np.min([1., beta * w * (2 * vol * vol) \
            #     / (expansion_order - curr_expansion_order)]))
            if not rnd() < min(1., beta * w * (2 * L * L) \
                / (expansion_order - curr_expansion_order)):
                continue

            rnd_bonds[site_a + rnd_dir * vol] = rnd_bonds[site_a + rnd_dir * vol] + 1
            rnd_a[site_a] = rnd_a[site_a] + 1
            rnd_b[site_b] = rnd_b[site_b] + 1

            insert_op = 1 if spins[site_a] == spins[site_b] else 2

            # print("Inserting ", i, insert_op, last_leg, curr_expansion_order)
            operator_string[i, 0] = insert_op
            operator_string[i, 1] = site_a
            operator_string[i, 2] = site_b
            operator_string[i, 3] = rnd_dir

            curr_expansion_order += 1

            leg_spins[last_leg + 0] = spins[site_a]
            leg_spins[last_leg + 1] = spins[site_a]
            leg_spins[last_leg + 2] = spins[site_b]
            leg_spins[last_leg + 3] = spins[site_b]

            legs[dangling_entry[site_a]] = last_leg + 0
            legs[dangling_entry[site_b]] = last_leg + 2
            legs[last_leg + 0] = dangling_entry[site_a]
            legs[last_leg + 2] = dangling_entry[site_b]
            dangling_entry[site_a] = last_leg + 1
            dangling_entry[site_b] = last_leg + 3

            # remove diagonal operator?
        elif op_type != 0 and op_type != 3:
            check_operator(i)
            if leg_spins[last_leg + 0] != spins[site_a] or leg_spins[last_leg + 2] != spins[site_b]:
                print("Operator error")
                print(leg_spins[last_leg + 1], "\t", leg_spins[last_leg + 3])
                print("==================")
                print(leg_spins[last_leg + 0], "\t", leg_spins[last_leg + 2])
                print("Spins are unequal for op_type $(op_type) - ", i, " - ", spins[site_a], " | ", spins[site_b])
                raise

            w = weight(spins[site_a], spins[site_b])

            # print("Removing with prob ", (expansion_order - curr_expansion_order + 1) / (beta * w * (2 * vol * vol)))
            # print("And thus ", np.min([1., (expansion_order - curr_expansion_order + 1) / (beta * w * (2 * vol * vol) )]))
            if rnd() < min(1., (expansion_order - curr_expansion_order + 1) / (beta * w * (2 * L * L) )):
                operator_string[i, 0] = 0
                operator_string[i, 1] = 0
                operator_string[i, 2] = 0
                operator_string[i, 3] = 0
                leg_spins[last_leg + 0] = 0
                leg_spins[last_leg + 2] = 0
                leg_spins[last_leg + 1] = 0
                leg_spins[last_leg + 3] = 0
                curr_expansion_order -= 1

            else:
                rnd_i = dangling_entry[site_a]
                legs[rnd_i] = last_leg + 0
                legs[dangling_entry[site_b]] = last_leg + 2
                legs[last_leg + 0] = dangling_entry[site_a]
                legs[last_leg + 2] = dangling_entry[site_b]
                dangling_entry[site_a] = last_leg + 1
                dangling_entry[site_b] = last_leg + 3

        elif op_type == 3:
            check_operator(i)
            if leg_spins[last_leg + 0] != spins[site_a] or \
                leg_spins[last_leg + 2] != spins[site_b]:

                print("Operator error")
                print(leg_spins[last_leg + 1], "\t", leg_spins[last_leg + 3])
                print("==================")
                print(leg_spins[last_leg + 0], "\t", leg_spins[last_leg + 2])
                print("ERROR: Spins are unequal for op_type $(op_type) - ", i, " - ", spins[site_a], " | ", spins[site_b])
                spins[:] = check_spins

                for j in range(i - 1):
                    o, a, b, d = operator_string[j, :]
                    if o == 3:
                        spins[a] *= -1
                        spins[b] *= -1

                if leg_spins[last_leg + 0] != spins[site_a] or \
                    leg_spins[last_leg + 2] != spins[site_b]:
                    print("ERROR: resolved")
                else:
                    print("ERROR remains: Spins are unequal for op_type $(op_type) - ", i, " - ", spins[site_a], " | ", spins[site_b])
                    raise

            if spins[site_a] == spins[site_b]:
                print(leg_spins[last_leg + 1], "\t", leg_spins[last_leg + 3])
                print("==================")
                print(leg_spins[last_leg + 0], "\t", leg_spins[last_leg + 2])
                print("Spins are equal on op_type $(op_type) - ", i, " - ", site_a, " | ", site_b)
                raise

            spins[site_a] = leg_spins[last_leg + 1]
            spins[site_b] = leg_spins[last_leg + 3]

            legs[dangling_entry[site_a]] = last_leg + 0
            legs[dangling_entry[site_b]] = last_leg + 2
            legs[last_leg + 0] = dangling_entry[site_a]
            legs[last_leg + 2] = dangling_entry[site_b]
            dangling_entry[site_a] = last_leg + 1
            dangling_entry[site_b] = last_leg + 3

    for i in range(vol):
        if num_od_ops[i] % 2 != 0:
            print("Uneven number of off diagonal ops on site ", i)
            print(num_od_ops)
            raise

    # if np.max(nabs(check_spins - spins)) > 0:
    #     print(check_spins - spins)
    #     print("Spins were incorrectly propagated")
    #     raise

    # print(dangling_entry)
    # connect in periodic boundary conditions
    for i in range(vol):
        legs[i] = dangling_entry[i]
        legs[dangling_entry[i]] = i


cdef check_leg_integrity():
    global L, vol, expansion_order, curr_expansion_order
    global spins, legs, leg_spins, operator_string

    cdef long n_od_ops = 0
    cdef long i
    cdef long exp_order
    # print("Check leg integrity\t",)

    for i in range(expansion_order):
        if operator_string[i, 0] != 0:
            check_operator(i)

        if operator_string[i, 0] == 3:
            n_od_ops += 1

    # print("#Od ops", n_od_ops)

    for i in range(len(legs)):
        if legs[i] == 0: continue

        if i >= 2 * vol:
            exp_order = int(floor((i - 2 * vol) / 4))
            if operator_string[exp_order, 0] == 0:
                print("Error in legs")
                raise

        if legs[legs[i]] != i:
            print("Error in legs ", i, " - ", legs[i], " - ", legs[legs[i]])
            raise

        if leg_spins[legs[legs[i]]] != leg_spins[i]:
            print("Error in leg spins ", i, " - ", legs[i], " - ", legs[legs[i]])
            raise

    # print("Check leg integrity successful")

cdef int leg_to_site(int leg):
    global L, vol, expansion_order, curr_expansion_order
    global spins, legs, leg_spins, operator_string

    cdef long exp_order, leg_type

    if leg < vol:
        return leg
    elif leg >= vol and leg < 2 * vol:
        return leg - vol
    else:
        exp_order = int(floor((leg - 2 * vol) / 4))
        leg_type = (leg - 2 * vol) % 4
        if leg_type < 2:
            return operator_string[exp_order, 1]
        else:
            return operator_string[exp_order, 2]

@cython.boundscheck(False)
cdef void off_diagonal_update():
    global L, vol, expansion_order, curr_expansion_order
    global spins, legs, leg_spins, operator_string
    global curr_conf, spm_corr, loop_instructions, loop_ops, loop_img, single_loop_img, sz_corr, spin_imgs, rnd_bonds, initial_sites
    # print("off diagonal update")

    # if curr_conf > 0:
    #     print("Saving in ", curr_conf)
    #     print("Spins\n", spins[0])
    #     print("Spins\n", np.asarray(spins)[:])

    if np.max(operator_string[:, 0]) == 0: return

    cdef long i, j, k, l, m, dx, dy
    cdef long total_loop_length = 0
    cdef long current_loop, loop
    cdef long op_leg, leg, initial_leg, leg_type, initial_op, next_leg
    cdef long n_off_diag_operators
    cdef long search_counter, exp_order, next_exp_order, step_counter
    cdef long initial_site, curr_site, curr_row, curr_col, init_row, init_col, op_time_bottom, op_time_top

    for i in range(vol):
        spin_imgs[curr_conf, i] = spins[i]

        for j in range(vol):
            sz_corr[curr_conf, i, j] = spins[i] * spins[j]

    loop = 0

    # while (total_loop_length < 10 * curr_expansion_order + 1)\
    #     or (loop < 10 * vol):
    while loop < 10 * vol:

        # if loop < 1:
        #     print("Starting loop", loop)
        search_counter = 0
        r = rnd()
        initial_op = int(floor(rnd() * expansion_order))

        while operator_string[initial_op, 0] == 0 and\
            initial_op < expansion_order - 1:
            initial_op += 1

        if initial_op >= expansion_order or operator_string[initial_op, 0] == 0:
            # loop -= 1
            continue

        initial_leg = 2 * vol + int(floor(rnd() * 4.)) + initial_op * 4
        leg = initial_leg
        initial_site = leg_to_site(initial_leg)
        init_row = int(floor(initial_site / float(L)))
        init_col = int(initial_site % L)

        step_counter = 0
        loop_length = 1

        initial_sites[leg_to_site(leg)] = initial_sites[leg_to_site(leg)] + 1

        while step_counter < 1e21:
            if step_counter > 0 and leg == initial_leg: break
            total_loop_length += 1
            leg_type = (leg - 2 * vol) % 4

            if leg < vol:
                next_leg = leg + vol
                exp_order = 0
                
            elif leg >= vol and leg < 2 * vol:
                next_leg = leg - vol
                exp_order = expansion_order - 1
            else:
                exp_order = int(floor((leg - 2 * vol) / 4))
                if operator_string[exp_order, 0] == 0:
                    print("No operator at order ", exp_order, leg)
                    raise
                # print("Leg type", leg_type)
                find_exit_leg(exp_order, leg_type)
                # print("Exiting to ", exit)
                if step_counter == 0 and exit_leg == leg_type: break

                if step_counter < 1000:
                    loop_instructions[curr_conf, step_counter, 0] = operator_string[exp_order, 0]
                    loop_instructions[curr_conf, step_counter, 1] = leg_type
                    loop_instructions[curr_conf, step_counter, 2] = exit_leg

                    loop_ops[curr_conf, step_counter, 0] = operator_string[exp_order, 0]
                    loop_ops[curr_conf, step_counter, 1] = leg_type
                    loop_ops[curr_conf, step_counter, 2] = exit_leg
                    loop_ops[curr_conf, step_counter, 3] = leg_spins[leg_type + 4 * exp_order + 2 * vol]

                    if leg_type == 0:
                        loop_ops[curr_conf, step_counter, 4] = leg_spins[(leg_type + 1) + 4 * exp_order + 2 * vol]
                        loop_ops[curr_conf, step_counter, 5] = leg_spins[(leg_type + 2) + 4 * exp_order + 2 * vol]
                        loop_ops[curr_conf, step_counter, 6] = leg_spins[(leg_type + 3) + 4 * exp_order + 2 * vol]
                    elif leg_type == 1:
                        loop_ops[curr_conf, step_counter, 4] = leg_spins[(leg_type - 1) + 4 * exp_order + 2 * vol]
                        loop_ops[curr_conf, step_counter, 5] = leg_spins[(leg_type + 2) + 4 * exp_order + 2 * vol]
                        loop_ops[curr_conf, step_counter, 6] = leg_spins[(leg_type + 1) + 4 * exp_order + 2 * vol]
                    elif leg_type == 2:
                        loop_ops[curr_conf, step_counter, 4] = leg_spins[(leg_type + 1) + 4 * exp_order + 2 * vol]
                        loop_ops[curr_conf, step_counter, 5] = leg_spins[(leg_type - 2) + 4 * exp_order + 2 * vol]
                        loop_ops[curr_conf, step_counter, 6] = leg_spins[(leg_type - 1) + 4 * exp_order + 2 * vol]
                    elif leg_type == 3:
                        loop_ops[curr_conf, step_counter, 4] = leg_spins[(leg_type - 1) + 4 * exp_order + 2 * vol]
                        loop_ops[curr_conf, step_counter, 5] = leg_spins[(leg_type - 2) + 4 * exp_order + 2 * vol]
                        loop_ops[curr_conf, step_counter, 6] = leg_spins[(leg_type - 3) + 4 * exp_order + 2 * vol]
                    

                if exit_op == 3: n_off_diag_operators += 1
                operator_string[exp_order, 0] = exit_op
                next_leg = exit_leg + 4 * exp_order + 2 * vol
                    
            leg_spins[leg] = leg_spins[leg] * -1
            leg_spins[next_leg] = leg_spins[next_leg] * -1

            if next_leg == initial_leg:
                break
                
            curr_site = leg_to_site(next_leg)

            if min(next_leg, legs[next_leg]) < initial_leg and \
                max(next_leg, legs[next_leg]) > initial_leg:
                spm_corr[curr_conf, initial_site, curr_site] += 1
                spm_corr[curr_conf, curr_site, initial_site] += 1

            # save loop image only once, do not accumulate
            # scratch that, accumulate!
            if loop != -1:
                if legs[next_leg] < vol:
                    next_exp_order = 0
                elif legs[next_leg] >= vol and legs[next_leg] < 2 * vol:
                    next_exp_order = expansion_order - 1
                else:
                    next_exp_order = int(floor((legs[next_leg] - 2 * vol) / 4))


                # print("Op times")
                # print(initial_op, exp_order, next_exp_order)
                # print(initial_op - exp_order, initial_op - next_exp_order)
                if ((initial_op - exp_order) > 100 and (initial_op - next_exp_order) > 100) or \
                ((initial_op - exp_order) < -100 and (initial_op - next_exp_order) < -100):
                    # print("Skipping")
                    # print ((initial_op - exp_order) > 100 and (initial_op - next_exp_order) > 100)
                    # print ((initial_op - exp_order) < -100 and (initial_op - next_exp_order) < -100)
                    pass
                else:
                    # print("Yeah", initial_op, exp_order, next_exp_order)
                    op_time_bottom = max(0, min(199, initial_op - exp_order + 100))
                    op_time_top = max(0, min(199, initial_op - next_exp_order + 100))

                    curr_row = int(floor(curr_site / float(L)))
                    curr_col = int(curr_site % L)

                    dx = init_row
                    dy = init_col

                    i = (curr_row + L - dx) % L
                    j = (curr_col + L - dy) % L
                    k = min(op_time_bottom, op_time_top)
                    l = max(op_time_bottom, op_time_top)
                    for m in range(k, l):
                        loop_img[curr_conf, i, j, m] += 1
                        if loop == 0:
                            single_loop_img[curr_conf, i, j, m] += 1

            if legs[next_leg] == initial_leg: break
            leg = legs[next_leg]
            step_counter += 1
        loop += 1

    check_leg_integrity()

cdef measurements():
    global L, vol, expansion_order, curr_expansion_order
    global spins, legs, leg_spins, operator_string
    global even_odd

    cdef double energy, magnetization, staggered_magnetization, density
    energy = -1.
    magnetization = np.abs(np.sum(spins)) / vol
    staggered_magnetization = np.abs(np.sum(np.multiply(even_odd, np.reshape(spins, (L, L))))) / vol
    density = (np.sum(spins) + 1.) / vol
    return energy, magnetization, staggered_magnetization, density

################################################################################
################################################################################
################################################################################
