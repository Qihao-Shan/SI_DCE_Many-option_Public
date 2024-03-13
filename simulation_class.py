import dm_objects
import arena_class
import numpy as np
import argparse
import time as time_ob
import csv

int_size = 16
float_size = 32


def dm_simulation(dm_method, fill_ratio, param_config, bandwidth, max_step=120000, discre_step=0.1, env_type='random', dist_decay_coeff=1.0, num_run=1):
    # dm_method: dm names
    # param_config: single parameter configuration
    # comm_config: comm_prob, comm_dist, (ballot_size)
    # env_cofiv: random/block fill_ratio num_hypo
    # op: mean error, mean time, mean mess count, fail rate
    start = time_ob.process_time()
    num_color = np.size(fill_ratio)
    fill_ratio_ = fill_ratio / np.sum(fill_ratio)
    if dm_method == 'dbbs':
        num_options = np.shape(dm_objects.make_hypothesis_mat(num_color=num_color, discretization_step=discre_step))[1]
        mess_size = float_size * num_options
        dm_ob = dm_objects.DM_object_BF(num_colors=num_color,
                                        discretization_step=discre_step,
                                        decay_coeff_1=param_config[0],
                                        decay_coeff_2=param_config[1],
                                        comm_prob=bandwidth / float(mess_size))
    elif dm_method == 'lcp':
        mess_size = num_color * float_size
        dm_ob = dm_objects.DM_object_LCP(num_color=num_color,
                                         dm_cycle=param_config[0],
                                         dm_delay=param_config[1],
                                         comm_prob=bandwidth / float(mess_size))
    elif dm_method == 'rv':
        num_options = np.shape(dm_objects.make_hypothesis_mat(num_color=num_color, discretization_step=discre_step))[1]
        ballot_size = max([int(param_config[1] * num_options), 1])
        mess_size = ballot_size * int_size
        dm_ob = dm_objects.DM_object_RV(num_colors=num_color,
                                        discretization_step=discre_step,
                                        evidence_rate=param_config[0],
                                        ballot_size=ballot_size,
                                        comm_prob=bandwidth / float(mess_size))
    else:
        print('Wrong decision-making method')
        return -1
    a = arena_class.arena(fill_ratio=fill_ratio_, dm_object=dm_ob, pattern=env_type, dist_decay_coeff=dist_decay_coeff)
    single_run_error_record = np.array([])
    single_run_scatter_record = np.array([])

    for t in range(max_step):
        a.random_walk_mat()
        if t % 100 == 0:
            a.dm_object.make_decision(a.walk_state_array, a.coo_array)
            if t != 0:
                error, scatter = a.dm_object.compute_error_scatter(fill_ratio_)
            else:
                error = -1
                scatter = -1
            single_run_error_record = np.hstack((single_run_error_record, error))
            single_run_scatter_record = np.hstack((single_run_scatter_record, scatter))

    error_record_valid = single_run_error_record[single_run_error_record >= 0]
    scatter_record_valid = single_run_scatter_record[single_run_scatter_record >= 0]
    min_scatter = np.min(scatter_record_valid)
    error_at_conv = error_record_valid[np.argmin(scatter_record_valid)]
    conv_time = np.argmax(np.logical_and(single_run_scatter_record <= (np.max(single_run_scatter_record) - min_scatter)*0.1 + min_scatter,
                                         single_run_scatter_record > 0))

    print('Metrics:')
    print('error at conv ', error_at_conv, 'conv_time ', conv_time, 'scatter ', min_scatter)
    end = time_ob.process_time()
    print('time ', end-start)

    return error_at_conv, conv_time, min_scatter


env_array = [np.array([0.3, 0.7]),
             np.array([0.9, 0.1]),
             np.array([0.1, 0.3, 0.6]),
             np.array([0.8, 0.1, 0.1]),
             np.array([0.1, 0.2, 0.3, 0.4]),
             np.array([0.7, 0.1, 0.1, 0.1]),
             np.array([0.1, 0.1, 0.2, 0.2, 0.4]),
             np.array([0.6, 0.1, 0.1, 0.1, 0.1])]
dist_decay_coeff = 0.5
bw_array = np.array([8, 16, 32, 64])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-op_file', type=str)
    parser.add_argument('-dm_type', type=str)
    parser.add_argument('-env_type', type=str)
    args = parser.parse_args()

    if args.dm_type == 'lcp':
        param_1_array = np.array([5, 10, 15, 20, 30, 40, 60, 80])
        param_2_array = np.array([10])
    elif args.dm_type == 'rv':
        param_1_array = np.array([0.2, 0.5, 1])
        param_2_array = np.array([0.05, 0.1, 0.2, 0.5])
    elif args.dm_type == 'dbbs':
        param_1_array = np.array([0.5, 0.7, 0.9])
        param_2_array = np.array([0.2, 0.4, 0.6, 0.8])
    else:
        print('Wrong decision-making method')
        param_1_array = np.array([])
        param_2_array = np.array([])

    output_list = []
    for env_id in range(np.shape(env_array)[0]):
        for discre_step in np.array([0.1]):
            for bw in bw_array:
                for param_1 in param_1_array:
                    for param_2 in param_2_array:
                        fill_ratio = env_array[env_id]
                        min_error_record, conv_time_record, scatter_at_conv_record = dm_simulation(
                            dm_method=args.dm_type,
                            fill_ratio=fill_ratio,
                            bandwidth=bw,
                            param_config=[param_1, param_2],
                            env_type=args.env_type,
                            dist_decay_coeff=dist_decay_coeff,
                            discre_step=discre_step)
                        output_list.append(
                            [args.dm_type, env_id, param_1, param_2,
                             bw, 0,
                             min_error_record, conv_time_record, scatter_at_conv_record])

    with open(args.op_file, 'a') as f1:
        writer = csv.writer(f1)
        for i in range(len(output_list)):
            writer.writerow(output_list[i])
        f1.close()







