import pickle
import numpy as np
import torch
import argparse
import os
from OracleSolver.oracles import parall_solve
import scipy.stats

def generate_tsp_data(num, problem_scale,dist='uni'):
    # problem_scale = [20,30,40,50,60,70,80,90,100]
    # generate tsp instances with scale of 20 30 40 50 60 70 80 90 100
    data_list = []
    gt_list = []
    for scale in problem_scale:
        base_inst = np.random.rand(num,scale,2)
        if dist == 'gaussian':
            permu_inst = np.random.normal(0,1,size=(num,scale,2))
            mean_list = np.array([i * .1 + np.random.rand() * .1 for i in range(10)])
            std_list = np.random.rand(10)
            gaussian_mix_prob = np.random.rand(10)
            gaussian_mix_prob = gaussian_mix_prob/np.sum(gaussian_mix_prob)
            select_idx = np.random.choice(len(gaussian_mix_prob),size=scale,p=gaussian_mix_prob)
            select_mean, select_std = mean_list[select_idx].reshape(1,-1,1), std_list[select_idx].reshape(1,-1,1)
            inst = base_inst + (permu_inst * select_std + select_mean)
            inst = (inst - np.min(inst))/(np.max(inst)-np.min(inst))
            gt_list+=parall_solve(inst)[0]
        else:
            inst = base_inst
            gt_list += parall_solve(inst)[0]
        data_list.append(torch.from_numpy(inst).to(torch.float32))
    return data_list, 1, np.array(gt_list)


def generate_cvrp_data(num, problem_scale):
    data_list = []
    gt_list = []
    CAPACITIES = {
        10: 20.,
        20: 30.,
        30: 40.,
        40: 40.,
        50: 40.,
        60: 50.,
        70: 50.,
        80: 50.,
        90: 50.,
        100: 50.
    }
    for scale in problem_scale:
        base_inst = np.random.rand(num,scale+1,2)
        permu_inst = np.random.normal(0,1,size=(num,scale+1,2))

        mean_list = np.array([i * .1 + np.random.rand() * .1 for i in range(10)])
        std_list = np.random.rand(10)
        gaussian_mix_prob = np.random.rand(10)
        gaussian_mix_prob = gaussian_mix_prob/np.sum(gaussian_mix_prob)
        select_idx = np.random.choice(len(gaussian_mix_prob),size=scale+1,p=gaussian_mix_prob)
        select_mean, select_std = mean_list[select_idx].reshape(1,-1,1), std_list[select_idx].reshape(1,-1,1)
        inst = base_inst + (permu_inst * select_std + select_mean)
        inst = (inst - np.min(inst))/(np.max(inst)-np.min(inst))
        demand = np.random.randint(1,10,size=(num, (scale + 1), 1))/ CAPACITIES[scale]
        inst = np.concatenate([inst, demand], axis=-1)
        gt_list+=parall_solve(inst, 'CVRP')[0]
        data_list.append(torch.from_numpy(inst).to(torch.float32))
    return data_list, 1, np.array(gt_list)


def generate_jssp_data(num, problem_scale):

    def permute_rows(x):
        '''
        x is a np array
        '''
        ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
        ix_j = np.random.sample(x.shape).argsort(axis=1)
        return x[ix_i, ix_j]

    def compute_normal_pdf(x,mean,std):
        return 1/np.sqrt(2)/std*np.exp(-(x-mean)**2/std**2)

    data_list = []
    gt_list = []
    for scale in problem_scale:
        n_j, n_m, low, high = scale
        data_list.append([])
        # gt_list.append([])
        for _ in range(num):
            machines = np.expand_dims(np.arange(1, n_m + 1), axis=0).repeat(repeats=n_j, axis=0)
            machines = permute_rows(machines)

            mean_list = np.array([i * .1 + np.random.rand() * .1 for i in range(10)])
            std_list = np.random.rand(10)
            gaussian_mix_prob = np.random.rand(10)
            gaussian_mix_prob = gaussian_mix_prob / np.sum(gaussian_mix_prob)
            select_idx = np.random.choice(len(gaussian_mix_prob), size=high, p=gaussian_mix_prob)
            select_mean, select_std = mean_list[select_idx], std_list[select_idx]
            normal_sample = np.random.normal(0,1,size=(high,))*select_std+select_mean
            normal_prob = compute_normal_pdf(normal_sample, select_mean, select_std)
            p = np.ones(high)/high + normal_prob
            p = p/np.sum(p)
            times = np.random.choice(np.arange(high)+1, size=(n_j, n_m), p=p)
            data_list[-1].append((times,machines))
        gt_list+=(parall_solve(data_list[-1], 'JSSP')[0])
    return data_list, 1, np.array(gt_list)


def generate_op_data(num, problem_scale):
    data_list = []
    gt_list = []
    for scale in problem_scale:
        base_inst = np.random.rand(num, scale + 1, 3)
        permu_inst = np.random.normal(0, 1, size=(num, scale + 1, 3))

        mean_list = np.array([i * .1 + np.random.rand() * .1 for i in range(10)])
        std_list = np.random.rand(10)
        gaussian_mix_prob = np.random.rand(10)
        gaussian_mix_prob = gaussian_mix_prob / np.sum(gaussian_mix_prob)
        select_idx = np.random.choice(len(gaussian_mix_prob), size=scale + 1, p=gaussian_mix_prob)
        select_mean, select_std = mean_list[select_idx].reshape(1, -1, 1), std_list[select_idx].reshape(1, -1, 1)
        inst = base_inst + (permu_inst * select_std + select_mean)
        inst = (inst - np.min(inst)) / (np.max(inst) - np.min(inst))
        inst = torch.from_numpy(inst).to(torch.float32)

        depot = inst[:,0,:2]
        loc = inst[:,1:,:2]
        prize_ = np.linalg.norm(depot[:,None,:]-loc,axis=-1)
        prize = (1 + (prize_ / np.max(prize_, axis=-1, keepdims=True) * 99)) / 100.
        inst[:,1:,-1] = prize
        data_list.append(torch.from_numpy(inst).to(torch.float32))

        gt_list+=parall_solve(inst,'OP')[0]

    return data_list, 1, np.array(gt_list)


def generate_pctsp_det_data(num, problem_scale):
    MAX_LENGTHS = {
        20: 2.,
        30: 3.,
        40: 3.,
        50: 3.,
        60: 4.,
        70: 4.,
        80: 4.,
        90: 4.,
        100: 4.
    }

    data_list = []
    gt_list = []
    for scale in problem_scale:
        base_inst = np.random.rand(num, scale + 1, 5)
        permu_inst = np.random.normal(0, 1, size=(num, scale + 1, 5))

        mean_list = np.array([i * .1 + np.random.rand() * .1 for i in range(10)])
        std_list = np.random.rand(10)
        gaussian_mix_prob = np.random.rand(10)
        gaussian_mix_prob = gaussian_mix_prob / np.sum(gaussian_mix_prob)
        select_idx = np.random.choice(len(gaussian_mix_prob), size=scale + 1, p=gaussian_mix_prob)
        select_mean, select_std = mean_list[select_idx].reshape(1, -1, 1), std_list[select_idx].reshape(1, -1, 1)
        inst = base_inst + (permu_inst * select_std + select_mean)
        inst = (inst - np.min(inst)) / (np.max(inst) - np.min(inst))

        penalty_max = MAX_LENGTHS[scale] * (3) / float(scale)
        penalty = np.random.rand(num, scale) * penalty_max
        deterministic_prize = np.random.rand(num, scale) * 4 / float(scale)
        stochastic_prize = np.random.rand(num, scale) * deterministic_prize * 2

        inst[:,1:,2], inst[:,1:,3], inst[:,1:,4] = penalty, deterministic_prize, stochastic_prize

        gt_list += parall_solve(inst, 'PCTSP_DET')[0]
        data_list.append(torch.from_numpy(inst).to(torch.float32))
    return data_list, 1, np.array(gt_list)


def generate_pctsp_stoch_data(num, problem_scale):
    MAX_LENGTHS = {
        20: 2.,
        30: 3.,
        40: 3.,
        50: 3.,
        60: 4.,
        70: 4.,
        80: 4.,
        90: 4.,
        100: 4.
    }

    data_list = []
    gt_list = []
    for scale in problem_scale:
        base_inst = np.random.rand(num, scale + 1, 5)
        permu_inst = np.random.normal(0, 1, size=(num, scale + 1, 5))

        mean_list = np.array([i * .1 + np.random.rand() * .1 for i in range(10)])
        std_list = np.random.rand(10)
        gaussian_mix_prob = np.random.rand(10)
        gaussian_mix_prob = gaussian_mix_prob / np.sum(gaussian_mix_prob)
        select_idx = np.random.choice(len(gaussian_mix_prob), size=scale + 1, p=gaussian_mix_prob)
        select_mean, select_std = mean_list[select_idx].reshape(1, -1, 1), std_list[select_idx].reshape(1, -1, 1)
        inst = base_inst + (permu_inst * select_std + select_mean)
        inst = (inst - np.min(inst)) / (np.max(inst) - np.min(inst))

        penalty_max = MAX_LENGTHS[scale] * (3) / float(scale)
        penalty = np.random.rand(num, scale) * penalty_max
        deterministic_prize = np.random.rand(num, scale) * 4 / float(scale)
        stochastic_prize = np.random.rand(num, scale) * deterministic_prize * 2

        inst[:,1:,2], inst[:,1:,3], inst[:,1:,4] = penalty, deterministic_prize, stochastic_prize

        gt_list += parall_solve(inst, 'PCTSP_STOCH')[0]
        data_list.append(torch.from_numpy(inst).to(torch.float32))
    return data_list, 1, np.array(gt_list)


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default='TSP',
                        help="Problem, 'TSP', 'CVRP','SDVRP', 'JSSP', 'OP', 'PCTSP_DET', 'PCTSP_STOCH'")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Size of the dataset")
    parser.add_argument('--problem_scale', nargs='+', type=int, help='A list of TSP scale need to evaluate')

    # set problem scale for jssp
    parser.add_argument('--nj', nargs='+', type=int, help='A list of job numbers in JSSP')
    parser.add_argument('--nm', nargs='+', type=int, help='A list of machine numbers in JSSP')
    parser.add_argument('--tl', nargs='+', type=int, help='A list of time-low schedule in JSSP')
    parser.add_argument('--th', nargs='+', type=int, help='A list of time-high schedule in JSSP')

    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()
    np.random.seed(opts.seed)
    problem_scale = opts.problem_scale
    problem_scale = [20,30,40,50,60,70,80,90,100]
    problem = opts.problem
    opts.nj, opts.nm, opts.tl, opts.th = [8,13],[8,13],[1,1],[99,99]

    datadir = os.path.abspath(os.path.join("datasets", problem, 'generated'))
    os.makedirs(datadir, exist_ok=True)

    if problem == 'TSP':
        generate_func = generate_tsp_data
    elif problem == 'CVRP' or problem == 'SDVRP':
        generate_func = generate_cvrp_data
    elif problem == 'JSSP':
        generate_func = generate_jssp_data
        assert len(opts.nj) == len(opts.nm) and len(opts.nj) == len(opts.tl) and len(opts.nj) == len(
            opts.th), 'num of jobs and machines lists should hava equal length'
        problem_scale = list(zip(opts.nj, opts.nm, opts.tl, opts.th))
    elif problem == 'OP':
        generate_func = generate_op_data
    elif problem == 'PCTSP_DET':
        generate_func = generate_pctsp_det_data
    elif problem == 'PCTSP_STOCH':
        generate_func = generate_pctsp_stoch_data
    else:
        NotImplementedError


    for scale in problem_scale:
        print('Begin to generate {}-{}'.format(problem, scale))
        filename = os.path.join(datadir, "{}-{}-{}.pkl".format(
            problem, "{}".format(scale), "{}".format(opts.dataset_size)))
        data, _, gt = generate_func(opts.dataset_size, [scale])
        dataset = {'data': data[0], 'gt': gt}
        save_dataset(dataset, filename)
