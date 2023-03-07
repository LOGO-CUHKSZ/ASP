import torch
import numpy as np
import pickle
import argparse
import pandas as pd
import tsplib95
import time
import os

# todo: For providing a complete fair comparison environment, we recommend to do evaluation in the corresponding source
#  codes by loading the prams obtained from ASP


def read_tsplib(filename):
    """
    Read a file in .tsp format into a pandas DataFrame

    The .tsp files can be found in the TSPLIB project. Currently, the library
    only considers the possibility of a 2D map.
    """
    probelm = tsplib95.load(filename).as_name_dict()
    coords = np.array([probelm['node_coords'][i+1] for i in range(probelm['dimension'])])
    optinfo = (np.array(tsplib95.load(filename[:-4]+'.opt.tour').tours)-1).reshape(-1)
    opt_seq_coords = coords[optinfo]
    gt = np.sqrt(np.square(
                opt_seq_coords - np.concatenate([opt_seq_coords[1:], opt_seq_coords[0].reshape(1, -1)], axis=0)).sum(
                -1)).sum()
    cities = pd.DataFrame(
        np.array(coords),
        columns=['y', 'x'],
    )[['x', 'y']]

    norm_factor = max(cities.x.max() - cities.x.min(), cities.y.max() - cities.y.min())
    norm_cities = cities.apply(lambda c: (c - c.min()) / norm_factor)[['x', 'y']].values
    return torch.from_numpy(norm_cities).to(torch.float32).unsqueeze(0), norm_factor, gt

# def read_vrp(filename):
#     probelm = tsplib95.load(filename).as_name_dict()
#     # coords = np.array([probelm['node_coords'][i + 1] for i in range(probelm['dimension'])])
#     dimension = probelm['dimension']
#     coords = np.array([probelm['node_coords'][i + 1] for i in range(probelm['dimension'])])
#     demand = np.array([probelm['demands'][i+1] for i in range(probelm['dimension'])]).tolist()
#     capacity = probelm['capacity']
#     xc = coords[:, 0]
#     yc = coords[:, 1]
#     depot = coords[0]
#
#     gt = calc_cvrp_cost_gurobi(xc,yc, dimension - 1, capacity, depot, coords.tolist(), demand)
#
#     # gt = calc_vrp_cost(depot, loc, tour)
#     cities = pd.DataFrame(
#         np.array(coords),
#         columns=['y', 'x'],
#     )[['x', 'y']]
#     norm_factor = max(cities.x.max() - cities.x.min(), cities.y.max() - cities.y.min())
#     norm_cities = cities.apply(lambda c: (c - c.min()) / norm_factor)[['x', 'y']].values
#
#     return torch.from_numpy(norm_cities).to(torch.float32).unsqueeze(0), norm_factor, gt


def read_generated_data(problem, offset=None):
    datadir = os.path.abspath(os.path.join("datasets", problem, 'generated'))
    file_list = os.listdir(datadir)
    problem_scale = [int(file.split('-')[1]) for file in file_list]
    datasets = [load_dataset(os.path.join(datadir,file)) for file in file_list]
    data_list = []
    gt_list = []
    for dataset in datasets:
        if offset is not None:
            data_list.append(dataset['data'][:offset].cuda())
            gt_list.append(dataset['gt'][:offset])
        else:
            data_list.append(dataset['data'].cuda())
            gt_list.append(dataset['gt'])
    return data_list, gt_list, problem_scale


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_result(result, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


def dict_to_argparse(dict):
    parser = argparse.ArgumentParser()
    for k in dict.keys():
        parser.add_argument('--' + k, type=type(dict[k]), default=dict[k])
    args = parser.parse_args()
    return args


def eval_time_record(eval, dataset, problem, solver, opts):
    duration = {}
    pred = []
    for _, per_scale_data in enumerate(dataset):
        try:
            scale = tuple(opts.problem_scale[_])
        except:
            scale = opts.problem_scale[_]
        per_scale_dataset = []
        per_scale_dataset.append(per_scale_data)
        start_time = time.time()
        try:
            pred_val = eval(problem, solver, per_scale_dataset, opts).cpu().numpy().tolist()
        except:
            pred_val = eval(problem, solver, per_scale_data, opts, scale).cpu().numpy().tolist()
        for i in pred_val:
            pred.append(i)

        end_time = time.time() - start_time
        duration[scale] = end_time

    return np.array(pred), duration



def eval(config):
    # set the dataset
    if config.problem == 'TSP':
        if config.real_ds:
            real_ds_list = []
            for file in os.listdir('./datasets/TSP/tsplib'):
                if file.split('.')[-1] == 'tsp':
                    real_ds_list.append('./datasets/TSP/tsplib/' + file)
            dataset, norm_factor, gt = zip(*[read_tsplib(file) for file in real_ds_list])
        else:
            dataset, gt, problem_scale = read_generated_data(config.problem)
            dataset = [data[:config.offset] for data in dataset]
            gt = [data[:config.offset] for data in gt]
            gt = np.concatenate(gt).tolist()
            norm_factor = 1
            eval_num = len(dataset)
    elif config.problem == 'CVRP':
        if config.real_ds:
            real_ds_list = []
            for file in os.listdir('./datasets/CVRP/tsplib_cvrp'):
                if file.split('.')[-1] == 'vrp':
                    real_ds_list.append('./datasets/CVRP/tsplib_cvrp/' + file)
            # dataset, norm_factor, gt = zip(*[read_vrp(file) for file in real_ds_list])
        else:
            dataset, gt, problem_scale = read_generated_data(config.problem)
            dataset = [data[:config.offset] for data in dataset]
            gt = [data[:config.offset] for data in gt]
            gt = np.concatenate(gt).tolist()
            norm_factor = 1
            eval_num = len(dataset)
    elif config.problem == 'SDVRP':
        if config.real_ds:
            real_ds_list = []
            for file in os.listdir('./datasets/CVRP/tsplib_cvrp'):
                if file.split('.')[-1] == 'vrp':
                    real_ds_list.append('./datasets/CVRP/tsplib_cvrp/' + file)
            # dataset, norm_factor, gt = zip(*[read_vrp(file) for file in real_ds_list])
        else:
            dataset, gt, problem_scale = read_generated_data(config.problem)
            dataset = [data[:config.offset] for data in dataset]
            gt = [data[:config.offset] for data in gt]
            gt = np.concatenate(gt).tolist()
            norm_factor = 1
            eval_num = len(dataset)
    elif config.problem == 'OP':
        if config.real_ds:
            ...
        else:
            dataset, gt, problem_scale = read_generated_data(config.problem)
            dataset = [data[:config.offset] for data in dataset]
            gt = [data[:config.offset] for data in gt]
            gt = np.concatenate(gt).tolist()
            norm_factor = 1
            eval_num = len(dataset)
    elif config.problem == 'PCTSP_DET':
        if config.real_ds:
            ...
        else:
            dataset, gt, problem_scale = read_generated_data(config.problem)
            dataset = [data[:config.offset] for data in dataset]
            gt = [data[:config.offset] for data in gt]
            gt = np.concatenate(gt).tolist()
            norm_factor = 1
            eval_num = len(dataset)
    elif config.problem == 'PCTSP_STOCH':
        if config.real_ds:
            ...
        else:
            dataset, gt, problem_scale = read_generated_data(config.problem)
            dataset = [data[:config.offset] for data in dataset]
            gt = [data[:config.offset] for data in gt]
            gt = np.concatenate(gt).tolist()
            norm_factor = 1
            eval_num = len(dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config.problem!='JSSP':
        dataset = [d.to(device) for d in dataset]
    # load mix-solver
    game_config = config.pretrained_gameinfo + '/game_info'
    loop_num = []
    for pt in os.listdir(game_config):
        psro_loop = int(pt.split('.')[0].split('_')[-1])
        loop_num.append(psro_loop)
    max_loop = min(max(loop_num), 20)
    game_config_pt = torch.load(game_config + '/game_info_{}.pt'.format(max_loop))
    mix_prob = game_config_pt['meta_strategy'][0]
    param_list = game_config_pt['policy'][0]
    with open(os.path.join(config.pretrained_gameinfo, 'args.json'), 'r') as f:
        model_args_dict = json.load(f)
    model_args = dict_to_argparse(model_args_dict)
    sys_path = os.getcwd() + '/NeuralSolver'
    other_type_routing = ['SDVRP', 'OP', 'PCTSP_DET', 'PCTSP_STOCH']
    if config.problem == 'TSP':
        sys_path += '/TSP'
        if config.method == 'AM':
            sys.path.append(sys_path + '/AM')
            from NeuralSolver.TSP.AM.model_func import initialize, eval
            from NeuralSolver.TSP.AM.options import get_options

        elif config.method == 'POMO':
            sys.path.append(sys_path + '/POMO')
            from NeuralSolver.TSP.POMO.model_func import initialize, eval
            from NeuralSolver.TSP.POMO.options import get_options

    elif config.problem == 'CVRP':
        sys_path += '/CVRP'
        if config.method == 'AM':
            sys_path = os.getcwd() + '/NeuralSolver/TSP'
            sys.path.append(sys_path + '/AM')
            from NeuralSolver.TSP.AM.model_func import initialize, eval
            from NeuralSolver.TSP.AM.options import get_options

        elif config.method == 'POMO':
            sys.path.append(sys_path + '/POMO')
            from NeuralSolver.CVRP.POMO.model_func import initialize, eval
            from NeuralSolver.CVRP.POMO.options import get_options
    elif config.problem in other_type_routing:
        sys_path += '/TSP'
        sys_path = os.getcwd() + '/NeuralSolver/TSP'
        sys.path.append(sys_path + '/AM')
        from NeuralSolver.TSP.AM.model_func import initialize, eval
        from NeuralSolver.TSP.AM.options import get_options

    else:
        NotImplementedError

    duration = {}
    if config.baseline_mode:
        opts = get_options()
        # if config.problem == 'CVRP':
        #     opts.problem = 'CVRP'
        # if config.problem == 'SDVRP':
        #     opts.problem = 'SDVRP'
        opts.problem = config.problem
        opts.problem_scale= problem_scale
        problem, solver, _, _, _ = initialize(opts,True)
        # pred = eval(problem, solver, dataset, opts).cpu().numpy()
        pred, duration = eval_time_record(eval, dataset, problem, solver, opts)
    else:
        model_args.device = device
        model_args.problem_scale = problem_scale
        problem = initialize(model_args)[0]
        solver = get_mix_solver(model_args, mix_prob, param_list)
        # pred = eval(problem, solver, dataset, model_args).cpu().numpy()
        pred, duration = eval_time_record(eval, dataset, problem, solver, model_args)

    if config.problem == 'OP':
        per_gap = np.abs(pred * norm_factor / gt - 1) * 100
    else:
        per_gap = (pred * norm_factor / gt - 1) * 100
    gap = np.mean(per_gap)


    if config.real_ds:
        out_file = 'eval_results/{}/real/{}-mix-{}.pkl'.format(config.problem, config.method,
                                                                     not config.baseline_mode)
        per_inst_res = [(file.split('/')[-1], [gt[i], pred[i]*norm_factor[i], per_gap[i]]) for i, file in enumerate(real_ds_list)]

        results = {
                    'gt, pred and gap of per instance': dict(per_inst_res),
                    'average gap': gap,
                    'duration': duration}
    else:
        out_file = 'eval_results/{}/generate/{}-mix-{}.pkl'.format(config.problem, config.method,
                                                                       not config.baseline_mode)
        results = {'gap of per scale': [('{}{}'.format(config.problem, problem_scale[i]),per_gap.reshape(eval_num, -1).mean(-1)[i]) for i in range(len(problem_scale))],
                   'average gap': gap,
                   'duration': duration}
    print(results)
    # save_result(results, out_file)


if __name__=="__main__":
    import sys
    import json

    parser = argparse.ArgumentParser()

    # set the solver
    parser.add_argument('--problem', type=str, default='PCTSP_STOCH')
    parser.add_argument('--method', type=str, default='AM')
    parser.add_argument('--baseline_mode', type=bool, default=False)

    # load pretrained game information
    parser.add_argument('--pretrained_gameinfo', help='load game information')

    # set the dataset
    parser.add_argument('--real_ds', type=bool, default=False)
    parser.add_argument('--offset', type=int, default=1000)


    config = parser.parse_args()
    path = '/mnt/data1/wangchenguang/PSRO-CO'

    if config.problem == 'CVRP':
        if config.method=='POMO':
            # config.pretrained_gameinfo = path + '/save_game/CVRP/POMO/bright-lion-52'
            config.pretrained_gameinfo = path + '/save_game/CVRP/POMO/clear-music-118'
        elif config.method=='AM':
            config.pretrained_gameinfo = path + '/save_game/CVRP/AM/eager-water-121'

    elif config.problem == 'TSP':
        if config.method=='POMO':
            # config.pretrained_gameinfo = './save_game/TSP/POMO/giddy-snow-50'
            config.pretrained_gameinfo = path + '/save_game/TSP/POMO/light-deluge-117'
        elif config.method=='AM':
            config.pretrained_gameinfo = path + '/save_game/TSP/AM/visionary-elevator-98'


    elif config.problem == 'SDVRP':
        if config.method=='AM':
            config.pretrained_gameinfo = path + '/save_game/SDVRP/AM/silvery-energy-86'
            # config.pretrained_gameinfo = path + '/save_game/SDVRP/AM/unique-star-120'
    elif config.problem == 'OP':
        if config.method=='AM':
            config.pretrained_gameinfo = path + '/save_game/OP/AM/comic-blaze-57'
            # config.pretrained_gameinfo = path + '/save_game/OP/AM/stellar-jazz-108'

    elif config.problem == 'PCTSP_DET':
        if config.method=='AM':
            # config.pretrained_gameinfo = path + '/save_game/PCTSP_DET/AM/winter-glitter-74'
            config.pretrained_gameinfo = path + '/save_game/PCTSP_DET/AM/stilted-lake-106'

    elif config.problem == 'PCTSP_STOCH':
        if config.method=='AM':
            config.pretrained_gameinfo = path + '/save_game/PCTSP_STOCH/AM/hardy-wood-75'
            # config.pretrained_gameinfo = path + '/save_game/PCTSP_STOCH/AM/eager-haze-105'

    elif config.problem == 'JSSP':
        config.pretrained_gameinfo = path + '/save_game/JSSP/L2D/toasty-rain-92'

    print("baseline_mode:", config.baseline_mode)
    print("game_info:", config.pretrained_gameinfo)
    eval(config)










