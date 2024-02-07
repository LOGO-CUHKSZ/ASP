import numpy as np
import math
import sys
import torch
from scipy.spatial import distance_matrix
import networkx
from numpy.linalg import norm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import os
from subprocess import check_call, check_output
from urllib.parse import urlparse
import random,string
import collections
from ortools.sat.python import cp_model
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from tqdm import tqdm

def run_all_in_pool(func, input, use_multiprocessing=True):
    num_cpus = min(os.cpu_count(),30)
    pool_cls = (Pool if use_multiprocessing and num_cpus>1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(pool.imap(func, input))
    return results


def _calc_insert_cost(D, prv, nxt, ins):
    """
    Calculates insertion costs of inserting ins between prv and nxt
    :param D: distance matrix
    :param prv: node before inserted node, can be vector
    :param nxt: node after inserted node, can be vector
    :param ins: node to insert
    :return:
    """
    return (
        D[prv, ins]
        + D[ins, nxt]
        - D[prv, nxt]
    )

def run_insertion(loc, method='farthest'):
    n = len(loc)
    D = distance_matrix(loc, loc)

    mask = np.zeros(n, dtype=bool)
    tour = []  # np.empty((0, ), dtype=int)
    for i in range(n):
        feas = mask == 0
        feas_ind = np.flatnonzero(mask == 0)
        if method == 'random':
            # Order of instance is random so do in order for deterministic results
            a = i
        elif method == 'nearest':
            if i == 0:
                a = 0  # order does not matter so first is random
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmin()] # node nearest to any in tour
        elif method == 'cheapest':
            assert False, "Not yet implemented" # try all and find cheapest insertion cost

        elif method == 'farthest':
            if i == 0:
                a = D.max(1).argmax()  # Node with farthest distance to any other node
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmax()]  # node which has closest node in tour farthest
        mask[a] = True

        if len(tour) == 0:
            tour = [a]
        else:
            # Find index with least insert cost
            ind_insert = np.argmin(
                _calc_insert_cost(
                    D,
                    tour,
                    np.roll(tour, -1),
                    a
                )
            )
            tour.insert(ind_insert + 1, a)

    cost = D[tour, np.roll(tour, -1)].sum()
    return cost, np.array(tour)


def edges_pair_to_route(edges_pair):
    edges_pair = np.array(edges_pair)-1
    route = []
    route.append(edges_pair[0,0])
    route.append(edges_pair[0,1])
    idx = 0
    while True:
        find_connect_row_idx = np.where(edges_pair== route[-1])[0]
        assert len(find_connect_row_idx)==2, 'invalid solution'
        idx = find_connect_row_idx[find_connect_row_idx != idx][0]
        temp = edges_pair[idx].tolist()
        temp.remove(route[-1])
        route.append(temp[0])
        if len(route) == len(edges_pair):
            return np.array(route)


def get_V_c(instance):
    n = len(instance)
    V = range(1, n + 1)
    c ={}
    dist_mat = norm(instance[None,:,:] - instance[:,None,:],axis=-1)
    for i in V:
        for j in V:
            if j > i:
                c[i, j] = dist_mat[i-1,j-1]
    return V, c


def distance(x1, y1, x2, y2):
    """distance: euclidean distance between (x1,y1) and (x2,y2)"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def write_vrplib(filename, depot, loc, demand, capacity, grid_size=1, name="problem"):
    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "CVRP"),
                ("DIMENSION", len(loc) + 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", capacity)
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x / grid_size * 100000 + 0.5), int(y / grid_size * 100000 + 0.5))  # VRPlib does not take floats
            #"{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate([0] + demand)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "SPECIAL": None,
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def read_vrplib(filename, n):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    tour[tour > n] = 0  # Any nodes above the number of nodes there are is also depot
    assert tour[0] == 0  # Tour should start with depot
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()


def calc_vrp_cost(depot, loc, tour):
    assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
    # TODO validate capacity constraints
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def get_lkh_executable(url="http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.9.tgz"):

    cwd = os.path.abspath(os.path.join("OracleSolver", "lkh"))
    os.makedirs(cwd, exist_ok=True)

    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]

    # if not os.path.isdir(filedir):
    #     print("{} not found, downloading and compiling".format(filedir))

    #     check_call(["wget", url], cwd=cwd)
    #     assert os.path.isfile(file), "Download failed, {} does not exist".format(file)
    #     check_call(["tar", "xvfz", file], cwd=cwd)

    #     assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
    #     check_call("make", cwd=filedir)
    #     os.remove(file)

    executable = os.path.join(filedir, "LKH")
    assert os.path.isfile(executable)
    return os.path.abspath(executable)


def get_solution(data, manager, routing, solution):
    total_distance = 0
    total_route = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0.
        route_load = 0.
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            total_route.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += data['distance_matrix'][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]

        total_distance += route_distance
    return total_distance, total_route


def write_tsplib(filename, loc, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "TSP"),
                ("DIMENSION", len(loc)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))  # tsplib does not take floats
            for i, (x, y) in enumerate(loc)
        ]))
        f.write("\n")
        f.write("EOF\n")


def read_tsplib(filename):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    return tour.tolist()


def calc_tsp_length(loc, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    assert len(tour) == len(loc)
    sorted_locs = np.array(loc)[np.concatenate((tour, [tour[0]]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def solve_tsp_by_lkh(instnace,runs=1):
    executable = get_lkh_executable()
    loc = instnace.tolist()
    rand_string = ''.join(random.sample(string.ascii_letters + string.digits, 10))
    tempdir = os.path.abspath(os.path.join("temp_tsp", rand_string))
    if not os.path.isdir(tempdir):
        os.makedirs(tempdir)

    problem_filename = os.path.join(tempdir, "problem.vrp")
    output_filename = os.path.join(tempdir, "output.tour")
    param_filename = os.path.join(tempdir, "params.par")
    write_tsplib(problem_filename, loc)
    params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": output_filename,"RUNS": runs, "SEED": 1234}
    write_lkh_par(param_filename, params)
    check_output([executable, param_filename])
    tour = read_tsplib(output_filename)
    obj = calc_tsp_length(loc, tour)
    os.system("rm -rf {}".format(tempdir))
    return obj, tour

def solve_cvrp_by_lkh(instance):
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
    executable = get_lkh_executable()
    capacity = CAPACITIES[len(instance)-1]
    depot, loc, demand = instance[0,:2].tolist(), instance[1:,:2].tolist(), (capacity*instance[1:,2]).astype(np.int32).tolist()
    rand_string = ''.join(random.sample(string.ascii_letters + string.digits, 10))
    tempdir = os.path.abspath(os.path.join("temp", rand_string))
    if not os.path.isdir(tempdir):
        os.makedirs(tempdir)
    problem_filename = os.path.join(tempdir, "problem.vrp")
    output_filename = os.path.join(tempdir, "output.tour")
    param_filename = os.path.join(tempdir, "params.par")
    write_vrplib(problem_filename, depot, loc, demand, capacity)
    params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": output_filename}
    write_lkh_par(param_filename, params)
    check_output([executable, param_filename])
    tour = read_vrplib(output_filename, n=len(demand))
    obj = calc_vrp_cost(depot, loc, tour)
    os.system("rm -rf {}".format(tempdir))
    return obj, tour


def solve_op_by_ortools(instance):
    sys_path = os.getcwd() + '/NeuralSolver'
    sys.path.append(sys_path + '/AM')
    from NeuralSolver.TSP.AM.problems.op.op_ortools import solve_op_ortools
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
    depot = instance[0,:2]
    loc = instance[1:,:2]
    prize = instance[1:,-1]
    max_length = MAX_LENGTHS[loc.shape[0]]
    obj, route = solve_op_ortools(depot.tolist(), loc.tolist(), prize.tolist(), max_length)
    return obj, route


def solve_pctsp_det_by_ortools(instance):
    sys_path = os.getcwd() + '/NeuralSolver'
    sys.path.append(sys_path + '/AM')
    from NeuralSolver.TSP.AM.problems.pctsp.pctsp_ortools import solve_pctsp_ortools
    loc = instance[1:, :2]
    depot = instance[0, :2]
    penalty = instance[1:, 2]
    prize = instance[1:, 3]
    min_prize = min(sum(prize), 1.)
    obj, route = solve_pctsp_ortools(depot.tolist(), loc.tolist(), prize.tolist(), penalty.tolist(), min_prize, sec_local_search=0)
    return obj, route


def solve_pctsp_stoch_by_ortools(instance):
    sys_path = os.getcwd() + '/NeuralSolver'
    sys.path.append(sys_path + '/AM')
    from NeuralSolver.TSP.AM.problems.pctsp.pctsp_ortools import solve_pctsp_ortools
    loc = instance[1:, :2]
    depot = instance[0, :2]
    penalty = instance[1:, 2]
    prize = instance[1:, 4]
    min_prize = min(sum(prize), 1.)
    obj, route = solve_pctsp_ortools(depot.tolist(), loc.tolist(), prize.tolist(), penalty.tolist(), min_prize, sec_local_search=0)
    return obj, route


def parall_solve(instances,problem='TSP'):
    if problem == 'TSP':
        if isinstance(instances, torch.Tensor):
            instances = instances.cpu().numpy()
        outputs = run_all_in_pool(solve_tsp_by_lkh, instances)
        # os.system("rm ./*.res")
        # os.system("rm ./*.sol")
        # os.system("rm ./*.pul")
        # os.system("rm ./*.sav")
    elif problem == 'CVRP' or problem == 'SDVRP':
        if isinstance(instances, torch.Tensor):
            instances = instances.cpu().numpy()
        outputs = run_all_in_pool(solve_cvrp_by_lkh, instances)
        # os.system("rm -rf ./temp/")
    # elif problem == 'JSSP':
    #     outputs = run_all_in_pool(solve_jssp_by_ortools, instances)
    #     outputs = [[output,None] for output in outputs]
    elif problem == 'OP':
        if isinstance(instances, torch.Tensor):
            instances = instances.cpu().numpy()
        outputs = run_all_in_pool(solve_op_by_ortools, instances)
    elif problem == 'PCTSP_DET':
        if isinstance(instances, torch.Tensor):
            instances = instances.cpu().numpy()
        outputs = run_all_in_pool(solve_pctsp_det_by_ortools, instances)
    elif problem == 'PCTSP_STOCH':
        if isinstance(instances, torch.Tensor):
            instances = instances.cpu().numpy()
        outputs = run_all_in_pool(solve_pctsp_stoch_by_ortools, instances)
    else:
        raise NotImplementedError

    objs = [output[0] for output in outputs]
    valid_objs = []
    for obj in objs:
        if obj is not None:
            valid_objs.append(obj)
    objs = [obj if obj is not None else np.mean(valid_objs) for obj in objs]
    solns = [output[1] for output in outputs]
    return objs, solns