from config.config_Solver import combine_Solver_configs
from game_utils import ASP
import warnings
import numpy as np
import torch
import random

warnings.filterwarnings('ignore')

def set_random_seed(seed=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--task_description',type=str, default='None',
                        help='description of task')

    # set training paradigm
    parser.add_argument('--train_from_scratch',default=True,)
    parser.add_argument('--not_use_task_selection', type=bool, default=False,
                        help='use task selection or not in curriculum learning')
    parser.add_argument('--not_use_std', type=bool, default=False,
                        help='use history performance for stability or not in curriculum learning')
    # set game
    parser.add_argument('--problem_scale_list', nargs='+', default=None,
                        help='preset problem scale list')
    parser.add_argument('--training_status', type=str, default=None,
                        help='begin with psro or AS')
    parser.add_argument('--problem_scale_start', type=int, default=20,
                        help='curriculum learning start with this problem scale')
    parser.add_argument('--problem_scale_step', type=int, default=20, help='incremental for the next problem scale')
    parser.add_argument('--problem_scale_end', type=int, default=100,
                        help='the largest problem scale we want to handle')
    parser.add_argument('--performance_thres', type=float, default=1)
    parser.add_argument('--keep_performance_thres', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=5)

    parser.add_argument('--iter_num', type=int, default=100, help='maximal training loop')
    parser.add_argument('--psro_loop', type=int, default=5, help='psro loop for each problem scale')
    parser.add_argument('--AS_loop', type=int, default=1, help='AS loop for each problem scale')

    parser.add_argument('--load_resume', default=None, help='Resume from previous ASP training')
    parser.add_argument('--create_dir', action='store_true', help='whether creating a log dir when beginning to train')
    parser.add_argument('--save_path', default='./')


    # eval the game
    parser.add_argument('--eval_num', type=int, default=1)
    parser.add_argument('--eval_mode', type=str, default='gt', help='gt: using oracle solver; mix: using mix-solver')

    # set solver
    parser.add_argument('--train_solver_only', action='store_true', help='True when we do curriculum on uniform')
    parser.add_argument('--problem', type=str, default='TSP',
                        help='TSP, CVRP, SDVRP, OP, PCTSP_DET, PCTSP_STOCH')
    parser.add_argument('--method', type=str, default='POMO')
    parser.add_argument('--solver_epochs', type=int, default=5,
                        help='The number of epochs to train')
    parser.add_argument('--num_batch', type=int, default=1,
                        help='The number of batches to fintune')
    parser.add_argument('--solver_val_size', type=int, default=10000,
                        help='The number of epochs to train')
    parser.add_argument('--solver_n_encode_layers', type=int, default=6,)

    parser.add_argument('--offset_test', default=100,
                        help='The number of epochs to train')

    # set data generator
    parser.add_argument('--dg_epochs', type=int, default=100,
                        help='The number of epochs to train')
    parser.add_argument('--dg_lr', type=float, default=1e-4)
    parser.add_argument('--dg_wd', type=float, default=1e-5,)
    parser.add_argument('--dg_train_batch', type=int, default=1280,)
    parser.add_argument('--dg_eval_batch', type=int, default=1000,)
    parser.add_argument('--dg_nf_layer', type=int, default=5,
                        help='Number of layers of Normalizing Flows')

    # set wandb recording
    parser.add_argument('--log_to_wandb', action='store_true')

    parser.add_argument('--seed', type=int, default=1234)


    config = combine_Solver_configs(parser)

    # config.problem_scale_list = [20,40,60,80,100]
    # config.training_status = 'AS'
    # config.not_use_task_selection = True
    # config.train_solver_only = True

    seed = np.random.randint(0,10000)
    set_random_seed(seed)

    # config.task_description = 'test'
    # config.train_from_scratch = True
    # config.create_dir = True
    # config.performance_thres = float(config.performance_thres)
    # config.problem_scale_start = float(config.problem_scale_start)
    # config.problem_scale_end = float(config.problem_scale_end)

    # config.task_description = 'fig3.1-pomo-tsp-nostd-1'

    # config.training_status = 'AS'
    # config.train_solver_only = True
    # config.train_from_scratch=True
    # config.not_use_task_selection=True
    asp = ASP(config)
    asp.train_asp()