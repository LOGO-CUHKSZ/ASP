from open_spiel.python.algorithms import lp_solver
from Agent import Agent_DG, Agent_Solver
import numpy as np
from copy import deepcopy
import torch
import itertools
import pyspiel
from evaluation import dict_to_argparse, read_generated_data
from OracleSolver.oracles import parall_solve
import wandb
import os, json
import time
from torch.distributions.uniform import Uniform


def empty_list_generator(num_dimensions):
    result = []
    for _ in range(num_dimensions - 1):
        result = [result]
    return result


class PSRO:
    def __init__(self, config):
        self.symmetric_game = False
        self.Oracle_Solver = parall_solve  # set Oracle solver to get ground-truth
        self._meta_solver = lp_solver
        self._num_players = 2
        self.config = config

    def init(self, problem_scale, solver_agent=None):
        self.config = deepcopy(self.config)
        self.config.problem_scale = [problem_scale]
        self.DG = Agent_DG(self.config)
        if solver_agent is None:
            self.SS = Agent_Solver(self.config)
        else:
            self.SS = solver_agent
            self.SS.update_config(self.config)
        self._iterations = 0
        self.eval_result = []
        self.pop_row_idx = [0]
        self.pop_col_idx = [0]
        self.generated_data = []
        self._initialize_policy(problem_scale)
        self._initialize_game_state()
        self.update_meta_strategies()
        eval_gap = self.eval_solver()
        self.eval_result.append(eval_gap)

    def _initialize_policy(self, problem_scale):
        self.problem_scale = [problem_scale]
        self._policies = [[] for k in range(self._num_players)]
        self._new_policies = [[self.SS.solver_agent_info()], [self.DG.oracle]]
        val_dataset = self.DG.sample_instance(self.DG.oracle, self.problem_scale[0],
                                              self.config.eval_num)
        gt = self.Oracle_Solver(val_dataset, self.config.problem)[0]
        self.generated_data.append([val_dataset, np.array(gt)])

    def _initialize_game_state(self):
        self._meta_payoff = [
            np.array(empty_list_generator(self._num_players))
            for _ in range(self._num_players)
        ]
        self.update_empirical_gamestate(seed=None)

    def update_agents(self, logger=None):  # Generate new, Best Response agents via oracle.
        meta_strategy_SS = list(self._meta_strategy[0])
        meta_strategy_DG = list(self._meta_strategy[1])

        if not self.config.train_solver_only:
            param_SS_list = self._policies[0]
            solver = deepcopy(self.SS.model)
            eval_func = self.SS.get_solver
            self.DG.train_oracle(solver, eval_func, meta_strategy_SS, param_SS_list, [1.], train_from_scratch=True,
                                 logger=logger)

        marginal_dist = torch.matmul(torch.tensor(meta_strategy_DG).unsqueeze(1),
                                     torch.tensor([1.]).unsqueeze(
                                         0).to(torch.float32))  # tensor: num_DG_polcy x num_PS_polcy
        sampler = lambda judge: self.sampler(marginal_dist, train=judge)
        self.SS.train_oracle(sampler=sampler, train_from_scratch=False, logger=logger)
        if not self.config.train_solver_only:
            self._new_policies = [[self.SS.solver_agent_info()], [self.DG.oracle]]
            val_dataset = self.DG.sample_instance(self.DG.oracle, self.problem_scale[0],
                                                  self.config.eval_num)
            gt = self.Oracle_Solver(val_dataset, self.config.problem)[0]
            self.generated_data.append([val_dataset, np.array(gt)])
        else:
            self._new_policies = [[self.SS.solver_agent_info()], []]

    def update_empirical_gamestate(self, seed):  # Update gamestate matrix by mix-sovler
        """Given new agents in _new_policies, update meta_games through simulations.
                Args:
                  seed: Seed for environment generation.
                Returns:
                  Meta game payoff matrix.
        """
        if seed is not None:
            np.random.seed(seed=seed)
            torch.manual_seed(seed=seed)

            # Concatenate both lists.
        updated_policies = [
            self._policies[k] + self._new_policies[k]
            for k in range(self._num_players)
        ]

        # Each metagame will be (num_strategies)^self._num_players.
        # There are self._num_player metagames, one per player.
        total_number_policies = [
            len(updated_policies[k]) for k in range(self._num_players)
        ]
        number_older_policies = [
            len(self._policies[k]) for k in range(self._num_players)
        ]
        number_new_policies = [
            len(self._new_policies[k]) for k in range(self._num_players)
        ]

        # Initializing the matrix with nans to recognize unestimated states.
        meta_payoff = [
            np.full(tuple(total_number_policies), np.nan)
            for k in range(self._num_players)
        ]

        # Filling the matrix with already-known values.
        older_policies_slice = tuple(
            [slice(len(self._policies[k])) for k in range(self._num_players)])

        for k in range(self._num_players):
            meta_payoff[k][older_policies_slice] = self._meta_payoff[k]

        for current_player in range(self._num_players):
            # Only iterate over new policies for current player ; compute on every
            # policy for the other players.
            range_iterators = [
                                  range(total_number_policies[k]) for k in range(current_player)
                              ] + [range(number_new_policies[current_player])] + [
                                  range(total_number_policies[k])
                                  for k in range(current_player + 1, self._num_players)
                              ]
            for current_index in itertools.product(*range_iterators):
                used_index = list(current_index)
                used_index[current_player] += number_older_policies[current_player]
                if np.isnan(meta_payoff[current_player][tuple(used_index)]):
                    estimated_policies = [
                                             updated_policies[k][current_index[k]]
                                             for k in range(current_player)
                                         ] + [
                                             self._new_policies[current_player][current_index[current_player]]
                                         ] + [
                                             updated_policies[k][current_index[k]]
                                             for k in range(current_player + 1, self._num_players)
                                         ]
                    val_dataset, gt = self.generated_data[used_index[1]]

                    neural_solver = self.SS.load_solver_agent_info(estimated_policies[0], 'training').model
                    pred = self.SS.eval_solver(neural_solver, val_dataset, self.problem_scale[0])
                    try:
                        pred = pred.cpu().numpy()
                    except:
                        pass
                    if self.config.problem == 'OP':
                        mean_gap = np.abs((pred / np.array(gt) - 1) * 100).mean()
                    else:
                        mean_gap = ((pred / np.array(gt) - 1) * 100).mean()
                    utility_estimates = [-mean_gap, mean_gap, mean_gap]
                    for k in range(self._num_players):
                        meta_payoff[k][tuple(used_index)] = utility_estimates[k]
        self._meta_payoff = meta_payoff
        self._policies = updated_policies
        return meta_payoff

    def update_meta_strategies(self, ):  # Compute meta strategy (e.g. Nash)
        # self.update_sampler()

        p0_sol, p1_sol, _, _ = (
            self._meta_solver.solve_zero_sum_matrix_game(
                pyspiel.create_matrix_game(
                    self._meta_payoff[0],
                    self._meta_payoff[1])))
        self._meta_strategy = [p0_sol, p1_sol]

    def get_meta_game(self):
        """Returns the meta game matrix."""
        return self._meta_payoff

    def sampler(self, marginal_dist, train=True):
        dataset = []
        if train:
            batch_size = self.config.solver_epoch_size
        else:
            batch_size = self.config.solver_val_size

        for j in range(marginal_dist.shape[1]):
            data_in_same_scale = []
            for i in range(marginal_dist.shape[0]):
                dg_param = self._policies[1][i]
                data = self.DG.sample_instance(dg_param, self.problem_scale[j],
                                               max(int(batch_size * marginal_dist[i, j]), 1))
                data_in_same_scale.append(data)
            dataset.append(torch.cat(data_in_same_scale, dim=0))
        return dataset

    def eval_solver(self):
        dataset, gt, problem_scale = read_generated_data(self.config.problem, self.config.offset_test)

        sample_size = len(gt[0])
        gt = np.concatenate(gt)
        norm_factor = 1
        param_list = self._policies[0]
        solver = deepcopy(self.SS.model)
        solver.load_state_dict({**solver.state_dict(), **param_list[-1]['model']})
        pred = self.SS.eval_solver(solver, dataset, problem_scale)
        if self.config.problem == 'OP':
            gap = np.abs(pred.cpu().numpy() * norm_factor / gt - 1) * 100
        else:
            gap = (pred.cpu().numpy() * norm_factor / gt - 1) * 100
        gap = gap.reshape(-1, sample_size).mean(-1)
        eval_gap = {}
        for i, s in enumerate(problem_scale):
            eval_gap[s] = gap[i]
        return eval_gap

    def iteration(self, logger, seed=None):
        """Main trainer loop.
                Args:
                  seed: Seed for random BR noise generation.
                """
        self._iterations += 1
        self.update_agents(logger)  # Generate new, Best Response agents via oracle.
        self.update_empirical_gamestate(seed=seed)  # Update gamestate matrix.
        self.update_meta_strategies()  # Compute meta strategy (e.g. Nash)
        meta_payoff1 = self._meta_payoff[0]
        meta_payoff2 = self._meta_payoff[1]
        self.pop_row_idx.append(np.argmax(meta_payoff1 @ np.array(list(self._meta_strategy[1])).reshape(-1, 1)).item())
        self.pop_col_idx.append(np.argmax(np.array(list(self._meta_strategy[0])).reshape(1, -1) @ meta_payoff2).item())

        eval_gap = self.eval_solver()
        self.eval_result.append(eval_gap)

    def train_psro(self, performance_log, logger=None, all_eval_steps=None, psro_eval_steps=None):
        for i in range(self.config.psro_loop):
            self.iteration(logger)
        total_gap = []
        all_total_gap = []
        for dic in self.eval_result:
            temp = []
            all_temp = []
            for key in dic.keys():
                if logger is not None:
                    logger.log({'All Evalution Process/Evaluation Gap on scale: {}'.format(key): dic[key],
                                'epoch': all_eval_steps})
                all_temp.append(dic[key])
                if key in performance_log.keys():
                    if logger is not None:
                        logger.log({'PSRO/Evaluation Gap on scale: {}'.format(key): dic[key], 'epoch': psro_eval_steps})
                    temp.append(dic[key])
                    performance_log[key].append(dic[key])
            all_eval_steps += 1
            psro_eval_steps += 1

            total_gap.append(np.mean(temp))
            all_total_gap.append(np.mean(all_temp))

        idx = np.argmin(np.array(total_gap))
        self.SS.load_solver_agent_info(self._policies[0][idx], 'training')
        return self.SS, {'mix_prob': list(self._meta_strategy[1]), 'dist_param': self._policies[1],
                         'data_gt': self.generated_data}, total_gap[idx], all_total_gap[idx]


class AS:
    def __init__(self, config, C=None):
        self.config = config
        self.old_C = None
        self.C = C
        self.hist_eval = {}

    def init(self, problem_scale, dist, solver_agent=None):
        if solver_agent is None:
            self.solver_agent = Agent_Solver(self.config)
        else:
            self.solver_agent = solver_agent

        self.problem_scale = problem_scale
        for s in problem_scale:
            self.hist_eval.setdefault(s, [])
        self.dist = dist
        self.config.problem_scale = problem_scale
        self.dg_agent = Agent_DG(self.config)

    def momentum_AS(self, performance_diff, hist_std):
        if not self.config.not_use_task_selection:
            performance_diff = performance_diff.reshape(-1, 1)
            hist_std = hist_std.reshape(-1, 1)
            if not self.config.not_use_std:
                prob = (performance_diff + hist_std) / 2
            else:
                prob = performance_diff
        else:
            prob = np.ones(len(self.problem_scale)).reshape(-1,1)
        return prob / prob.sum()

    def get_cost_vector(self, ):
        cost_vec = np.full((len(self.problem_scale), 1), np.nan)
        for i in range(len(self.problem_scale)):
            cost_vec[i, 0] = self.eval_func(self.problem_scale[i], self.dist[i])
            self.hist_eval[self.problem_scale[i]].append(cost_vec[i, 0])
        return cost_vec

    def train_AS(self, performance_log, logger=None, all_eval_steps=None, AS_eval_steps=None):
        eval_res = self.eval_solver()
        best_eval = []
        all_best_eval = []
        for key in eval_res.keys():
            all_best_eval.append(eval_res[key])
            if key in self.problem_scale:
                best_eval.append(eval_res[key])
        hist_std = np.array([np.std(v[-10:]) for v in performance_log.values()])
        hist_std /= np.sum(hist_std)
        hist_std[np.isnan(hist_std)] = 1
        hist_std /= np.sum(hist_std)
        performance_diff = []
        for _ in range(len(hist_std)):
            performance_diff.append((best_eval[_] - self.config.performance_thres) / self.config.performance_thres)

        performance_diff = np.array(performance_diff)
        performance_diff[performance_diff < 0] = 0
        print('performance diff:', (performance_diff).reshape(-1))
        if not (performance_diff == 0).all():
            performance_diff = performance_diff / np.sum(performance_diff)
        best_eval = np.mean(np.array(best_eval))
        all_best_eval = np.mean(np.array(all_best_eval))
        for i in range(self.config.AS_loop):
            marginal_dist = self.momentum_AS(performance_diff, hist_std)
            temp_solver_agent_info = self.train_func(marginal_dist)
            eval_res = self.eval_solver()
            eval_temp = []
            all_eval_temp = []
            for key in eval_res.keys():
                if logger is not None:
                    logger.log({'All Evalution Process/Evaluation Gap on scale: {}'.format(key): eval_res[key],
                                'epoch': all_eval_steps})
                    all_eval_steps += 1
                all_eval_temp.append(eval_res[key])
                if key in self.problem_scale:
                    if logger is not None:
                        logger.log(
                            {'AS/Evaluation Gap on scale: {}'.format(key): eval_res[key], 'epoch': AS_eval_steps})
                        AS_eval_steps += 1

                    eval_temp.append(eval_res[key])
                    performance_log[key].append(eval_res[key])

            hist_std = np.array([np.std(v[-10:]) for v in performance_log.values()])
            hist_std /= np.sum(hist_std)
            hist_std[np.isnan(hist_std)] = 1
            hist_std /= np.sum(hist_std)
            performance_diff = []
            for _ in range(len(hist_std)):
                performance_diff.append((eval_temp[_] - self.config.performance_thres) / self.config.performance_thres)
            performance_diff = np.array(performance_diff)
            performance_diff[performance_diff < 0] = 0
            if not (performance_diff == 0).all():
                performance_diff = performance_diff / np.sum(performance_diff)
            eval_temp = np.mean(np.array(eval_temp))
            all_eval_temp = np.mean(np.array(all_eval_temp))

            best_eval = eval_temp
            all_best_eval = all_eval_temp

        return self.solver_agent, best_eval, all_best_eval

    def train_func(self, marginal_dist):
        sampler = lambda judge: self.sampler(marginal_dist, train=judge)
        self.solver_agent.train_oracle(sampler=sampler, train_from_scratch=False)
        return self.solver_agent.solver_agent_info()

    def eval_func(self, scale, dist):
        self_generated_data_gt = dist['data_gt']
        mix_prob = dist['mix_prob']
        num_data = self_generated_data_gt[0][0].shape[0]
        num_data_per_dist = (np.array(mix_prob) * num_data).astype(int)
        num_data_per_dist[num_data_per_dist == 0] = 1
        data_list = []
        gt_list = []
        for i in range(len(mix_prob)):
            idx = np.random.choice(np.arange(num_data), num_data_per_dist[i], False)
            data_list.append(self_generated_data_gt[i][0][idx])
            gt_list.append(self_generated_data_gt[i][1][idx])
        data = torch.cat(data_list, 0)
        gt = np.concatenate(gt_list, 0)
        pred = self.solver_agent.eval_solver(self.solver_agent.model, data, None).cpu().numpy()
        gap = (pred / gt - 1) * 100
        return gap.mean()

    def sampler(self, marginal_dist, train=True):
        dataset = []
        if train:
            batch_size = self.config.solver_epoch_size
        else:
            batch_size = self.config.solver_val_size
        for i in range(marginal_dist.shape[0]):
            mix_dist = self.dist[i]
            data = self.dg_agent.sample_mix_dist(mix_dist, self.problem_scale[i],
                                                 max(int(batch_size * marginal_dist[i, -1]), 1))

            dataset.append(data)
        return dataset

    def eval_solver(self):
        dataset, gt, problem_scale = read_generated_data(self.config.problem, self.config.offset_test)

        sample_size = len(gt[0])
        gt = np.concatenate(gt)
        norm_factor = 1

        solver = self.solver_agent.model

        pred = self.solver_agent.eval_solver(solver, dataset, problem_scale)
        if self.config.problem == 'OP':
            gap = np.abs(pred.cpu().numpy() * norm_factor / gt - 1) * 100
        else:
            gap = (pred.cpu().numpy() * norm_factor / gt - 1) * 100
        gap = gap.reshape(-1, sample_size).mean(-1)
        eval_gap = {}
        for i, s in enumerate(problem_scale):
            eval_gap[s] = gap[i]
        return eval_gap


class ASP:
    def __init__(self, config, ):
        if config.load_resume is not None:
            asp_info = torch.load(config.load_resume)
            args_path = '/'.join(config.load_resume.split('/')[:-1])
            name = config.load_resume.split('/')[-2]
            with open(os.path.join(args_path, 'args.json'), 'r') as f:
                dict_param = json.load(f)
                resume_config = dict_to_argparse(dict_param)
                if not config.keep_performance_thres:
                    resume_config.performance_thres = config.performance_thres
                config = resume_config
            self.psro = PSRO(config, )
            try:
                self.psro._meta_strategy = asp_info['psro_info']['meta_strategy']
                self.psro._meta_payoff = asp_info['psro_info']['meta_payoff']
            except:
                self.psro._meta_strategy = None
                self.psro._meta_payoff = None

            self.ps_list = asp_info['problem_scale_list']
            self.resume_solver_agent = Agent_Solver(config).load_solver_agent_info(asp_info['solver_param'])
            self.mix_dist_list = asp_info['mix_dist_list']
            self.his_performance = asp_info['his_performance']
            C = asp_info['cost_mat']
            self.eval_res_list = asp_info['eval_res']
            self.all_eval_res_list = asp_info['all_eval_res']

            self.count = asp_info['count']
            config.performance_thres = float(config.performance_thres)
            config.eval_num = int(config.eval_num)
            config.psro_loop = int(config.psro_loop)

            if self.eval_res_list[-1] < config.performance_thres and self.ps_list[-1] < config.problem_scale_end:
                self.status = 'psro'
                self.ps_list += [self.ps_list[-1] + config.problem_scale_step]
            else:
                self.status = 'AS'

            self.iterations = asp_info['iterations']
            self.resume = True
            self.all_eval_steps = asp_info['all_eval_steps']
            self.psro_eval_steps = asp_info['psro_eval_steps']
            self.AS_eval_steps = asp_info['AS_eval_steps']
            self.best_eval = asp_info['best_eval']
            self.all_best_eval = asp_info['all_best_eval']
            self.time_cost = asp_info['time_cost']

            if config.log_to_wandb:
                wandb.init(
                    name=name+'-resume',
                    group='Combine-Agent-ts:{}-psro:{}-{}-{}'.format(config.train_from_scratch,
                                                                     not config.train_solver_only, config.problem,
                                                                     config.method),
                    project='ASP',
                    config=vars(config),
                    save_code=True,
                    resume='allow'
                )
                self.logger = wandb
            else:
                self.logger = None

            self.create_dir = config.create_dir
            self.path = args_path

        else:
            if config.problem_scale_list is not None:
                self.ps_list = config.problem_scale_list
            else:
                self.ps_list = [config.problem_scale_start]
            self.resume_solver_agent = None
            C = None
            if config.training_status == 'AS':
                self.status = config.training_status
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.mix_dist_list = [{'mix_prob': [1.],
                                       'dist_param': [Uniform(torch.zeros(2).to(device),torch.ones(2).to(device))],
                         'data_gt': None} for _ in range(len(self.ps_list))]
                self.his_performance = {_:[] for _ in self.ps_list}
            else:
                self.status = 'psro'
                self.mix_dist_list = []
                self.his_performance = {}
            self.iterations = 0
            self.resume = False
            self.eval_res_list = []
            self.all_eval_res_list = []
            self.count = 0
            self.all_eval_steps = 0
            self.psro_eval_steps = 0
            self.AS_eval_steps = 0
            self.best_eval = 1e6
            self.all_best_eval = 1e6
            self.time_cost = 0

            if config.log_to_wandb:
                wandb.init(
                    name=None,
                    group='Combine-Agent-ts:{}-psro:{}-{}-{}'.format(config.train_from_scratch,
                                                                     not config.train_solver_only, config.problem,
                                                                     config.method),
                    project='ASP',
                    config=vars(config),
                    save_code=True,
                )
                self.logger = wandb
            else:
                self.logger = None

            self.create_dir = config.create_dir

            if config.create_dir:
                path = config.save_path
                problem = config.problem
                method = config.method
                if self.logger is not None:
                    name = self.logger.run.name
                else:
                    name = config.task_description
                self.path = os.path.join(path, 'save_asp', problem, method, name)

                if not os.path.exists(self.path):
                    os.makedirs(self.path)

                with open(os.path.join(self.path, "args.json"), 'w') as f:
                    dic = deepcopy(vars(config))
                    if 'device' in dic.keys():
                        dic.pop('device')
                    json.dump(dic, f, indent=True)

            self.psro = PSRO(config, )

        routing_problem = ['OP', 'PCTSP_DET', 'PCTSP_STOCH', 'SDVRP']
        if config.problem in routing_problem:
            config.method = 'AM'

        self.config = config
        self.patience = config.patience
        self.AS = AS(config, C)
        self.start = time.time()

    def sample_from_dist(self, dist):
        self_generated_data_gt = dist['data_gt']
        mix_prob = dist['mix_prob']
        num_data = self_generated_data_gt[0][0].shape[0]
        num_data_per_dist = (np.array(mix_prob) * num_data).astype(int)
        num_data_per_dist[num_data_per_dist == 0] = 1
        data_list = []
        gt_list = []
        for i in range(len(mix_prob)):
            idx = np.random.choice(np.arange(num_data), num_data_per_dist[i], False)
            data_list.append(self_generated_data_gt[i][0][idx])
            gt_list.append(self_generated_data_gt[i][1][idx])
        data = torch.cat(data_list, 0)
        gt = np.concatenate(gt_list, 0)
        return data, gt

    def train_asp(self):
        if self.status == 'psro':
            print('Train initial PSRO')
            self.psro.init(problem_scale=self.ps_list[-1], solver_agent=self.resume_solver_agent)
            self.his_performance[self.ps_list[-1]] = []
            solver_agent, mix_dist, eval_res, all_eval_res = self.psro.train_psro(self.his_performance, self.logger,
                                                                                  self.all_eval_steps,
                                                                                  self.psro_eval_steps)
            self.all_eval_steps += self.config.psro_loop
            self.psro_eval_steps += self.config.psro_loop
            self.mix_dist_list.append(mix_dist)

        else:
            print('Train initial AS')
            self.AS.init(self.ps_list, self.mix_dist_list, self.resume_solver_agent)
            solver_agent, eval_res, all_eval_res = self.AS.train_AS(self.his_performance, self.logger,
                                                                        self.all_eval_steps, self.AS_eval_steps)
            self.all_eval_steps += self.config.AS_loop
            self.AS_eval_steps += self.config.AS_loop
            self.count += 1

        self.eval_res_list.append(eval_res)
        self.all_eval_res_list.append(all_eval_res)

        if self.create_dir:
            self.save_asp_info(solver_agent.solver_agent_info())
            if eval_res < self.best_eval:
                self.best_eval = eval_res
                self.save_asp_info(solver_agent.solver_agent_info(), best=True)
            if all_eval_res < self.all_best_eval:
                self.all_best_eval = all_eval_res
                self.save_asp_info(solver_agent.solver_agent_info(), all=True)
        self.iterations += 1
        if self.logger is not None:
            self.logger.log(
                {'All Evalution Process/Performance': eval_res,
                 'All Evalution Process/All Performance': all_eval_res,
                 'All Evalution Process/Threshold': self.config.performance_thres,
                 'All Evalution Process/Best Eval': self.best_eval,
                 'epoch': self.iterations})
        while True:
            if eval_res < self.config.performance_thres or (
                    self.count > self.patience and self.ps_list[-1] < self.config.problem_scale_end):
                print('Train new PSRO')
                ps_temp = self.ps_list[-1] + self.config.problem_scale_step

                if ps_temp <= self.config.problem_scale_end:
                    self.ps_list.append(ps_temp)
                    self.his_performance[self.ps_list[-1]] = []
                else:
                    break
                self.psro.init(problem_scale=self.ps_list[-1], solver_agent=solver_agent)
                solver_agent, mix_dist, eval_res, all_eval_res = self.psro.train_psro(self.his_performance, self.logger,
                                                                                      self.all_eval_steps,
                                                                                      self.psro_eval_steps)
                self.all_eval_steps += self.config.psro_loop
                self.psro_eval_steps += self.config.psro_loop
                self.mix_dist_list.append(mix_dist)
                self.count = 0
                self.best_eval = eval_res

            else:
                print('Train new AS')
                self.AS.init(self.ps_list, self.mix_dist_list, solver_agent)
                solver_agent, eval_res, all_eval_res = self.AS.train_AS(self.his_performance, self.logger,
                                                                            self.all_eval_steps, self.AS_eval_steps)
                self.all_eval_steps += self.config.AS_loop
                self.AS_eval_steps += self.config.AS_loop
                self.count += 1

            self.iterations += 1
            self.eval_res_list.append(eval_res)
            self.all_eval_res_list.append(all_eval_res)

            if self.create_dir:
                self.save_asp_info(solver_agent.solver_agent_info())
                if eval_res <= self.best_eval:
                    self.best_eval = eval_res
                    self.save_asp_info(solver_agent.solver_agent_info(), best=True)
                    # for AS, if having improvement, reset the count
                    self.count = 0
                if all_eval_res < self.all_best_eval:
                    self.all_best_eval = all_eval_res
                    self.save_asp_info(solver_agent.solver_agent_info(), all=True)

            if self.logger is not None:
                self.logger.log(
                    {'All Evalution Process/Performance': eval_res,
                     'All Evalution Process/All Performance': all_eval_res,
                     'All Evalution Process/Threshold': self.config.performance_thres,
                     'All Evalution Process/Best Eval': self.best_eval,
                     'epoch': self.iterations})
            if self.iterations > self.config.iter_num - 1:
                break

    def save_asp_info(self, solver, best=False, all=False):
        self.time_cost += (time.time() - self.start)

        try:
            psro_info = {'meta_strategy': self.psro._meta_strategy,
                                      'meta_payoff': self.psro._meta_payoff,}
        except:
            psro_info = None
        if self.resume:
            if best:
                torch.save(
                    {
                        'psro_info': psro_info,
                        'problem_scale_list': self.ps_list,
                        'solver_param': solver,
                        'mix_dist_list': self.mix_dist_list,
                        'cost_mat': self.AS.C,
                        'iterations': self.iterations,
                        'eval_res': self.eval_res_list,
                        'all_eval_res':self.all_eval_res_list,
                        'his_performance': self.his_performance,
                        'all_eval_steps': self.all_eval_steps,
                        'psro_eval_steps': self.psro_eval_steps,
                        'AS_eval_steps': self.AS_eval_steps,
                        'best_eval': self.best_eval,
                        'all_best_eval': self.all_best_eval,
                        'count': self.count,
                        'time_cost': self.time_cost
                    }
                    , self.path + '/best_asp_info.pt'
                )
            elif all:
                torch.save(
                    {
                        'psro_info': psro_info,
                        'problem_scale_list': self.ps_list,
                        'solver_param': solver,
                        'mix_dist_list': self.mix_dist_list,
                        'cost_mat': self.AS.C,
                        'iterations': self.iterations,
                        'eval_res': self.eval_res_list,
                        'all_eval_res': self.all_eval_res_list,
                        'his_performance': self.his_performance,
                        'all_eval_steps': self.all_eval_steps,
                        'psro_eval_steps': self.psro_eval_steps,
                        'AS_eval_steps': self.AS_eval_steps,
                        'best_eval': self.best_eval,
                        'all_best_eval': self.all_best_eval,
                        'count': self.count,
                        'time_cost': self.time_cost
                    }
                    , self.path + '/all_best_asp_info.pt'
                )

            else:
                torch.save(
                    {
                        'psro_info': psro_info,
                        'problem_scale_list': self.ps_list,
                        'solver_param': solver,
                        'mix_dist_list': self.mix_dist_list,
                        'cost_mat': self.AS.C,
                        'iterations': self.iterations,
                        'eval_res': self.eval_res_list,
                        'all_eval_res': self.all_eval_res_list,
                        'his_performance': self.his_performance,
                        'all_eval_steps': self.all_eval_steps,
                        'psro_eval_steps': self.psro_eval_steps,
                        'AS_eval_steps': self.AS_eval_steps,
                        'best_eval': self.best_eval,
                        'all_best_eval': self.all_best_eval,
                        'count': self.count,
                        'time_cost': self.time_cost

                    }
                    , self.path + '/asp_info_latest.pt'
                )
        else:
            if best:
                torch.save(
                    {
                        'psro_info': psro_info,
                        'problem_scale_list': self.ps_list,
                        'solver_param': solver,
                        'mix_dist_list': self.mix_dist_list,
                        'cost_mat': self.AS.C,
                        'iterations': self.iterations,
                        'eval_res': self.eval_res_list,
                        'all_eval_res': self.all_eval_res_list,
                        'his_performance': self.his_performance,
                        'all_eval_steps': self.all_eval_steps,
                        'psro_eval_steps': self.psro_eval_steps,
                        'AS_eval_steps': self.AS_eval_steps,
                        'best_eval': self.best_eval,
                        'all_best_eval': self.all_best_eval,
                        'count': self.count,
                        'time_cost': self.time_cost

                    }
                    , self.path + '/best_asp_info.pt'
                )

            elif all:
                torch.save(
                    {
                        'psro_info': psro_info,
                        'problem_scale_list': self.ps_list,
                        'solver_param': solver,
                        'mix_dist_list': self.mix_dist_list,
                        'cost_mat': self.AS.C,
                        'iterations': self.iterations,
                        'eval_res': self.eval_res_list,
                        'all_eval_res': self.all_eval_res_list,
                        'his_performance': self.his_performance,
                        'all_eval_steps': self.all_eval_steps,
                        'psro_eval_steps': self.psro_eval_steps,
                        'AS_eval_steps': self.AS_eval_steps,
                        'best_eval': self.best_eval,
                        'all_best_eval': self.all_best_eval,
                        'count': self.count,
                        'time_cost': self.time_cost
                    }
                    , self.path + '/all_best_asp_info.pt'
                )

            else:
                torch.save(
                    {
                        'psro_info': psro_info,
                        'problem_scale_list': self.ps_list,
                        'solver_param': solver,
                        'mix_dist_list': self.mix_dist_list,
                        'cost_mat': self.AS.C,
                        'iterations': self.iterations,
                        'eval_res': self.eval_res_list,
                        'all_eval_res': self.all_eval_res_list,
                        'his_performance': self.his_performance,
                        'all_eval_steps': self.all_eval_steps,
                        'psro_eval_steps': self.psro_eval_steps,
                        'AS_eval_steps': self.AS_eval_steps,
                        'best_eval': self.best_eval,
                        'all_best_eval': self.all_best_eval,
                        'count': self.count,
                        'time_cost': self.time_cost

                    }
                    , self.path + '/asp_info_latest.pt'
                )
