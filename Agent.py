import torch
from DataGenerator.FlowGenerator import NormalizingFlowModel, AffineHalfFlow, AffineConstantFlow
from torch.distributions.uniform import Uniform
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
import sys
import os
import numpy as np


class Agent_DG:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.problem_scale = config.problem_scale
        self.data_generator, self.oracle, self.prior = self.setup_DG()
        # set uniform as initial distribution
        self.oracle = self.prior

    def setup_DG(self, ):

        def set_dg(input_dim, prior, type='affine'):
            if type == 'affine':
                flows = [AffineHalfFlow(dim=input_dim, parity=0 % 2, norm=True)] + [
                    AffineHalfFlow(dim=input_dim, parity=i % 2, norm=True) for i in
                    range(1, self.config.dg_nf_layer - 1)]
            else:
                flows = [AffineConstantFlow(dim=input_dim) for i in range(self.config.dg_nf_layer)]
            data_generator = NormalizingFlowModel(prior, flows).to(self.device)
            oracle = deepcopy(data_generator.state_dict())
            return data_generator, oracle

        routing_problem = ['TSP', 'CVRP', 'SDVRP', 'OP', 'PCTSP_DET', 'PCTSP_STOCH', 'SDVRP']
        if self.config.problem in routing_problem:
            prior = Uniform(torch.zeros(2).to(self.device),
                            torch.ones(2).to(self.device))
            data_generator, oracle = set_dg(2, prior)

        else:
            NotImplementedError
        return data_generator, oracle, prior

    def train_oracle(self, solver, eval_func, meta_strategy_SS, param_SS_list, scale_strategy, train_from_scratch=True,
                     logger=None):
        '''
        :param neural_solver: mixed neural solver
        :return:
        '''
        self.best_result = 0
        if train_from_scratch:
            self.data_generator, self.oracle, self.prior = self.setup_DG()
        optimizer = optim.Adam(self.data_generator.parameters(), lr=self.config.dg_lr, weight_decay=self.config.dg_wd)
        sample_prob = np.abs(np.array(list(meta_strategy_SS))) / np.sum(np.abs(np.array(list(meta_strategy_SS))))
        # for epoch in tqdm(range(self.config.dg_epochs), desc='Training Process of Generator'):
        for epoch in tqdm(range(self.config.dg_epochs), desc='Training Process of Data Generator'):
            sample_solver_idx = np.random.choice(np.arange(len(param_SS_list)), p=sample_prob)
            solver.load_state_dict({**solver.state_dict(), **param_SS_list[sample_solver_idx]['model']})
            num_batch_per_scale = [int(self.config.dg_train_batch * scale_strategy[i]) for i in
                                   range(len(scale_strategy))]
            if self.config.problem == 'TSP':
                num_sample_per_scale = [num_batch_per_scale[i] * (self.problem_scale[i]) for i in
                                        range(len(scale_strategy))]
            elif self.config.problem == 'CVRP' or self.config.problem == 'SDVRP' or self.config.problem == 'OP' or self.config.problem[
                                                                                                                   :5] == 'PCTSP':
                num_sample_per_scale = [num_batch_per_scale[i] * (self.problem_scale[i] + 1) for i in
                                        range(len(scale_strategy))]
            elif self.config.problem == 'JSSP':
                num_sample_per_scale = [num_batch_per_scale[i] * (self.problem_scale[i][0] * self.problem_scale[i][1])
                                        for i in range(len(scale_strategy))]
            else:
                NotImplementedError

            z = self.prior.sample((sum(num_sample_per_scale),))
            prior_logprob = self.prior.log_prob(z).view(z.size(0), -1).sum(1)
            output = self.data_generator.backward(z)
            data, log_det = output[0][-1], output[1]
            log_prob = prior_logprob + log_det
            list_batch_data, log_prob = self.get_diff_scales(data, log_prob, num_batch_per_scale, num_sample_per_scale)
            with torch.no_grad():
                func = eval_func(solver)
                obj_of_solver = func(list_batch_data).to(log_prob.device).view(*log_prob.shape)
            loss = -torch.mean(obj_of_solver * log_prob)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if logger is not None:
                logger.log({'PSRO:DG/Objective of PSRO': obj_of_solver.mean().cpu().data.item(),
                            'PSRO:DG/loss of PSRO': loss.cpu().data.item()})
            is_best = obj_of_solver.mean() > self.best_result
            if is_best:
                self.best_result = obj_of_solver.mean()
                self.oracle = deepcopy(self.data_generator.state_dict())

    def get_diff_scales(self, data, log_prob, num_batch_per_scale, num_sample_per_scale):
        count = 0
        list_batch_data = []
        list_log_prob = []
        for i in range(len(num_batch_per_scale)):
            if self.config.problem == 'TSP':
                list_batch_data.append(
                    data[count:count + num_sample_per_scale[i]].view(num_batch_per_scale[i], self.problem_scale[i], 2))
                list_log_prob.append(log_prob[count:count + num_sample_per_scale[i]].view(num_batch_per_scale[i],
                                                                                          self.problem_scale[i]).sum(
                    -1))
            elif self.config.problem == 'CVRP' or self.config.problem == 'SDVRP':
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
                demand = torch.FloatTensor(num_batch_per_scale[i], (self.problem_scale[i] + 1), 1).uniform_(1,
                                                                                                            10).int() / \
                         CAPACITIES[
                             self.problem_scale[i]]
                demand[:, 0, 0] = 0

                list_batch_data.append(
                    torch.cat([data[count:count + num_sample_per_scale[i]].view(num_batch_per_scale[i],
                                                                                self.problem_scale[i] + 1, 2),
                               demand.to(data.device)], dim=-1)
                )
                list_log_prob.append(log_prob[count:count + num_sample_per_scale[i]].view(num_batch_per_scale[i],
                                                                                          self.problem_scale[
                                                                                              i] + 1).sum(
                    -1))
            elif self.config.problem == 'OP' or self.config.problem[:5] == 'PCTSP':
                list_batch_data.append(
                    data[count:count + num_sample_per_scale[i]].view(num_batch_per_scale[i], self.problem_scale[i] + 1,
                                                                     2))
                list_log_prob.append(log_prob[count:count + num_sample_per_scale[i]].view(num_batch_per_scale[i],
                                                                                          self.problem_scale[
                                                                                              i] + 1).sum(
                    -1))
            else:
                NotImplementedError

            count += num_sample_per_scale[i]
        return list_batch_data, torch.cat(list_log_prob)

    def sample_instance(self, param, problem_scale, batch_size):

        data_generator, _, prior = self.setup_DG()
        data_generator.eval()
        try:
            data_generator.load_state_dict(param)
            uni_judge = False
        except:
            data_generator = prior
            uni_judge = True
        with torch.no_grad():
            if self.config.problem == 'TSP':
                z = self.prior.sample((problem_scale * batch_size,))
                if uni_judge:
                    data = z.view(batch_size, problem_scale, -1)
                else:
                    data_list = []
                    for _ in range(problem_scale):
                        data_list.append(data_generator.backward(z[_*batch_size:(_+1)*batch_size])[0][-1])
                    data = torch.stack(data_list,dim=1).detach()
                    # data = data_generator.backward(z)[0][-1].view(batch_size, problem_scale, -1).detach()
            elif self.config.problem == 'CVRP' or self.config.problem == 'SDVRP':
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
                z = self.prior.sample(((problem_scale + 1) * batch_size,))
                if uni_judge:
                    data = z.view(batch_size, (problem_scale + 1), -1)
                else:
                    data_list = []
                    for _ in range(problem_scale+1):
                        data_list.append(data_generator.backward(z[_ * batch_size:(_ + 1) * batch_size])[0][-1])
                    data = torch.stack(data_list, dim=1).detach()
                    # data = data_generator.backward(z)[0][-1].view(batch_size, (problem_scale + 1), -1).detach()
                demand = torch.FloatTensor(batch_size, (problem_scale + 1), 1).uniform_(1, 10).int() / CAPACITIES[
                    problem_scale]
                data = torch.cat([data, demand.to(data.device)], dim=-1)
            elif self.config.problem == 'OP':
                z = self.prior.sample(((problem_scale + 1) * batch_size,))
                if uni_judge:
                    data = z.view(batch_size, (problem_scale + 1), -1)
                else:
                    data_list = []
                    for _ in range(problem_scale + 1):
                        data_list.append(data_generator.backward(z[_ * batch_size:(_ + 1) * batch_size])[0][-1])
                    data = torch.stack(data_list, dim=1).detach()
                    # data = data_generator.backward(z)[0][-1].view(batch_size, (problem_scale + 1), -1).detach()
                depot = data[:, 0, :2]
                loc = data[:, 1:, :2]
                temp = torch.zeros(batch_size, (problem_scale + 1)).to(data.device)
                prize_ = (depot[:, None, :] - loc).norm(p=2, dim=-1)
                prize = (1 + (prize_ / (prize_).max(dim=-1, keepdim=True)[0] * 99)) / 100.
                temp[:, 1:] = prize
                data = torch.cat([data, temp.unsqueeze(-1)], dim=-1)
            elif self.config.problem[:5] == 'PCTSP':
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
                z = self.prior.sample(((problem_scale + 1) * batch_size,))
                if uni_judge:
                    data = z.view(batch_size, (problem_scale + 1), -1)
                else:
                    data_list = []
                    for _ in range(problem_scale + 1):
                        data_list.append(data_generator.backward(z[_ * batch_size:(_ + 1) * batch_size])[0][-1])
                    data = torch.stack(data_list, dim=1).detach()
                    # data = data_generator.backward(z)[0][-1].view(batch_size, (problem_scale + 1), -1).detach()

                penalty_max = MAX_LENGTHS[problem_scale] * (3) / float(problem_scale)
                penalty = torch.rand(batch_size, problem_scale).to(data.device) * penalty_max
                deterministic_prize = torch.rand(batch_size, problem_scale).to(data.device) * 4 / float(problem_scale)
                stochastic_prize = torch.rand(batch_size, problem_scale).to(data.device) * deterministic_prize * 2
                temp = torch.zeros(batch_size, (problem_scale + 1), 3).to(data.device)
                temp[:, 1:, 0] = penalty
                temp[:, 1:, 1] = deterministic_prize
                temp[:, 1:, 2] = stochastic_prize
                data = torch.cat([data, temp], dim=-1)
            else:
                NotImplementedError
        return data

    def load_state_dict_dg(self, param):
        self.data_generator.load_state_dict(param)

    def sample_mix_dist(self, mix_dist, scale, sample_size):
        '''
        mix_dist: dict {'mix_prob':[p1,...], 'dist_param':[param1,...]}
        '''
        mix_prob = np.array(mix_dist['mix_prob'])
        param_list = mix_dist['dist_param']
        sample_size_per_dist = np.ceil((sample_size * mix_prob)).astype(np.int32)
        data_l = []
        for i, s in enumerate(sample_size_per_dist):
            if s >= 1:
                param = param_list[i]
                data = self.sample_instance(param, scale, s)
                data_l.append(data)
        return torch.cat(data_l, 0)


class Agent_Solver:
    def __init__(self, config):
        sys_path = os.getcwd() + '/NeuralSolver'
        if config.problem == 'TSP':
            sys_path += '/TSP'
            if config.method == 'AM':
                sys.path.append(sys_path + '/AM')
                from NeuralSolver.TSP.AM.model_func import initialize, train_one_epoch, eval
            elif config.method == 'POMO':
                sys.path.append(sys_path + '/POMO')
                from NeuralSolver.TSP.POMO.model_func import initialize, train_one_epoch, eval
        elif config.problem == 'CVRP' or config.problem == 'SDVRP':
            if config.method == 'AM':
                sys.path.append(sys_path + '/TSP' + '/AM')
                from NeuralSolver.TSP.AM.model_func import initialize, train_one_epoch, eval
            elif config.method == 'POMO':
                sys.path.append(sys_path + '/CVRP' + '/POMO')
                from NeuralSolver.CVRP.POMO.model_func import initialize, train_one_epoch, eval
        elif config.problem == 'OP' or config.problem[:5] == 'PCTSP':
            sys.path.append(sys_path + '/TSP' + '/AM')
            from NeuralSolver.TSP.AM.model_func import initialize, train_one_epoch, eval

        self.config = config
        self.init_func = initialize
        self.train_one_epoch = train_one_epoch
        self.eval_func = eval
        self.init()
        self.oracle = deepcopy(self.model.state_dict())

    def init(self):
        self.problem, self.model, self.baseline, self.optimizer, self.lr_scheduler = self.init_func(self.config, )
        return self.problem, self.lr_scheduler

    def update_config(self, config):
        self.config = config

    def train_oracle(self, sampler, train_from_scratch=False, logger=None):
        self.best_result = 1e10
        if train_from_scratch:
            self.init()
        model_input = (self.problem, self.model, self.baseline, self.optimizer, self.lr_scheduler)
        # for epoch in tqdm(range(self.config.epoch_start, self.config.epoch_start + self.config.n_epochs),
        #                   desc='Training Process of Solver'):
        for epoch in tqdm(range(self.config.solver_epochs), desc='Training Process of Neural Solver'):
            train_dataset, val_dataset = sampler(True), sampler(False)
            best_val = self.train_one_epoch(train_dataset, val_dataset, epoch, self.config, *model_input)

            if logger is not None:
                logger.log({'PSRO:SS/Objective of PSRO': best_val.mean().cpu().data.item()})
            is_best = best_val.mean() < self.best_result
            if is_best:
                self.best_result = best_val.mean()
                self.oracle = deepcopy(self.model.state_dict())

    def get_solver(self, solver):
        return lambda dataset: self.eval_solver(solver, dataset, self.config.problem_scale)

    def eval_solver(self, neural_solver, dataset, problem_scale):
        return self.eval_func(self.problem, neural_solver, dataset, self.config)

    def load_state_dict_solver(self, param):
        model_dict = param
        model_state_dict = self.model.state_dict()
        state_dict = {k: v for k, v in model_dict.items() if k in model_state_dict.keys()}
        model_state_dict.update(state_dict)
        self.model.load_state_dict({**self.model.state_dict(), **model_state_dict})
        return self.model

    def solver_agent_info(self):
        try:
            baseline_info=self.baseline.state_dict()
        except:
            baseline_info = None
        try:
            optimizer_info = self.optimizer.state_dict()
        except:
            optimizer_info = None
        return {'model': self.model.state_dict(),
                'baseline': baseline_info,
                'optimizer': optimizer_info}

    def load_solver_agent_info(self, info, type='resume'):
        self.problem, self.lr_scheduler = self.init()
        if type == 'resume':
            try:
                self.baseline.load_state_dict(info['baseline'])
            except:
                pass
            try:
                self.optimizer.load_state_dict(info['optimizer'])
            except:
                pass
            self.model.load_state_dict(info['model'])
        elif type == 'training':
            self.model.load_state_dict(info['model'])
            try:
                self.baseline.load_state_dict(info['baseline'])
            except:
                pass
        return self
