import os

import torch
import torch.optim as optim
from collections import OrderedDict
from nets.critic_network import CriticNetwork
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel, MixModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem


def initialize(opts, baseline_mode=False):
    # Set the random seed
    # torch.manual_seed(opts.random_seed)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    try:
        assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
        if opts.problem == 'TSP':
            if opts.n_encode_layers == 3:
                load_path = './NeuralSolver/TSP/AM/pretrained/tsp_20/epoch-99.pt'
            elif opts.n_encode_layers ==6:
                load_path = './NeuralSolver/TSP/AM/pretrained/tsp20_6layer/epoch-99.pt'
            else:
                load_path = './NeuralSolver/TSP/AM/pretrained/tsp20_9layer/epoch-99.pt'
        elif opts.problem == 'CVRP':
            load_path = './NeuralSolver/TSP/AM/pretrained/cvrp_20/epoch-99.pt'
        elif opts.problem == 'SDVRP':
            load_path = './NeuralSolver/TSP/AM/pretrained/sdvrp_20/epoch-99.pt'
        elif opts.problem == 'OP':
            load_path = './NeuralSolver/TSP/AM/pretrained/op_dist_20/epoch-99.pt'
        elif opts.problem == 'PCTSP_DET':
            load_path = './NeuralSolver/TSP/AM/pretrained/pctsp_det_20/epoch-99.pt'
        elif opts.problem == 'PCTSP_STOCH':
            load_path = './NeuralSolver/TSP/AM/pretrained/pctsp_stoch_20/epoch-99.pt'
        else:
            load_data = ''
        if load_path is not None:
            # print('  [*] Loading data from {}'.format(load_path))
            load_data = torch_load_cpu(load_path)
    except:
        pass

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        baseline_mode=baseline_mode
    ).to(opts.device)
    if not opts.train_from_scratch:
        model_dict = load_data.get('model', {})
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in model_dict.items() if k in model_state_dict.keys()}
        model_state_dict.update(state_dict)
        model.load_state_dict({**model.state_dict(), **model_state_dict})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'TSP', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if not opts.train_from_scratch:
        if 'baseline' in load_data:
            baseline.load_state_dict(load_data['baseline'])


    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if not opts.train_from_scratch:
        if 'optimizer' in load_data:
            optimizer.load_state_dict(load_data['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    # if isinstance(v, torch.Tensor):
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    return problem, model, baseline, optimizer, lr_scheduler


def train_one_epoch(train_datasets, val_datasets, epoch, opts, *model_input):
    problem, model, baseline, optimizer, lr_scheduler = model_input
    best_val_list = []
    for i in range(len(train_datasets)):
        train_dataset, val_dataset = train_datasets[i], val_datasets[i]
        if opts.problem == 'TSP':
            from problems.tsp.problem_tsp import Pre_TSPDataset
            train_dataset = Pre_TSPDataset(train_dataset)
        elif opts.problem == 'CVRP' or opts.problem == 'SDVRP':
            from problems.vrp.problem_vrp import Pre_VRPDataset
            train_dataset = Pre_VRPDataset(train_dataset)
        elif opts.problem == 'OP':
            from problems.op.problem_op import Pre_OPDataset
            train_dataset = Pre_OPDataset(train_dataset)
        elif opts.problem[:5] == 'PCTSP':
            from problems.pctsp.problem_pctsp import Pre_PCTSPDataset
            train_dataset = Pre_PCTSPDataset(train_dataset)

        if len(train_dataset) > 0 and len(val_dataset) > 0:
            best_val = train_epoch(problem, model, optimizer, baseline, lr_scheduler, epoch, train_dataset, val_dataset, opts)
            best_val_list.append(best_val)
    return torch.cat(best_val_list)


def eval(problem, model, dataset, opts,):
    with torch.no_grad():
        best_val = validate(model, dataset, opts)
    return best_val


