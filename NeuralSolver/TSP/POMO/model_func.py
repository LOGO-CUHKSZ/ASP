import torch
from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from TSPModel import MixModel

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from utils import *
from collections import OrderedDict
from torch.utils.data import DataLoader


def initialize(opts, baseline_mode=False):
    env_params = opts.env_params
    model_params = opts.model_params
    optimizer_params = opts.optimizer_params
    trainer_params = opts.trainer_params
    # cuda
    USE_CUDA = trainer_params['use_cuda']
    if USE_CUDA:
        cuda_device_num = trainer_params['cuda_device_num']
        torch.cuda.set_device(cuda_device_num)
        device = torch.device('cuda', cuda_device_num)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    # Main Components
    model = Model(baseline_mode, **model_params)
    # param = torch.load('/mnt/data1/wangchenguang/PSRO-CO/save_game_asp/TSP/POMO/zany-universe-197/asp_info_38.pt')['solver_param']['model']
    # model.load_state_dict(param)
    env = Env(**env_params)
    # optimizer = Optimizer(model.adapter_after_embedder.parameters(), **optimizer_params['optimizer'])
    optimizer = Optimizer(model.parameters(), **optimizer_params['optimizer'])
    scheduler = Scheduler(optimizer, **optimizer_params['scheduler'])

    if not opts.train_from_scratch:
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)

            model_dict = checkpoint['model_state_dict']
            model_state_dict = model.state_dict()
            state_dict = {k: v for k, v in model_dict.items() if k in model_state_dict.keys()}
            model_state_dict.update(state_dict)
            model.load_state_dict({**model.state_dict(), **model_state_dict})

        # for name, p in model.named_parameters():
        #     if name.split('.')[0] != 'adapter_after_embedder':
        #         p.requires_grad = False

    return env, model, None, optimizer, scheduler


def train_one_epoch(train_datasets, val_datasets, epoch, opts, *model_input):
    env, model, baseline, optimizer, lr_scheduler = model_input

    def _train_one_batch(train_dataset):
        # Prep
        ###############################################
        batch_size, n = train_dataset.shape[0], train_dataset.shape[1]

        model.train()
        env.load_problems(batch_size, prepare_dataset=train_dataset)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()
        while not done:
            selected, prob = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float()  # negative sign to make positive value

        # Step & Return
        ###############################################
        model.zero_grad()
        loss_mean.backward()
        optimizer.step()
        return score_mean, loss_mean.item()

    score_list = []
    for i, train_dataset in enumerate(train_datasets):
        train_dataset, val_dataset = train_datasets[i], val_datasets[i]
        if len(train_dataset) > 0 and len(val_dataset) > 0:
            opts.env_params['problem_size'] = train_dataset.size(1)
            # batch_size = train_dataset.size(0)
            batch_size = opts.solver_batch_size
            dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
            for batch_data in dataloader:
                avg_score, avg_loss = _train_one_batch(batch_data)
                score_list.append(avg_score)
    return torch.cat(score_list)


def eval(env, model, dataset, opts):

    def _test_one_batch(data):
        batch_size, n = data.shape[0], data.shape[1]
        # Augmentation
        ###############################################
        if opts.tester_params['augmentation_enable']:
            aug_factor = opts.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        model.eval()
        with torch.no_grad():
            env.load_problems(batch_size, aug_factor, prepare_dataset=data)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :]  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward  # negative sign to make positive value

        return no_aug_score, aug_score


    with torch.no_grad():
        if isinstance(dataset, list):
            score_list = []
            for data in dataset:
                if len(data)>0:
                    batch_size = min(1000, data.shape[0])
                    dataloader = DataLoader(dataset=data, batch_size=batch_size)
                    aug_score_list = []
                    for batch_data in dataloader:
                        score, aug_score = _test_one_batch(batch_data)
                        aug_score_list.append(aug_score)

                    score_list.append(torch.cat(aug_score_list))
            return torch.cat(score_list)
        else:
            batch_size = min(1000, dataset.shape[0])
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
            aug_score_list = []
            for batch_data in dataloader:
                score, aug_score = _test_one_batch(batch_data)
                aug_score_list.append(aug_score)
            return torch.cat(aug_score_list)


def get_mix_solver(opts, mix_prob, param_list):
    model_params = opts.model_params
    # Main Components
    model = MixModel(mix_prob, **model_params)
    merge_state_dict = OrderedDict()
    model_state_dict = model.state_dict()
    for i in range(len(param_list)):
        param = param_list[i]
        for k, v in param.items():
            if k=='adapter_after_embedder.modulelist.down_proj.weight':
                merge_state_dict['adapter_list.adapter_{}.modulelist.down_proj.weight'.format(i)] = v
            elif k=='adapter_after_embedder.modulelist.down_proj.bias':
                merge_state_dict['adapter_list.adapter_{}.modulelist.down_proj.bias'.format(i)] = v
            elif k=='adapter_after_embedder.modulelist.up_proj.weight':
                merge_state_dict['adapter_list.adapter_{}.modulelist.up_proj.weight'.format(i)] = v
            elif k=='adapter_after_embedder.modulelist.up_proj.bias':
                merge_state_dict['adapter_list.adapter_{}.modulelist.up_proj.bias'.format(i)] = v
            else:
                merge_state_dict[k] = v
    model_state_dict.update(merge_state_dict)
    model.load_state_dict({**model.state_dict(), **model_state_dict})
    return model
