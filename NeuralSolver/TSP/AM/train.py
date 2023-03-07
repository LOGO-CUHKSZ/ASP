import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    # print('Validating...')
    if isinstance(dataset, list):
        cost = []
        for data in dataset:
            if opts.problem == 'TSP':
                from problems.tsp.problem_tsp import Pre_TSPDataset
                data = Pre_TSPDataset(data)
            elif opts.problem == 'CVRP' or opts.problem=='SDVRP':
                from problems.vrp.problem_vrp import Pre_VRPDataset
                data = Pre_VRPDataset(data)
            elif opts.problem == 'OP':
                from problems.op.problem_op import Pre_OPDataset
                data = Pre_OPDataset(data)
            elif opts.problem[:5] == 'PCTSP':
                from problems.pctsp.problem_pctsp import Pre_PCTSPDataset
                data = Pre_PCTSPDataset(data)
            if len(data)>0:
                cost.append(rollout(model, data, opts))
        cost = torch.cat(cost)
    else:
        if opts.problem == 'TSP':
            from problems.tsp.problem_tsp import Pre_TSPDataset
            dataset = Pre_TSPDataset(dataset)
        elif opts.problem == 'CVRP' or opts.problem == 'SDVRP':
            from problems.vrp.problem_vrp import Pre_VRPDataset
            dataset = Pre_VRPDataset(dataset)
        elif opts.problem == 'OP':
            from problems.op.problem_op import Pre_OPDataset
            dataset = Pre_OPDataset(dataset)
        elif opts.problem[:5] == 'PCTSP':
            from problems.pctsp.problem_pctsp import Pre_PCTSPDataset
            dataset = Pre_PCTSPDataset(dataset)
        cost = rollout(model, dataset, opts)
    # avg_cost = cost.mean()
    # print('Validation overall avg_cost: {} +- {}'.format(
    #     avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in DataLoader(dataset, batch_size=opts.eval_batch_size)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(problem, model, optimizer, baseline, lr_scheduler, epoch, train_dataset, val_dataset, opts):
    # print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    train_dataset = baseline.wrap_dataset(train_dataset)
    training_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(training_dataloader):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            opts
        )

        step += 1

    # print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    #
    # if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
    #     print('Saving model and state...')
    #     torch.save(
    #         {
    #             'model': get_inner_model(model).state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'rng_state': torch.get_rng_state(),
    #             'cuda_rng_state': torch.cuda.get_rng_state_all(),
    #             'baseline': baseline.state_dict()
    #         },
    #         os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
    #     )
    #
    # avg_reward = validate(model, val_dataset, opts)
    #
    # if not opts.no_tensorboard:
    #     tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch, val_dataset)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    best_val = validate(model, val_dataset, opts)
    return best_val


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # # Logging
    # if step % int(opts.log_step) == 0:
    #     log_values(cost, grad_norms, epoch, batch_id, step,
    #                log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
