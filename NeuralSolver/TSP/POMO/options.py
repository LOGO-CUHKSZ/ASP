import argparse

def get_options(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--env_params', type=dict, default={
    'problem_size': 20,
    'pomo_size': 20,})

    parser.add_argument('--model_params', type=dict, default={
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',})

    parser.add_argument('--optimizer_params', type=dict, default={
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [501,],
        'gamma': 0.1
    }})

    parser.add_argument('--trainer_params', type=dict, default={
    'use_cuda': True,
    'cuda_device_num': 0,
    'epochs': 510,
    'train_episodes': 100 * 1000,
    'train_batch_size': 64,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_20.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        'path': './NeuralSolver/TSP/POMO/result/saved_tsp20_model_9layer',  # directory path of pre-trained model and log files saved.
        'epoch': 480,  # epoch version of pre-trained model to laod.

    }})

    parser.add_argument('--tester_params', type=dict, default={
    'use_cuda': True,
    'cuda_device_num': 0,
    'model_load': {
        'path': './result/saved_tsp20_model',  # directory path of pre-trained model and log files saved.
        'epoch': 510,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 100*1000,
    'test_batch_size': 10000,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 1000,})


    opts = parser.parse_args()
    try:
        opts.model_params['encoder_layer_num'] = opts.solver_n_encode_layers
    except:
        opts.model_params['encoder_layer_num'] = 6

    if opts.model_params['encoder_layer_num'] == 3:
        opts.trainer_params['model_load'] = {
        'enable': True,  # enable loading pre-trained model
        'path': './NeuralSolver/TSP/POMO/result/saved_tsp20_model_3layer',  # directory path of pre-trained model and log files saved.
        'epoch': 510,  # epoch version of pre-trained model to laod.
    }
    elif opts.model_params['encoder_layer_num']==6:
        opts.trainer_params['model_load'] = {
            'enable': True,  # enable loading pre-trained model
            'path': './NeuralSolver/TSP/POMO/result/saved_tsp20_model',
            # directory path of pre-trained model and log files saved.
            'epoch': 510,  # epoch version of pre-trained model to laod.
        }
    else:
        opts.trainer_params['model_load'] = {
            'enable': True,  # enable loading pre-trained model
            'path': './NeuralSolver/TSP/POMO/result/saved_tsp20_model_9layer',
            # directory path of pre-trained model and log files saved.
            'epoch': 480,  # epoch version of pre-trained model to laod.
        }


    if opts.tester_params['augmentation_enable']:
        opts.tester_params['test_batch_size'] = opts.tester_params['aug_batch_size']

    try:
        if opts.train_from_scratch == True:
            opts.solver_batch_size = opts.trainer_params['train_batch_size']
            opts.solver_epoch_size = int(opts.trainer_params['train_episodes']/opts.num_batch)
            # opts.solver_epoch_size = opts.solver_batch_size * opts.num_batch

        else:
            opts.solver_batch_size = opts.trainer_params['train_batch_size']
            opts.solver_epoch_size = opts.solver_batch_size * opts.num_batch
    except:
        pass
    return opts


