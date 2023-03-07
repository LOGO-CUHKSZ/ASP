import argparse
import torch


def get_DG_configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100,
                        help='The number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-5,)
    parser.add_argument('--train_batch', type=int, default=128,)
    parser.add_argument('--eval_batch', type=int, default=10,)
    parser.add_argument('--nf_layer', type=int, default=5,
                        help='Number of layers of Normalizing Flows')

    config = parser.parse_args()

    config.use_cuda = torch.cuda.is_available()
    return config