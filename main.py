"""main.py"""

import argparse

import numpy as np
import torch

from solver import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)

    if args.train:
        net.train()
    else:
        net.traverse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e5, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    parser.add_argument('--z_dim', default=32, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=5, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='3DChairs', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

    parser.add_argument('--gather_step', default=5e3, type=int, help='number of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=1e4, type=int, help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=1e5, type=int, help='number of iterations after which a checkpoint is saved')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')

    args = parser.parse_args()

    main(args)
