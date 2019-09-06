import os

import torch
from torch import nn
import torch.nn.functional as F

from dataset import get_iterator
from model import get_model
from utils import get_dirname_from_args


def get_ckpt_path(args, epoch, loss):
    ckpt_name = get_dirname_from_args(args)
    ckpt_path = args.ckpt_path / ckpt_name
    args.ckpt_path.mkdir(exist_ok=True)
    ckpt_path.mkdir(exist_ok=True)
    loss = '{:.4f}'.format(loss)
    ckpt_path = ckpt_path / \
        f'loss_{loss}_epoch_{epoch}.pickle'

    return ckpt_path


def save_ckpt(args, epoch, loss, model, vocab):
    print(f'saving epoch {epoch}')
    dt = {
        'args': args,
        'epoch': epoch,
        'loss': loss,
        'model': model.state_dict(),
        'vocab': vocab,
    }

    ckpt_path = get_ckpt_path(args, epoch, loss)
    print(f"Saving checkpoint {ckpt_path}")
    torch.save(dt, ckpt_path)


def get_model_ckpt(args):
    ckpt_available = args.ckpt_name is not None
    if ckpt_available:
        name = f'{args.ckpt_name}'
        name = f'{name}*' if not name.endswith('*') else name
        ckpt_paths = sorted(args.ckpt_path.glob(f'{name}'), reverse=False)
        assert len(ckpt_paths) > 0, f"no ckpt candidate for {args.ckpt_path / args.ckpt_name}"
        ckpt_path = ckpt_paths[0]  # monkey patch for choosing the best ckpt
        print(f"loading from {ckpt_path}")
        dt = torch.load(ckpt_path)
        args.update(dt['args'])
        vocab = dt['vocab']

    iters, vocab = get_iterator(args, vocab)
    model = get_model(args, vocab)

    if ckpt_available:
        model.load_state_dict(dt['model'])
    return args, model, iters, vocab, ckpt_available
