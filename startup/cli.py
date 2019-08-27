from pathlib import Path

from fire import Fire
from munch import Munch

import torch

from config import config, debug_options
from dataset import get_iterator
from utils import wait_for_key
from train import train


class Cli:
    def __init__(self):
        self.defaults = config
        self.debug = debug_options

    def _default_args(self, **kwargs):
        args = self.defaults
        if 'debug' in kwargs:
            args.update(self.debug)
        args.update(kwargs)
        args.update(resolve_paths(config))
        args.update(get_device(args))
        print(args)

        return Munch(args)

    def check_dataloader(self, **kwargs):
        args = self._default_args(**kwargs)

        iters, vocab = get_iterator(args)
        for batch in iters['train']:
            import ipdb; ipdb.set_trace()  # XXX DEBUG

    def train(self, **kwargs):
        args = self._default_args(**kwargs)

        train(args)

        wait_for_key()


def resolve_paths(config):
    paths = [k for k in config.keys() if k.endswith('_path')]
    res = {}
    for path in paths:
        res[path] = Path(config[path])

    return res


def get_device(args):
    if hasattr(args, 'device'):
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {'device': device}


if __name__ == "__main__":
    Fire(Cli)
