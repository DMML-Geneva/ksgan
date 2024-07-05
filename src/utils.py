import contextlib
from collections import OrderedDict
from collections.abc import Callable
import os
import random
from functools import partial

import numpy as np
import wandb
import torch

EPS = 1e-6


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Copied from: https://stackoverflow.com/a/34333710

    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


def set_seed(seed, seed_shift=1670):
    if seed is not None:
        seed += seed_shift
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        env_context = modified_environ(CUBLAS_WORKSPACE_CONFIG=":4096:8")
    else:
        env_context = contextlib.nullcontext()
    return env_context, np.random.RandomState(seed=seed)


def wandb_log_params(models, log="all"):
    if not isinstance(models, (tuple, list)):
        models = (models,)
    for local_idx, model in enumerate(models):
        # Prefix used for name consistency with wandb convention.
        # Copied from wandb_watch.py:L82-L87
        if local_idx > 0:
            # This makes ugly chart names like gradients/graph_1conv1d.bias
            prefix = "graph_%i" % local_idx
        else:
            prefix = ""
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if log in ("parameters", "all"):
                    wandb.run._torch.log_tensor_stats(
                        parameter, "parameters/" + prefix + name
                    )
                if log in ("gradients", "all") and parameter.grad is not None:
                    wandb.run._torch.log_tensor_stats(
                        parameter.grad.data, "gradients/" + prefix + name
                    )


def set_torch_defaults(precision=32):
    torch.distributions.Distribution.set_default_validate_args(False)
    if precision == 16:
        torch.set_default_dtype(torch.float16)
    elif precision == 32:
        torch.set_default_dtype(torch.float32)
    elif precision == 64:
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError(f"Unknown precision value: {precision}")


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if default_factory is not None and not isinstance(
            default_factory, Callable
        ):
            raise TypeError("first argument must be callable")
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = (self.default_factory,)
        return type(self), args, None, None, iter(self.items())

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy

        return type(self)(self.default_factory, copy.deepcopy(self.items()))

    def __repr__(self):
        return "OrderedDefaultDict(%s, %s)" % (
            self.default_factory,
            OrderedDict.__repr__(self),
        )


class TrainingWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


def expand_over_dim(tensor, expand_dim, expand_size):
    tensor.unsqueeze_(expand_dim)
    return tensor.expand(
        *[
            -1 if idx != expand_dim else expand_size
            for idx in range(tensor.ndim + 1)
        ]
    )


def get_grid(extent, resolution, device=torch.device("cpu"), eps=EPS):
    extent = np.array(extent).T
    tensors = [
        torch.linspace(
            start=dim[0] - eps,
            end=dim[1] + eps,
            steps=resolution,
            device=device,
        )
        for dim in extent
    ]
    return torch.cartesian_prod(*tensors).view(-1, len(extent)), (
        np.log(np.prod((extent[:, 1] - extent[:, 0] + 2 * eps) / resolution))
    )


def even_divide(num, div):
    groupSize, remainder = divmod(num, div)
    return [groupSize + (1 if x < remainder else 0) for x in range(div)]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    num = len(lst)
    sizes = [num // n + (1 if x < num % n else 0) for x in range(n)]
    start_idx = 0
    for s in sizes:
        yield lst[start_idx : start_idx + s]
        start_idx += s


def load_model(
    path=None,
    checkpoint=None,
    model=None,
    optimizer=None,
    lr_scheduler=None,
    best=True,
    map_location=None,
):
    assert (
        path is not None or checkpoint is not None
    ), "One of `path` and `checkpoint` has to be provided!"
    assert not (
        path is not None and checkpoint is not None
    ), "Cannot prove both`path` and `checkpoint`!"
    if checkpoint is None:
        checkpoint = torch.load(
            os.path.join(path, "model.pt"), map_location=map_location
        )
    else:
        assert (
            map_location is None
        ), "Cannot provide `map_location` when `checkpoint` passed!"
    if model is not None:
        model.load_state_dict(
            checkpoint["model"] if not best else checkpoint["best_model"]
        )
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


def parse_torch_nn_class(cfg, default):
    if cfg is None:
        return default
    elif isinstance(cfg, dict):
        c = getattr(
            torch.nn,
            cfg.pop(
                "type",
            ),
        )
        if c_params := cfg.pop("parameters", False):
            c = partial(c, **c_params)
        return c
    else:
        return cfg


def get_func_on_data(
    func,
    dataloader=None,
    x=None,
    batch_size=None,
    device=torch.device("cpu"),
    comp_device=torch.device("cpu"),
):
    log_probs = []
    if dataloader is not None:
        data_iter = iter(dataloader)
        with torch.no_grad():
            while True:
                try:
                    log_probs.append(func(next(data_iter)).to(device))
                except StopIteration:
                    break
    else:
        with torch.no_grad():
            for idxs in chunks(range(x.shape[0]), batch_size):
                log_probs.append(func(x[idxs].to(comp_device)).to(device))
    return torch.cat(log_probs, dim=0)
