import importlib
import json
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
import wandb
from wandb.sdk.wandb_summary import SummarySubDict, Summary
from wandb.sdk.lib.disabled import SummaryDisabled
import scipy
import torch.utils.data
from pytorch_image_generation_metrics import get_inception_score, get_fid
from pytorch_image_generation_metrics.core import (
    get_inception_feature,
    torch_cov,
)

from src.data.utils import (
    get_dataloaders,
    SingleTensorDataset,
)
from src.utils import (
    set_torch_defaults,
    set_seed,
    even_divide,
    get_grid,
    load_model,
    get_func_on_data,
)


def model_has_log_prob(model_name):
    return importlib.import_module(
        f"src.models.{model_name}"
    ).Generator.has_log_prob


def model_has_critic(model_name):
    return importlib.import_module(
        f"src.models.{model_name}"
    ).Generator.has_critic


def model_has_rsample(model_name):
    return importlib.import_module(
        f"src.models.{model_name}"
    ).Generator.has_rsample


def get_samples(model, n, batch_size, device=torch.device("cpu")):
    samples = []
    with torch.no_grad():
        for bs in even_divide(n, batch_size):
            samples.append(model.sample((bs,)).to(device))
    return torch.cat(samples, dim=0)


def evaluate_rsample(
    model_name,
    simulator,
    model_config,
    config,
    model_path,
    data_path,
    out_path,
    metrics_summary_path,
    seed,
    cuda,
):
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(model_path, "wandb_group"), "r") as f:
        wandb_group = f.read()
    wandb_project = f"ksgan-{model_name}-{simulator}"
    wandb.init(
        project=wandb_project,
        group=wandb_group,
        name=f"{wandb_group}-evaluate",
        job_type="evaluate",
        dir="/dev/shm",
        id=f"{wandb_group}-evaluate",
        resume="allow",
        settings=wandb.Settings(
            start_method="fork",
        ),
    )
    use_cuda = torch.cuda.is_available() if cuda else False
    device = torch.device("cuda" if use_cuda else "cpu")
    set_torch_defaults(config["precision"])
    simulator_module = importlib.import_module(f"src.simulators.{simulator}")
    with open(os.path.join(data_path, "std.pkl"), "rb") as f:
        std = pickle.load(f)
    model_module = importlib.import_module(f"src.models.{model_name}")
    model = model_module.get_model(
        simulator_module.shape,
        cfg=model_config["model"],
        std=torch.from_numpy(std).to(torch.get_default_dtype()),
    )
    env_context, _ = set_seed(seed)
    dataloaders = get_dataloaders(
        path=data_path,
        batch_size=config["evaluation"]["batch_size"],
        cached=config["cached_dataset"][simulator],
        cached_device=torch.device(
            "cuda"
            if use_cuda and config["use_cuda_cached_dataset"][simulator]
            else "cpu"
        ),
        device=device,
        dataloaders=("test",),
    )
    load_model(path=model_path, model=model, map_location=device)
    model.to(device)
    model.eval()

    with env_context:
        if simulator_module.shape == (2,):
            n_samples = config["test"]
            samples = get_samples(
                model,
                n_samples,
                config["evaluation"]["batch_size"],
            ).numpy()
            np.save(os.path.join(out_path, "samples.npy"), samples)
            wandb.run.summary["mmd"] = (
                MMD(
                    bandwidth=config["evaluation"]["mmd_bandwidht"][simulator],
                    chunk_size=config["evaluation"]["batch_size"],
                )(
                    dataloaders["test"].dataset[:],
                    torch.from_numpy(samples).to(device),
                )
                .cpu()
                .item()
            )
        else:
            raise NotImplementedError()
    with open(
        os.path.join(metrics_summary_path, "metrics-summary-rsample.json"),
        "w",
    ) as f:
        if isinstance(wandb.run.summary, (Summary, SummaryDisabled)):
            json.dump(dict(wandb.run.summary), f)
        elif isinstance(wandb.run.summary, SummarySubDict):
            json.dump(wandb.run.summary._as_dict(), f)
        else:
            raise ValueError(
                f"Unrecognized wandb summary type - {type(wandb.run.summary)}"
            )


CV_DATASETS = ("cifar10",)


def evaluate_rsample_real(
    model_name,
    dataset,
    model_config,
    config,
    model_path,
    data_path,
    out_path,
    metrics_summary_path,
    seed,
    cuda,
):
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(model_path, "wandb_group"), "r") as f:
        wandb_group = f.read()
    wandb_project = f"ksgan-{model_name}-{dataset}"
    wandb.init(
        project=wandb_project,
        group=wandb_group,
        name=f"{wandb_group}-evaluate",
        job_type="evaluate",
        dir="/dev/shm",
        id=f"{wandb_group}-evaluate",
        resume="allow",
        settings=wandb.Settings(
            start_method="fork",
        ),
    )
    use_cuda = torch.cuda.is_available() if cuda else False
    device = torch.device("cuda" if use_cuda else "cpu")
    set_torch_defaults(config["precision"])
    dataset_module = importlib.import_module(f"src.datasets.{dataset}")
    with open(os.path.join(data_path, "std.pkl"), "rb") as f:
        std = pickle.load(f)
    model_module = importlib.import_module(f"src.models.{model_name}")
    model = model_module.get_model(
        dataset_module.shape,
        cfg=model_config["model"],
        std=torch.from_numpy(std).to(torch.get_default_dtype()),
    )
    env_context, _ = set_seed(seed)
    dataloaders = get_dataloaders(
        path=data_path,
        batch_size=config["evaluation"]["batch_size"],
        cached=config["cached_dataset"][dataset],
        cached_device=torch.device(
            "cuda"
            if use_cuda and config["use_cuda_cached_dataset"][dataset]
            else "cpu"
        ),
        device=device,
        dataloaders=("test",),
    )
    load_model(path=model_path, model=model, map_location=device)
    model.to(device)
    model.eval()

    with env_context:
        with torch.no_grad():
            if os.path.exists(f"src.classifiers.{dataset}.classifier.py"):
                classifier = importlib.import_module(
                    f"src.classifiers.{dataset}.classifier"
                ).Classifier()
                classifier.load_state_dict(
                    torch.load(f"src/classifiers/{dataset}/classifier.pt")
                )
                classifier.to(device)
                modecollapseeval = ModeCollapseEval(
                    classifier=classifier,
                    n_stack=model_config["model"]["n_stacked"],
                    z_dim=model_config["model"]["generator_cfg"]["parameters"][
                        "latent_dim"
                    ],
                )
                num_modes, kld = modecollapseeval.count_modes(model.generator)
                wandb.run.summary["num_modes"] = num_modes
                wandb.run.summary["kld"] = kld
            if dataset in CV_DATASETS:
                n_batches = 64
                lazy_generator_loader = SamplingIteratorWithRegister(
                    model, config["evaluation"]["batch_size"], n_batches
                )
                inception_score, inception_score_std = get_inception_score(
                    images=lazy_generator_loader, use_torch=True, device=device
                )
                wandb.run.summary["is"] = inception_score
                wandb.run.summary["is_std"] = inception_score_std
                if os.path.exists(
                    path := os.path.join(data_path, "fid_stats.npz")
                ):
                    pass
                else:
                    (acts,) = get_inception_feature(
                        (
                            next(
                                iter(dataloaders["test"].dataloaders)
                            ).dataset[:]
                            + 1.0
                        )
                        / 2.0,
                        dims=[2048],
                        use_torch=True,
                        verbose=True,
                        device=device,
                    )

                    mu = torch.mean(acts, dim=0).cpu().numpy()
                    sigma = torch_cov(acts, rowvar=False).cpu().numpy()
                    np.savez_compressed(
                        path,
                        mu=mu,
                        sigma=sigma,
                    )
                fid_score = get_fid(
                    torch.utils.data.DataLoader(
                        SingleTensorDataset(lazy_generator_loader.register),
                        batch_size=config["evaluation"]["batch_size"],
                        shuffle=False,
                        collate_fn=lambda batch: batch,
                    ),
                    path,
                    use_torch=True,
                    device=device,
                    verbose=True,
                )
                wandb.run.summary["fid"] = fid_score
    with open(
        os.path.join(metrics_summary_path, "metrics-summary-rsample.json"),
        "w",
    ) as f:
        if isinstance(wandb.run.summary, (Summary, SummaryDisabled)):
            json.dump(dict(wandb.run.summary), f)
        elif isinstance(wandb.run.summary, SummarySubDict):
            json.dump(wandb.run.summary._as_dict(), f)
        else:
            raise ValueError(
                f"Unrecognized wandb summary type - {type(wandb.run.summary)}"
            )


class SamplingIteratorWithRegister(torch.utils.data.DataLoader):
    def __init__(self, model, batch_size, n_batches):
        self.model = model
        self.bs = batch_size
        self.n_batches = n_batches
        super().__init__(dataset=self)

    def __len__(self):
        return self.n_batches * self.bs

    def __iter__(self):
        self.register = torch.empty(0)
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.n_batches:
            raise StopIteration
        with torch.no_grad():
            self.register = torch.cat(
                (
                    self.register,
                    sample := (
                        (self.model.sample(shape=(self.bs,)) + 1.0) / 2.0
                    ).cpu(),
                )
            )
        self.index += 1
        return sample


def evaluate_log_prob(
    model_name,
    simulator,
    model_config,
    config,
    model_path,
    data_path,
    out_path,
    metrics_summary_path,
    seed,
    cuda,
):
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(model_path, "wandb_group"), "r") as f:
        wandb_group = f.read()
    wandb_project = f"ksgan-{model_name}-{simulator}"
    wandb.init(
        project=wandb_project,
        group=wandb_group,
        name=f"{wandb_group}-evaluate",
        job_type="evaluate",
        dir="/dev/shm",
        id=f"{wandb_group}-evaluate",
        resume="allow",
        settings=wandb.Settings(
            start_method="fork",
        ),
    )
    use_cuda = torch.cuda.is_available() if cuda else False
    device = torch.device("cuda" if use_cuda else "cpu")
    set_torch_defaults(config["precision"])
    simulator_module = importlib.import_module(f"src.simulators.{simulator}")
    with open(os.path.join(data_path, "std.pkl"), "rb") as f:
        std = pickle.load(f)
    model_module = importlib.import_module(f"src.models.{model_name}")
    model = model_module.get_model(
        simulator_module.shape,
        cfg=model_config["model"],
        std=torch.from_numpy(std).to(torch.get_default_dtype()),
    )
    env_context, _ = set_seed(seed)
    dataloaders = get_dataloaders(
        path=data_path,
        batch_size=config["evaluation"]["batch_size"],
        cached=config["cached_dataset"][simulator],
        cached_device=torch.device(
            "cuda"
            if use_cuda and config["use_cuda_cached_dataset"][simulator]
            else "cpu"
        ),
        device=device,
        dataloaders=("test",),
    )
    load_model(path=model_path, model=model, map_location=device)
    model.to(device)
    model.eval()

    with env_context:
        log_probs = get_func_on_data(
            model,
            dataloader=dataloaders["test"],
        ).numpy()
        if simulator_module.shape == (2,):
            with open(os.path.join(data_path, "extent.pkl"), "rb") as f:
                extent = pickle.load(f)
            grid, log_volume_grid = get_grid(
                extent,
                resolution=config["evaluation"]["grid_resolution"],
                device=torch.device("cpu"),
            )
            log_prob_landscape = get_func_on_data(
                model.log_prob,
                x=grid,
                batch_size=config["evaluation"]["batch_size"],
                comp_device=device,
            )
            normalized_log_probs = log_probs - (
                log_prob_int := scipy.special.logsumexp(log_prob_landscape)
                + log_volume_grid
            )
            wandb.run.summary[f"test/n-log-prob"] = np.mean(
                normalized_log_probs
            ).item()
            wandb.run.summary[f"test/prob-integral"] = np.exp(
                log_prob_int
            ).item()
            np.save(
                os.path.join(out_path, "normalized_log_probs"),
                normalized_log_probs,
            )
            np.save(
                os.path.join(out_path, "log_prob_landscape"),
                log_prob_landscape,
            )
    np.save(os.path.join(out_path, "log_probs"), log_probs)
    wandb.run.summary[f"test/log-prob"] = np.mean(log_probs).item()
    with open(
        os.path.join(metrics_summary_path, "metrics-summary-log_prob.json"),
        "w",
    ) as f:
        if isinstance(wandb.run.summary, (Summary, SummaryDisabled)):
            json.dump(dict(wandb.run.summary), f)
        elif isinstance(wandb.run.summary, SummarySubDict):
            json.dump(wandb.run.summary._as_dict(), f)
        else:
            raise ValueError(
                f"Unrecognized wandb summary type - {type(wandb.run.summary)}"
            )


def evaluate_critic(
    model_name,
    simulator,
    model_config,
    config,
    model_path,
    data_path,
    out_path,
    metrics_summary_path,
    seed,
    cuda,
):
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(model_path, "wandb_group"), "r") as f:
        wandb_group = f.read()
    wandb_project = f"ksgan-{model_name}-{simulator}"
    wandb.init(
        project=wandb_project,
        group=wandb_group,
        name=f"{wandb_group}-evaluate",
        job_type="evaluate",
        dir="/dev/shm",
        id=f"{wandb_group}-evaluate",
        resume="allow",
        settings=wandb.Settings(
            start_method="fork",
        ),
    )
    use_cuda = torch.cuda.is_available() if cuda else False
    device = torch.device("cuda" if use_cuda else "cpu")
    set_torch_defaults(config["precision"])
    simulator_module = importlib.import_module(f"src.simulators.{simulator}")
    with open(os.path.join(data_path, "std.pkl"), "rb") as f:
        std = pickle.load(f)
    model_module = importlib.import_module(f"src.models.{model_name}")
    model = model_module.get_model(
        simulator_module.shape,
        cfg=model_config["model"],
        std=torch.from_numpy(std).to(torch.get_default_dtype()),
    )
    env_context, _ = set_seed(seed)
    dataloaders = get_dataloaders(
        path=data_path,
        batch_size=config["evaluation"]["batch_size"],
        cached=config["cached_dataset"][simulator],
        cached_device=torch.device(
            "cuda"
            if use_cuda and config["use_cuda_cached_dataset"][simulator]
            else "cpu"
        ),
        device=device,
        dataloaders=("test",),
    )
    load_model(path=model_path, model=model, map_location=device)
    model.to(device)
    model.eval()

    with env_context:
        critic_scores = get_func_on_data(
            model.critic,
            dataloader=dataloaders["test"],
        ).numpy()
        if simulator_module.shape == (2,):
            with open(os.path.join(data_path, "extent.pkl"), "rb") as f:
                extent = pickle.load(f)
            grid, log_volume_grid = get_grid(
                extent,
                resolution=config["evaluation"]["grid_resolution"],
                device=torch.device("cpu"),
            )
            critic_landscape = get_func_on_data(
                model.critic,
                x=grid,
                batch_size=config["evaluation"]["batch_size"],
                comp_device=device,
            )
            np.save(
                os.path.join(out_path, "critic_landscape"),
                critic_landscape,
            )
    np.save(os.path.join(out_path, "critic_scores"), critic_scores)
    wandb.run.summary[f"test/critic-score"] = np.mean(critic_scores).item()
    with open(
        os.path.join(metrics_summary_path, "metrics-summary-critic.json"),
        "w",
    ) as f:
        if isinstance(wandb.run.summary, (Summary, SummaryDisabled)):
            json.dump(dict(wandb.run.summary), f)
        elif isinstance(wandb.run.summary, SummarySubDict):
            json.dump(wandb.run.summary._as_dict(), f)
        else:
            raise ValueError(
                f"Unrecognized wandb summary type - {type(wandb.run.summary)}"
            )


### Based on https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py ###


class MMD(torch.nn.Module):

    def __init__(self, bandwidth, chunk_size=1024):
        super().__init__()
        self.bandwidth = bandwidth
        self.chunk_size = chunk_size

    def forward(self, X, Y):
        X_size = X.shape[0]
        Y_size = Y.shape[0]
        xx = (
            torch.vmap(
                lambda x: (-(X - x).pow(2).sum(-1) / self.bandwidth).logsumexp(
                    0
                ),
                chunk_size=self.chunk_size,
            )(X)
            .logsumexp(0)
            .exp()
            - X_size
        ) / (X_size * (X_size - 1))
        xy = torch.vmap(
            lambda y: (-(X - y).pow(2).sum(-1) / self.bandwidth).logsumexp(0),
            chunk_size=self.chunk_size,
        )(Y).logsumexp(0).exp() / (X_size * Y_size)
        yy = (
            torch.vmap(
                lambda y: (-(Y - y).pow(2).sum(-1) / self.bandwidth).logsumexp(
                    0
                ),
                chunk_size=self.chunk_size,
            )(Y)
            .logsumexp(0)
            .exp()
            - Y_size
        ) / (Y_size * (Y_size - 1))
        return xx - 2 * xy + yy


### Copied from https://github.com/ritheshkumar95/energy_based_generative_models/blob/master/scripts/evals.py ###


def KLD(p, q):
    if 0 in q:
        raise ValueError
    return sum(_p * np.log(_p / _q) for (_p, _q) in zip(p, q) if _p != 0)


class ModeCollapseEval:
    def __init__(self, classifier, n_stack, z_dim):
        self.classifier = classifier
        self.n_stack = n_stack
        self.n_samples = 26 * 10**n_stack
        self.z_dim = z_dim

    def count_modes(self, generator):
        counts = np.zeros([10] * self.n_stack)
        n_batches = max(1, self.n_samples // 1000)
        for i in tqdm(range(n_batches)):
            with torch.no_grad():
                x_fake = generator.sample(shape=(1000,))
                x_fake = x_fake.view(-1, 1, 28, 28)
                classes = torch.nn.functional.softmax(
                    self.classifier(x_fake), -1
                ).max(1)[1]
                classes = classes.view(1000, self.n_stack).cpu().numpy()

                for line in classes:
                    counts[tuple(line)] += 1

        n_modes = 10**self.n_stack
        true_data = np.ones(n_modes) / float(n_modes)
        num_modes_cap = len(np.where(counts > 0)[0])
        counts = counts.flatten() / counts.sum()
        kld = KLD(counts, true_data)
        return num_modes_cap, kld


######
