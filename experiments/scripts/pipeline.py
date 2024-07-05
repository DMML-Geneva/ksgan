import argparse
import os.path
import shutil

import yaml

from dawgz import schedule, job, context, ensure, after

from src.training.utils import train
from src.evaluation.utils import (
    evaluate_log_prob,
    evaluate_critic,
    evaluate_rsample,
    model_has_log_prob,
    model_has_critic,
    model_has_rsample,
)
from src.visualization.utils import visualize_training, visualize_evaluation

parser = argparse.ArgumentParser(description="Deploy pipeline")
parser.add_argument(
    "--out-path-prefix",
    type=str,
    default="data",
    metavar="OUTPATHPREFIX",
    help="Prefix for output path",
)
parser.add_argument(
    "--n-train-stages",
    type=int,
    default=10,
    metavar="NTRAIN",
    help="Number of train jobs",
)
parser.add_argument(
    "--config",
    type=str,
    default="./experiments/configs/global.yaml",
    metavar="CONFIG",
    help="Global config of experiments",
)
parser.add_argument(
    "--VramPerGpuGB",
    type=int,
    default=None,
    metavar="VramPerGpu",
    help="VramPerGpu",
)
parser.add_argument(
    "--clean",
    action=argparse.BooleanOptionalAction,
    default=False,
    metavar="CLEAN",
    help="Clean run flag, removes all existing data",
)
parser.add_argument(
    "--clean-evaluation",
    action=argparse.BooleanOptionalAction,
    default=False,
    metavar="CLEANEVAL",
    help="Clean run flag, removes all existing data of evaluation",
)
parser.add_argument(
    "--train-only",
    action=argparse.BooleanOptionalAction,
    default=False,
    metavar="TRAINONLY",
    help="Flag to run only training as its visualization",
)
parser.add_argument(
    "--not-prune",
    action=argparse.BooleanOptionalAction,
    default=False,
    metavar="NOTPRUNE",
    help="Flag to not prune DAG",
)
with open("dvc.yaml", "r") as fin:
    dvc_file = yaml.load(fin, Loader=yaml.SafeLoader)


def get_seed(vars):
    return next(
        filter(
            lambda item: isinstance(item, dict) and "seed" in item.keys(), vars
        )
    )["seed"]


parser.add_argument(
    "--seed",
    type=int,
    default=get_seed(dvc_file["vars"]),
    metavar="SEED",
    help="Seed",
)
parser.add_argument("--datasets", default=None, nargs="+")
parser.add_argument("--models", default=None, nargs="+")
parser.add_argument("--sbs", default=None, nargs="+")
parser.add_argument("--seeds", default=None, nargs="+")
parser.add_argument(
    "--job-time",
    type=str,
    default="12:00:00",
    metavar="TIME",
    help="Maximum time for the job",
)
parser.add_argument(
    "--cpus",
    type=int,
    default=2,
    metavar="CPU",
    help="Number of CPUs for job",
)
parser.add_argument(
    "--ram",
    type=int,
    default=32,
    metavar="RAM",
    help="Number of GBs of RAM for job",
)
parser.add_argument(
    "--backend",
    type=str,
    default="slurm",
    metavar="BACKEND",
    help="Backend to execute the job",
)


DATA_PATH_PREFIX = "data/preprocessed"
TRAINING_ARTIFACTS = ("wandb_group", "loss_history.pkl", "model.pt")
MINIMAL_EVALUATION_LOG_PROB_ARTIFACTS = ("log_probs.npy",)
OPTIONAL_EVALUATION_LOG_PROB_ARTIFACTS = (
    "normalized_log_probs.npy",
    "log_prob_landscape.npy",
)
METRICS_LOG_PROB_ARTIFACTS = ("metrics-summary-log_prob.json",)
EVALUATION_RSAMPLE_ARTIFACTS = ("samples.npy",)
METRICS_RSAMPLE_ARTIFACTS = ("metrics-summary-rsample.json",)
MINIMAL_EVALUATION_CRITIC_ARTIFACTS = ("critic_scores.npy",)
OPTIONAL_EVALUATION_CRITIC_ARTIFACTS = ("critic_landscape.npy",)
METRICS_CRITIC_ARTIFACTS = ("metrics-summary-critic.json",)
TRAINING_VIS_ARTIFACTS = ("training.png",)
MINIMAL_EVALUATION_VIS_ARTIFACTS = ("evaluation.png",)
OPTIONAL_EVALUATION_VIS_ARTIFACTS = ("hist.png",)


def split_into_chunks(num, div):
    return [num // div + (1 if x < num % div else 0) for x in range(div)]


def touch_files(path, files):
    os.makedirs(path, exist_ok=True)
    for file in files:
        try:
            open(
                os.path.join(path, file),
                "x",
            ).close()
        except FileExistsError:
            pass


def onerror_missing_dir(function, path, excinfo):
    if not os.path.isdir(path):
        pass
    else:
        raise excinfo[1]


def split_training(
    n,
    total_iterations,
    job_name,
    train_fn,
    out_path,
    clean=False,
    settings={},
    **kwargs,
):
    iterations_chunks = split_into_chunks(total_iterations, n)
    if len(iterations_chunks) == 1:
        if clean:
            clean_dir(out_path)
            shutil.rmtree(kwargs["vis_path"], onerror=onerror_missing_dir)

        @context(
            iterations_chunk=iterations_chunks[0],
            out_path=out_path,
            kwargs=kwargs,
        )
        @ensure(
            lambda: all(
                check_artifacts(out_path, artifacts=TRAINING_ARTIFACTS)
            )
        )
        @job(name=job_name, settings=settings)
        def train():
            train_fn(
                iterations=iterations_chunk,
                out_path=out_path,
                cuda=True,
                **kwargs,
            )

        return train
    elif len(iterations_chunks) == 2:
        if clean:
            clean_dir(os.path.join(out_path, "0"))

        @context(
            iterations_chunk=iterations_chunks[0],
            out_path=os.path.join(out_path, "0"),
            kwargs=kwargs,
        )
        @ensure(
            lambda: all(
                check_artifacts(out_path, artifacts=TRAINING_ARTIFACTS)
            )
        )
        @job(name=f"{job_name}_0", settings=settings)
        def train_0():
            train_fn(
                iterations=iterations_chunk,
                out_path=out_path,
                cuda=True,
                **kwargs,
            )

        if clean:
            clean_dir(out_path)
            shutil.rmtree(kwargs["vis_path"], onerror=onerror_missing_dir)

        @after(train_0)
        @context(
            iterations_chunk=iterations_chunks[1],
            out_path=out_path,
            checkpoint_path=os.path.join(out_path, "0"),
            kwargs=kwargs,
        )
        @ensure(
            lambda: all(
                check_artifacts(out_path, artifacts=TRAINING_ARTIFACTS)
            )
        )
        @job(name=f"{job_name}_1", settings=settings)
        def train_last():
            train_fn(
                iterations=iterations_chunk,
                out_path=out_path,
                checkpoint_path=checkpoint_path,
                cuda=True,
                **kwargs,
            )
            shutil.rmtree(checkpoint_path)

        return train_last
    else:
        if clean:
            clean_dir(os.path.join(out_path, "0"))

        @context(
            iterations_chunk=iterations_chunks[0],
            out_path=os.path.join(out_path, "0"),
            kwargs=kwargs,
        )
        @ensure(
            lambda: all(
                check_artifacts(out_path, artifacts=TRAINING_ARTIFACTS)
            )
        )
        @job(name=f"{job_name}_0", settings=settings)
        def train():
            train_fn(
                iterations=iterations_chunk,
                out_path=out_path,
                cuda=True,
                **kwargs,
            )

        for i, iterations in enumerate(iterations_chunks[1:-1], start=1):
            if clean:
                clean_dir(os.path.join(out_path, str(i)))

            @after(train)
            @context(
                i=i,
                iterations_chunk=iterations,
                checkpoint_path=os.path.join(out_path, str(i - 1)),
                out_path=os.path.join(out_path, str(i)),
                kwargs=kwargs,
            )
            @ensure(
                lambda: all(
                    check_artifacts(
                        out_path,
                        artifacts=TRAINING_ARTIFACTS,
                    )
                )
            )
            @job(name=f"{job_name}_{i}", settings=settings)
            def train():
                train_fn(
                    iterations=iterations_chunk,
                    checkpoint_path=checkpoint_path,
                    out_path=out_path,
                    cuda=True,
                    **kwargs,
                )

        if clean:
            clean_dir(out_path)
            shutil.rmtree(kwargs["vis_path"], onerror=onerror_missing_dir)

        @after(train)
        @context(
            last_i=i,
            iterations_chunk=iterations_chunks[-1],
            checkpoint_path=os.path.join(out_path, str(i)),
            out_path=out_path,
            kwargs=kwargs,
        )
        @ensure(
            lambda: all(
                check_artifacts(out_path, artifacts=TRAINING_ARTIFACTS)
            )
        )
        @job(
            name=f"{job_name}_{len(iterations_chunks) - 1}", settings=settings
        )
        def train_last():
            train_fn(
                iterations=iterations_chunk,
                out_path=out_path,
                checkpoint_path=checkpoint_path,
                cuda=True,
                **kwargs,
            )
            for j in range(last_i + 1):
                if os.path.exists(
                    path_to_rm := os.path.join(out_path, str(j))
                ):
                    shutil.rmtree(path_to_rm)

        return train_last


def clean_dir(path, artifacts=TRAINING_ARTIFACTS):
    for artifact in artifacts:
        if os.path.exists(artifact_path := os.path.join(path, artifact)):
            os.remove(artifact_path)


def check_artifacts(path, artifacts=TRAINING_ARTIFACTS):
    for artifact in artifacts:
        yield os.path.exists(os.path.join(path, artifact))


def main():
    args = parser.parse_args()
    OUT_PATH_PREFIX = args.out_path_prefix
    assert not (
        args.train_only and args.clean_evaluation
    ), "--train-only and --clean-evaluation does not make sense"
    with open(args.config, "r") as fin:
        cfg = yaml.load(fin, Loader=yaml.SafeLoader)

    datasets = tuple(extract_from_list(args.datasets, "datasets", cfg))
    models = tuple(extract_model_name_type(args.models, cfg["models"]))
    sbs = tuple(extract_from_list(args.sbs, "simulation-budgets", cfg))
    seeds = tuple(extract_from_list(args.seeds, "model_seeds", cfg))

    if args.backend == "slurm":
        vis_settings = {
            "cpus": 2,
            "gpus": False,
            "ram": "4000",
            "timelimit": "30:00",
            "partition": os.environ.get(
                "CPU_PARTITIONS", "shared-cpu,public-cpu"
            ),
        }
        settings = {
            "cpus": args.cpus,
            "gres": (
                "gpu:1"
                if args.VramPerGpuGB is None
                else f"gpu:1,VramPerGpu:{args.VramPerGpuGB}G"
            ),
            "ram": f"{args.ram}000",
            "timelimit": args.job_time,
            "partition": os.environ.get(
                "GPU_PARTITIONS", "shared-gpu,public-gpu"
            ),
        }
        if os.environ.get("GPU_CONSTRAINTS", None) is not None:
            settings["constraint"] = os.environ.get("GPU_CONSTRAINTS")
    else:
        vis_settings = dict()
        settings = dict()

    leaf_jobs = []
    for dataset in datasets:
        for model_name, model_type in models:
            for sb in sbs:
                for m_seed in seeds:
                    with open(
                        f"experiments/configs/{dataset}/{model_name}.yaml", "r"
                    ) as fin:
                        model_cfg = yaml.load(fin, Loader=yaml.SafeLoader)

                        last_train_job = split_training(
                            n=args.n_train_stages,
                            total_iterations=model_cfg["train"]["iterations"],
                            job_name=f"train-{model_name}-{dataset}-{sb}-{m_seed}",
                            train_fn=train,
                            model_name=model_type,
                            simulator=dataset,
                            config=model_cfg,
                            config_global=cfg,
                            simulation_budget=sb,
                            data_path=os.path.join(DATA_PATH_PREFIX, dataset),
                            out_path=os.path.join(
                                OUT_PATH_PREFIX,
                                "models",
                                dataset,
                                model_name,
                                str(sb),
                                str(m_seed),
                            ),
                            vis_path=os.path.join(
                                OUT_PATH_PREFIX,
                                "models",
                                dataset,
                                model_name,
                                str(sb),
                                str(m_seed),
                                "vis",
                            ),
                            seed=args.seed,
                            model_seed=m_seed,
                            clean=args.clean,
                            settings=settings,
                        )

                        if args.clean:
                            clean_dir(
                                os.path.join(
                                    OUT_PATH_PREFIX,
                                    "visualization",
                                    dataset,
                                    model_name,
                                    str(sb),
                                    str(m_seed),
                                ),
                                artifacts=TRAINING_VIS_ARTIFACTS,
                            )

                        @after(last_train_job)
                        @context(
                            history_file=os.path.join(
                                OUT_PATH_PREFIX,
                                "models",
                                dataset,
                                model_name,
                                str(sb),
                                str(m_seed),
                                "loss_history.pkl",
                            ),
                            out_path=os.path.join(
                                OUT_PATH_PREFIX,
                                "visualization",
                                dataset,
                                model_name,
                                str(sb),
                                str(m_seed),
                            ),
                            model_name=model_name,
                            dataset=dataset,
                            sb=sb,
                            m_seed=m_seed,
                        )
                        @ensure(
                            lambda: all(
                                check_artifacts(
                                    out_path, artifacts=TRAINING_VIS_ARTIFACTS
                                )
                            )
                        )
                        @job(
                            name=f"visualize-training-{model_name}-{dataset}-{sb}-{m_seed}",
                            settings=vis_settings,
                        )
                        def visualize_training_job():
                            visualize_training(
                                history_file=history_file, out_path=out_path
                            )

                        leaf_jobs.append(visualize_training_job)

                        if not args.train_only:
                            evaluate_leaf_jobs = []

                            if args.clean or args.clean_evaluation:
                                clean_dir(
                                    os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                        "artifacts",
                                    ),
                                    artifacts=MINIMAL_EVALUATION_LOG_PROB_ARTIFACTS
                                    + OPTIONAL_EVALUATION_LOG_PROB_ARTIFACTS,
                                )
                                clean_dir(
                                    os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                    ),
                                    artifacts=METRICS_LOG_PROB_ARTIFACTS,
                                )

                            if model_has_log_prob(model_type):

                                @after(last_train_job)
                                @context(
                                    model_name=model_name,
                                    model_type=model_type,
                                    dataset=dataset,
                                    model_config=model_cfg,
                                    config=cfg,
                                    sb=sb,
                                    m_seed=m_seed,
                                    seed=args.seed,
                                    out_path=os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                        "artifacts",
                                    ),
                                    model_path=os.path.join(
                                        OUT_PATH_PREFIX,
                                        "models",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                    ),
                                    metrics_summary_path=os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                    ),
                                    data_path=os.path.join(
                                        DATA_PATH_PREFIX, dataset
                                    ),
                                )
                                @ensure(
                                    lambda: all(
                                        check_artifacts(
                                            out_path,
                                            artifacts=MINIMAL_EVALUATION_LOG_PROB_ARTIFACTS,
                                        )
                                    )
                                    and all(
                                        check_artifacts(
                                            metrics_summary_path,
                                            artifacts=METRICS_LOG_PROB_ARTIFACTS,
                                        )
                                    )
                                )
                                @job(
                                    name=f"evaluate-log_prob-{model_name}-{dataset}-{sb}-{m_seed}",
                                    settings=settings,
                                )
                                def evaluate_log_prob_job():
                                    evaluate_log_prob(
                                        model_name=model_type,
                                        simulator=dataset,
                                        model_config=model_config,
                                        config=config,
                                        model_path=model_path,
                                        data_path=data_path,
                                        out_path=out_path,
                                        metrics_summary_path=metrics_summary_path,
                                        seed=seed,
                                        cuda=True,
                                    )

                                evaluate_leaf_jobs.append(
                                    evaluate_log_prob_job
                                )
                            else:
                                touch_files(
                                    os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                    ),
                                    METRICS_LOG_PROB_ARTIFACTS,
                                )

                            if args.clean or args.clean_evaluation:
                                clean_dir(
                                    os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                        "artifacts",
                                    ),
                                    artifacts=EVALUATION_RSAMPLE_ARTIFACTS,
                                )
                                clean_dir(
                                    os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                    ),
                                    artifacts=METRICS_RSAMPLE_ARTIFACTS,
                                )

                            if model_has_rsample(model_type):

                                @after(last_train_job)
                                @context(
                                    model_name=model_name,
                                    model_type=model_type,
                                    dataset=dataset,
                                    model_config=model_cfg,
                                    config=cfg,
                                    sb=sb,
                                    m_seed=m_seed,
                                    seed=args.seed,
                                    out_path=os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                        "artifacts",
                                    ),
                                    model_path=os.path.join(
                                        OUT_PATH_PREFIX,
                                        "models",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                    ),
                                    metrics_summary_path=os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                    ),
                                    data_path=os.path.join(
                                        DATA_PATH_PREFIX, dataset
                                    ),
                                )
                                @ensure(
                                    lambda: all(
                                        check_artifacts(
                                            out_path,
                                            artifacts=EVALUATION_RSAMPLE_ARTIFACTS,
                                        )
                                    )
                                    and all(
                                        check_artifacts(
                                            metrics_summary_path,
                                            artifacts=METRICS_RSAMPLE_ARTIFACTS,
                                        )
                                    )
                                )
                                @job(
                                    name=f"evaluate-rsample-{model_name}-{dataset}-{sb}-{m_seed}",
                                    settings=settings,
                                )
                                def evaluate_rsample_job():
                                    evaluate_rsample(
                                        model_name=model_type,
                                        simulator=dataset,
                                        model_config=model_config,
                                        config=config,
                                        model_path=model_path,
                                        data_path=data_path,
                                        out_path=out_path,
                                        metrics_summary_path=metrics_summary_path,
                                        seed=seed,
                                        cuda=True,
                                    )

                                evaluate_leaf_jobs.append(evaluate_rsample_job)
                            else:
                                touch_files(
                                    os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                    ),
                                    METRICS_RSAMPLE_ARTIFACTS,
                                )

                            if args.clean or args.clean_evaluation:
                                clean_dir(
                                    os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                        "artifacts",
                                    ),
                                    artifacts=MINIMAL_EVALUATION_CRITIC_ARTIFACTS
                                    + OPTIONAL_EVALUATION_CRITIC_ARTIFACTS,
                                )
                                clean_dir(
                                    os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                    ),
                                    artifacts=METRICS_CRITIC_ARTIFACTS,
                                )

                            if model_has_critic(model_type):

                                @after(last_train_job)
                                @context(
                                    model_name=model_name,
                                    model_type=model_type,
                                    dataset=dataset,
                                    model_config=model_cfg,
                                    config=cfg,
                                    sb=sb,
                                    m_seed=m_seed,
                                    seed=args.seed,
                                    out_path=os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                        "artifacts",
                                    ),
                                    model_path=os.path.join(
                                        OUT_PATH_PREFIX,
                                        "models",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                    ),
                                    metrics_summary_path=os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                    ),
                                    data_path=os.path.join(
                                        DATA_PATH_PREFIX, dataset
                                    ),
                                )
                                @ensure(
                                    lambda: all(
                                        check_artifacts(
                                            out_path,
                                            artifacts=MINIMAL_EVALUATION_CRITIC_ARTIFACTS,
                                        )
                                    )
                                    and all(
                                        check_artifacts(
                                            metrics_summary_path,
                                            artifacts=METRICS_CRITIC_ARTIFACTS,
                                        )
                                    )
                                )
                                @job(
                                    name=f"evaluate-critic-{model_name}-{dataset}-{sb}-{m_seed}",
                                    settings=settings,
                                )
                                def evaluate_critic_job():
                                    evaluate_critic(
                                        model_name=model_type,
                                        simulator=dataset,
                                        model_config=model_config,
                                        config=config,
                                        model_path=model_path,
                                        data_path=data_path,
                                        out_path=out_path,
                                        metrics_summary_path=metrics_summary_path,
                                        seed=seed,
                                        cuda=True,
                                    )

                                evaluate_leaf_jobs.append(evaluate_critic_job)
                            else:
                                touch_files(
                                    os.path.join(
                                        OUT_PATH_PREFIX,
                                        "evaluation",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                    ),
                                    METRICS_CRITIC_ARTIFACTS,
                                )

                            if args.clean or args.clean_evaluation:
                                clean_dir(
                                    os.path.join(
                                        OUT_PATH_PREFIX,
                                        "visualization",
                                        dataset,
                                        model_name,
                                        str(sb),
                                        str(m_seed),
                                        "artifacts",
                                    ),
                                    artifacts=MINIMAL_EVALUATION_VIS_ARTIFACTS
                                    + OPTIONAL_EVALUATION_VIS_ARTIFACTS,
                                )

                            @after(*evaluate_leaf_jobs)
                            @context(
                                evaluation_dir=os.path.join(
                                    OUT_PATH_PREFIX,
                                    "evaluation",
                                    dataset,
                                    model_name,
                                    str(sb),
                                    str(m_seed),
                                    "artifacts",
                                ),
                                out_path=os.path.join(
                                    OUT_PATH_PREFIX,
                                    "visualization",
                                    dataset,
                                    model_name,
                                    str(sb),
                                    str(m_seed),
                                    "artifacts",
                                ),
                                model_name=model_name,
                                dataset=dataset,
                                sb=sb,
                                m_seed=m_seed,
                                config=cfg,
                                data_path=os.path.join(
                                    DATA_PATH_PREFIX, dataset
                                ),
                            )
                            @ensure(
                                lambda: all(
                                    check_artifacts(
                                        out_path,
                                        artifacts=MINIMAL_EVALUATION_VIS_ARTIFACTS,
                                    )
                                )
                            )
                            @job(
                                name=f"visualize-evaluation-{model_name}-{dataset}-{sb}-{m_seed}",
                                settings=vis_settings,
                            )
                            def visualize_evaluation_job():
                                visualize_evaluation(
                                    evaluation_dir=evaluation_dir,
                                    out_path=out_path,
                                    config=config,
                                    simulator=dataset,
                                    data_path=data_path,
                                )

                            leaf_jobs.append(visualize_evaluation_job)

    if args.backend == "async":
        schedule(
            *leaf_jobs, backend="async", prune=not args.not_prune, pools=1
        )
    elif args.backend == "slurm":
        jobs_queue = schedule(
            *leaf_jobs,
            backend="slurm",
            prune=not args.not_prune,
            shell="/bin/bash",
            env=["#SBATCH --export=ALL,PYTHONOPTIMIZE=1"],
        )
        with open(f"{jobs_queue.path}/jobids", "w") as f:
            for line in jobs_queue.results.values():
                f.write(f"{line}\n")
    elif args.backend == "dummy":
        schedule(*leaf_jobs, backend="dummy", prune=not args.not_prune)
    else:
        raise ValueError(f"Unknown backend - `{args.backend}`!")


def extract_from_list(args_list, key, cfg):
    if args_list is not None:
        args_list = tuple(map(type(next(iter(cfg[key]))), args_list))
        assert set(args_list).issubset(
            cfg[key]
        ), f"Some of the {key} do not exist"
        return args_list
    else:
        return cfg[key]


def extract_model_from_list_of_dicts(args_list, cfg, dict_key="name"):
    cfg_members = tuple(map(lambda d: d[dict_key], cfg))
    if args_list is not None:
        args_list = tuple(map(type(next(iter(cfg_members))), args_list))
        if len(args_list) == 1 and args_list[0] == "None":
            return tuple()
        assert set(args_list).issubset(
            cfg_members
        ), f"Some of the models do not exist"
        return args_list
    else:
        return cfg_members


def extract_model_name_type(args_models_list, models_list):
    names = extract_model_from_list_of_dicts(args_models_list, models_list)
    return map(
        lambda model: (model["name"], model["model"]),
        [
            next(
                filter(
                    lambda model: model["name"] == name,
                    models_list,
                )
            )
            for name in names
        ],
    )


if __name__ == "__main__":
    main()
