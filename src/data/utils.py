import collections.abc as container_abcs
from itertools import groupby
import math
import os
import random
import re
import shutil
from queue import Queue
from threading import Thread
from typing import Any, Optional, Union

import numpy as np
import tensorstore as ts
from tensorstore import TensorStore
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def archive_tensorstore(path, delete=False):
    shutil.make_archive(
        base_name=path,
        format="zip",
        root_dir=path,
        base_dir=".",
    )
    if delete:
        shutil.rmtree(path)


def open_tensorstore(path, config=None) -> TensorStore:
    if config is None:
        config = dict()
    return ts.open(
        {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": path,
            },
            "context": {
                "cache_pool": {
                    "total_bytes_limit": int(
                        os.environ.get("TS_IN_MEMORY_CACHE", 100_000_000)
                    )
                }
            },
            **config,
        }
    ).result(timeout=None)


def open_archived_tensorstore(path, config=None) -> TensorStore:
    if config is None:
        config = dict()
    return ts.open(
        {
            "driver": "zarr",
            "kvstore": {
                "driver": "zip",
                "base": {
                    "driver": "file",
                    "path": f"{path}.zip",
                },
            },
            "context": {
                "cache_pool": {
                    "total_bytes_limit": int(
                        os.environ.get("TS_IN_MEMORY_CACHE", 100_000_000)
                    )
                }
            },
            **config,
        },
        write=False,
    ).result(timeout=None)


BIG_ENDIAN_DTYPE_DICT = {16: "<f2", 32: "<f4", 64: "<f8"}
DATA_PRECISION = 32


class TensorstoreDataset(Dataset):
    def __init__(self, path, n=-1, start_idx=0):
        if torch.get_default_dtype() == torch.float16:
            dtype = ts.float16
        elif torch.get_default_dtype() == torch.float32:
            dtype = ts.float32
        elif torch.get_default_dtype() == torch.float64:
            dtype = ts.float64
        else:
            dtype = ts.float32
        self.data = open_archived_tensorstore(
            path,
        ).astype(dtype)
        if n != -1:
            self.data = self.data[start_idx : start_idx + n]
        else:
            self.data = self.data[start_idx:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].read().result()).to(
            torch.get_default_dtype()
        )

    def __getitems__(self, idxs):
        return torch.from_numpy(self.data[idxs].read().result()).to(
            torch.get_default_dtype()
        )


class SingleTensorDataset(Dataset):
    def __init__(self, x, n=-1, start_idx=0):
        self.data = x
        if n != -1:
            self.data = self.data[start_idx : start_idx + n]
        else:
            self.data = self.data[start_idx:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].to(torch.get_default_dtype())

    def __getitems__(self, idxs):
        return self.data[idxs].to(torch.get_default_dtype())


class CachedTensorstoreDataset(TensorstoreDataset):
    def __init__(self, path, device, n=-1, start_idx=0):
        super(CachedTensorstoreDataset, self).__init__(
            path=path, n=n, start_idx=start_idx
        )
        self.data = torch.from_numpy(self.data[:].read().result()).to(
            device, torch.get_default_dtype()
        )

    def __getitem__(self, idx):
        return self.data[idx]

    def __getitems__(self, idxs):
        return self.data[idxs]


def seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ZippedDataLoader:
    def __init__(self, *dataloaders):
        if not (
            all_equal(map(len, dataloaders))
            and all_equal(map(lambda d: d.batch_size, dataloaders))
        ):
            raise ValueError(
                "`dataloaders` must have the same length with same batch size"
            )
        self.dataloaders = tuple(dataloaders)
        self._len = len(next(iter(self.dataloaders)))

    def __iter__(self):
        self._iterators = zip(*map(iter, self.dataloaders))
        return self

    def __len__(self):
        return self._len

    def n_instances(self):
        return len(next(iter(self.dataloaders)).dataset)

    def __next__(self):
        return torch.stack(
            next(self._iterators),
            dim=1,
        ).squeeze(1)

    @property
    def dataset(self):
        return next(iter(self.dataloaders)).dataset


def get_dataloaders(
    path,
    batch_size,
    device,
    dataloaders=("train", "validation", "test"),
    cached=False,
    cached_device=None,
    simulation_budget=-1,
    validation_fraction=None,
    seed=None,
    q_size=10,
    test_start_idx=0,
    test_n=-1,
    n_stacked=1,
    k=1,
):
    if not cached:
        paired_dataset = TensorstoreDataset
    else:
        if device.type == "cpu":
            cached_device = device
        else:
            assert isinstance(
                cached_device, torch.device
            ), "`cached_device` has to be `torch.device` when using cached dataet!"
        paired_dataset = (
            lambda path, n=-1, start_idx=0: CachedTensorstoreDataset(
                path=path, device=cached_device, n=n, start_idx=start_idx
            )
        )
    dls = dict()
    for dl in dataloaders:
        if dl == "train":
            if seed is not None:
                dataloader_config = []
                for n in range(n_stacked):
                    g = torch.Generator()
                    g.manual_seed(seed + n)
                    dataloader_config.append(
                        {
                            "worker_init_fn": seed_dataloader_worker,
                            "generator": g,
                        }
                    )
            else:
                dataloader_config = [{"shuffle": True}] * n_stacked
            train_dataset = paired_dataset(
                path=os.path.join(path, "train"), n=simulation_budget
            )
            dls["train"] = ZippedDataLoader(
                *(
                    DataLoader(
                        train_dataset,
                        batch_sampler=torch.utils.data.BatchSampler(
                            sampler=torch.utils.data.RandomSampler(
                                data_source=train_dataset,
                                replacement=True,
                                num_samples=len(train_dataset) * k,
                                generator=dc.pop("generator"),
                            ),
                            batch_size=batch_size,
                            drop_last=True,
                        ),
                        collate_fn=lambda batch: batch,
                        **dc,
                    )
                    for dc in dataloader_config
                )
            )
        elif dl == "validation":
            validation_dataset = paired_dataset(
                path=os.path.join(path, "validation"),
                n=(
                    math.floor(
                        validation_fraction * dls["train"].n_instances()
                    )
                    if validation_fraction is not None
                    else -1
                ),
            )
            dls["validation"] = ZippedDataLoader(
                *(
                    DataLoader(
                        validation_dataset,
                        batch_size=batch_size,
                        drop_last=False,
                        collate_fn=lambda batch: batch,
                        shuffle=False,
                    )
                    for _ in range(n_stacked)
                )
            )
        elif dl == "test":
            test_dataset = paired_dataset(
                path=os.path.join(path, dl),
                n=test_n,
                start_idx=test_start_idx,
            )
            dls[dl] = ZippedDataLoader(
                *(
                    DataLoader(
                        test_dataset,
                        batch_size=batch_size,
                        drop_last=False,
                        collate_fn=lambda batch: batch,
                        shuffle=True,
                    )
                    for _ in range(n_stacked)
                )
            )
        else:
            raise ValueError(f"Unknown data split name: `{dl}`")
    if device.type != "cpu" and (
        not cached or (cached and cached_device != device)
    ):
        for k, v in dls.items():
            dls[k] = AsynchronousLoader(v, device=device, q_size=q_size)
    return dls


class AsynchronousLoader:
    # Copied from pl_bolts.datamodules.async_dataloader
    """Class for asynchronously loading from CPU memory to device memory with DataLoader.

    Note that this only works for single GPU training, multiGPU uses PyTorch's DataParallel or
    DistributedDataParallel which uses its own code for transferring data across GPUs. This could just
    break or make things slower with DataParallel or DistributedDataParallel.

    Args:
        data: The PyTorch Dataset or DataLoader we're using to load.
        device: The PyTorch device we are loading to
        q_size: Size of the queue used to store the data loaded to the device
        num_batches: Number of batches to load. This must be set if the dataloader
            doesn't have a finite __len__. It will also override DataLoader.__len__
            if set and DataLoader has a __len__. Otherwise it can be left as None
        **kwargs: Any additional arguments to pass to the dataloader if we're
            constructing one here
    """

    def __init__(
        self,
        data: Union[DataLoader, Dataset],
        device: torch.device = torch.device("cuda", 0),
        q_size: int = 10,
        num_batches: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(data, torch.utils.data.DataLoader):
            self.dataloader = data
        else:
            self.dataloader = DataLoader(data, **kwargs)

        if num_batches is not None:
            self.num_batches = num_batches
        elif hasattr(self.dataloader, "__len__"):
            self.num_batches = len(self.dataloader)
        else:
            raise Exception(
                "num_batches must be specified or data must have finite __len__"
            )

        self.device = device
        self.q_size = q_size

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue: Queue = Queue(maxsize=self.q_size)

        self.idx = 0

        self.np_str_obj_array_pattern = re.compile(r"[SaUO]")

    def load_loop(
        self,
    ) -> None:  # The loop that will load into the queue in the background
        for i, sample in enumerate(self.dataloader):
            self.queue.put(self.load_instance(sample))
            if i == len(self):
                break

    # Recursive loading for each instance based on torch.utils.data.default_collate
    def load_instance(self, sample: Any) -> Any:
        elem_type = type(sample)

        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                # Can only do asynchronous transfer if we use pin_memory
                if not sample.is_pinned():
                    sample = sample.pin_memory()
                return sample.to(self.device, non_blocking=True)
        elif (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
        ):
            if (
                elem_type.__name__ == "ndarray"
                and self.np_str_obj_array_pattern.search(sample.dtype.str)
                is not None
            ):
                return self.load_instance(sample)
            return self.load_instance(torch.as_tensor(sample))
        elif isinstance(sample, container_abcs.Mapping):
            return {key: self.load_instance(sample[key]) for key in sample}
        elif isinstance(sample, tuple) and hasattr(
            sample, "_fields"
        ):  # namedtuple
            return elem_type(*(self.load_instance(d) for d in sample))
        elif isinstance(sample, container_abcs.Sequence) and not isinstance(
            sample, str
        ):
            return [self.load_instance(s) for s in sample]
        else:
            return sample

    def __iter__(self) -> "AsynchronousLoader":
        # We don't want to run the thread more than once
        # Start a new thread if we are at the beginning of a new epoch, and our current worker is dead
        if_worker = not hasattr(self, "worker") or not self.worker.is_alive()  # type: ignore[has-type]
        if if_worker and self.queue.empty() and self.idx == 0:
            self.worker = Thread(target=self.load_loop)
            self.worker.daemon = True
            self.worker.start()
        return self

    def __next__(self) -> Tensor:
        # If we've reached the number of batches to return
        # or the queue is empty and the worker is dead then exit
        done = not self.worker.is_alive() and self.queue.empty()
        done = done or self.idx >= len(self)
        if done:
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # Otherwise return the next batch
        out = self.queue.get()
        self.queue.task_done()
        self.idx += 1
        return out

    def __len__(self) -> int:
        return self.num_batches
