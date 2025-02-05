import os
import h5py
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

# import torchdata.datapipes as dp
# from torchdata.datapipes.iter import IterDataPipe

import torch.utils.data.datapipes as dp
from torch.utils.data.datapipes.datapipe import IterDataPipe

from logging import getLogger

logger = getLogger()

DatasetIdx = {
    "react_diff": 0,
    "shallow_water": 1,
    "incom_ns": 2,
    "com_ns": 3,
    "incom_ns_arena": 4,
    "incom_ns_arena_u": 5,
    "cfdbench": 6,
}


class myIterDp(IterDataPipe):
    """
    Base class for all iterable datasets, and contains some shared helper methods.
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__()

        # general initialization, should be called by all subclasses

        # self.train = split == "train"
        self.train = train
        self.params = params
        self.symbol_env = symbol_env
        self.split = split

        self.num_workers = params.num_workers if train else params.num_workers_eval
        self.local_rank = params.local_rank
        self.n_gpu_per_node = params.n_gpu_per_node

        self.t_num = params.data.t_num
        self.x_num = params.data.x_num

        if params.overfit_test:
            self.random_start = params.data.random_start["test"]
        else:
            self.random_start = params.data.random_start[split]

        self.rng = None
        self.type_label = ""
        self.fully_shuffled = False
        self.symbol_ids = None

    def init_rng(self):
        """
        Initialize different random generator for each worker.
        """
        if self.rng is not None:
            return

        worker_id = self.get_worker_id()
        self.worker_id = worker_id
        params = self.params
        if self.train:
            # base_seed = params.base_seed
            base_seed = np.random.randint(1_000_000_000)  # ensure seed is different for each epoch

            seed = [worker_id, DatasetIdx[self.type_label], params.global_rank, base_seed]
            self.rng = np.random.default_rng(seed)
            # logger.info(f"Initialize random generator with seed {seed} (worker, dataset, rank, base_seed)")
        else:
            seed = [worker_id, DatasetIdx[self.type_label], params.global_rank, params.test_seed]
            self.rng = np.random.default_rng(seed)
            # logger.info(f"Initialize random generator with seed {seed} (worker, dataset, rank, test_seed)")

    def get_worker_id(self):
        worker_info = torch.utils.data.get_worker_info()
        return 0 if worker_info is None else worker_info.id

    def augment_data(self, data: np.ndarray):
        """
        data: (t_num, x_num, x_num, data_dim)
        """
        if self.train:
            # self.init_rng()
            if self.params.noise > 0:
                # add noise
                gamma = self.params.noise

                if self.params.noise_type == "multiplicative":
                    cur_noise = self.rng.normal(size=data.shape).astype(np.single)
                    data = data + gamma * np.abs(data) * cur_noise
                elif self.params.noise_type == "additive":
                    cur_noise = self.rng.normal(size=data.shape).astype(np.single)
                    eps = 1e-6
                    sigma = gamma * np.linalg.norm(data) / (np.linalg.norm(cur_noise) + eps)
                    data = data + sigma * cur_noise

            if self.params.flip:
                # flip data
                flip = self.rng.choice(4)
                if flip == 1:
                    data = np.flip(data, axis=1)
                elif flip == 2:
                    data = np.flip(data, axis=2)
                elif flip == 3:
                    data = np.flip(data, axis=(1, 2))

            if self.params.rotate:
                # rotate data
                rot = self.rng.choice(4)
                if rot > 0:
                    data = np.rot90(data, axes=(1, 2), k=rot)

        return np.ascontiguousarray(data)

    def get_iter_range(self, total_len):
        # split data based on train/val/test ratio and number of workers
        ratio = self.params.data.train_val_test_ratio
        start1 = int(total_len * ratio[0])
        start2 = int(total_len * (ratio[0] + ratio[1]))
        start3 = int(total_len * (ratio[0] + ratio[1] + ratio[2]))
        if self.split == "train":
            start = 0
            end = start1
        elif self.split == "val":
            start = start1
            end = start2
        else:  # test
            start = start2
            end = start3

        if self.num_workers <= 1:
            # return start, end
            return np.arange(start, end)
        else:
            # subdivide based on number of workers
            return np.arange(start + self.worker_id, end, self.num_workers)

    def sample_initial_time(self, max_len):
        data_limit = max_len - self.t_num * self.t_step
        start_limit = self.params.data.random_start.start_max
        if start_limit > 0:
            data_limit = min(data_limit, start_limit)
        if data_limit <= 0:
            return 0
        else:
            return self.rng.integers(0, data_limit)


class ReactDiff2D(myIterDp):
    """
    PDEBench 2D reaction_diffusion dataset.
        size:  1000
        t_num: 101           [0, 5] dt=0.05
        x_num: (128, 128)    (-1, 1)
        data_dim: 2
        bc: neumann

    Dataset structure:
    0000 - 0999
        data: (101, 128, 128, 2)
        grid
            t: (101,)
            x: (128,)
            y: (128,)
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "react_diff"

        self.t_step = params.data.react_diff.t_step
        self.x_step = params.data.react_diff.x_num // params.data.x_num
        self.data_path = params.data.react_diff.data_path
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def __iter__(self):
        self.init_rng()

        with h5py.File(self.data_path, "r") as hf:
            iter_range = self.get_iter_range(len(hf))[self.local_rank :: self.n_gpu_per_node]
            if self.train:
                iter_range = self.rng.permutation(iter_range)

            for i in iter_range:
                sample = hf[f"{i:04d}"]
                t0 = self.sample_initial_time(sample["data"].shape[0]) if self.random_start else 0

                data = sample["data"][
                    t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                ]  # (t_num, x_num, x_num, 1)
                data = self.augment_data(data)
                data = torch.from_numpy(data).float()

                d = {}
                # d["type"] = self.type_label

                if not self.params.data.tie_fields:
                    d["data_mask"] = self.c_mask
                    nt, nx, ny, _ = data.size()
                    tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                    tmp[..., self.c_mask_bool] = data
                    data = tmp

                d["data"] = data

                if self.params.use_raw_time:
                    t_grid = sample["grid"]["t"][t0 : (t0 + self.t_num * self.t_step) : self.t_step]  # (t_num, )
                    t_grid = torch.from_numpy(t_grid).float()
                    d["t"] = t_grid

                if self.params.symbol.symbol_input:
                    d["symbol_input"] = self.symbol_ids

                # x_grid = sample["grid"]["x"][::self.x_step]  # (x_num, )
                # y_grid = sample["grid"]["y"][::self.x_step]  # (x_num, )

                yield d


class ShallowWater2D(myIterDp):
    """
    PDEBench 2D shallow_water dataset.
        size:  1000
        t_num: 101            [0, 1] dt=0.01
        x_num: (128, 128)     (-2.5, 2.5)
        data_dim: 1
        bc: neumann

    Dataset structure:
    0000 - 0999
        data: (101, 128, 128, 1)
        grid
            t: (101,)
            x: (128,)
            y: (128,)
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "shallow_water"

        self.t_step = params.data.shallow_water.t_step
        self.x_step = params.data.shallow_water.x_num // params.data.x_num
        self.data_path = params.data.shallow_water.data_path
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def __iter__(self):
        self.init_rng()

        with h5py.File(self.data_path, "r") as hf:
            iter_range = self.get_iter_range(len(hf))[self.local_rank :: self.n_gpu_per_node]
            if self.train:
                iter_range = self.rng.permutation(iter_range)

            for i in iter_range:
                sample = hf[f"{i:04d}"]
                t0 = self.sample_initial_time(sample["data"].shape[0]) if self.random_start else 0

                data = sample["data"][
                    t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                ]  # (t_num, x_num, x_num, 1)

                d = {}
                # d["type"] = self.type_label

                data = self.augment_data(data)
                data = torch.from_numpy(data).float()

                if not self.params.data.tie_fields:
                    d["data_mask"] = self.c_mask
                    nt, nx, ny, _ = data.size()
                    tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                    tmp[..., self.c_mask_bool] = data
                    data = tmp

                d["data"] = data

                if self.params.use_raw_time:
                    t_grid = sample["grid"]["t"][t0 : (t0 + self.t_num * self.t_step) : self.t_step]  # (t_num, )
                    t_grid = torch.from_numpy(t_grid).float()
                    d["t"] = t_grid

                if self.params.symbol.symbol_input:
                    d["symbol_input"] = self.symbol_ids

                # x_grid = sample["grid"]["x"][::self.x_step]  # (x_num, )
                # y_grid = sample["grid"]["y"][::self.x_step]  # (x_num, )

                yield d


class IncomNS2D(myIterDp):
    """
    PDEBench 2D incompressible navier-stokes dataset. (assumes n_gpu <= 4)
        size:  1096
        t_num: 1000              [0, 5) dt=0.005
        x_num: (512, 512)        [0, 1] ?
        data_dim: 2+1
        bc: dirichlet

    Dataset structure:
    274 files (missing idx 49). In each file:
        velocity:  (4, 1000, 512, 512, 2)
        particles: (4, 1000, 512, 512, 1)
        force:     (4, 512, 512, 2)
        t:         (4, 1000)
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "incom_ns"

        self.t_step = params.data.incom_ns.t_step
        self.x_step = params.data.incom_ns.x_num // params.data.x_num

        self.folder = params.data.incom_ns.folder
        self.data_files = sorted(os.listdir(self.folder))
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def __iter__(self):
        self.init_rng()
        iter_range = self.get_iter_range(len(self.data_files))

        if self.train:
            iter_range = self.rng.permutation(iter_range)

        for file_idx in iter_range:
            data_path = os.path.join(self.folder, self.data_files[file_idx])

            with h5py.File(data_path, "r") as hf:
                file_size = len(hf["velocity"])

                for i in range(self.local_rank, file_size, self.n_gpu_per_node):
                    t0 = self.sample_initial_time(hf["velocity"].shape[1]) if self.random_start else 0

                    velocity = hf["velocity"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num, 2)
                    particles = hf["particles"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num, 1)

                    d = {}
                    # d["type"] = self.type_label

                    data = np.concatenate([velocity, particles], axis=-1)  # (t_num, x_num, x_num, 3)
                    data = self.augment_data(data)
                    data = torch.from_numpy(data).float()

                    if not self.params.data.tie_fields:
                        d["data_mask"] = self.c_mask
                        nt, nx, ny, _ = data.size()
                        tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                        tmp[..., self.c_mask_bool] = data
                        data = tmp

                    d["data"] = data

                    if self.params.use_raw_time:
                        t_grid = hf["t"][i, t0 : (t0 + self.t_num * self.t_step) : self.t_step]  # (t_num, )
                        t_grid = torch.from_numpy(t_grid).float()
                        d["t"] = t_grid

                    if self.params.symbol.symbol_input:
                        d["symbol_input"] = self.symbol_ids

                    # force = hf["force"][i, :: self.x_step, :: self.x_step]  # (x_num, x_num, 2)

                    yield d


class ComNS2D(myIterDp):
    """
    PDEBench 2D compressible navier-stokes dataset.
        t_num: 21           [0, 1] dt=0.05
        data_dim: 4
        bc: periodic

        Raw shape: (now all converted to the same space grid (128, 128))

        Random fields - Regular:
            size:  40000
            x_num: (128, 128)

        Random fields - Euler (low shear and bulk viscosity):
            size:  2000
            x_num: (512, 512)

        Turbulence:
            size: 2000
            x_num: (512, 512)

    Raw Dataset structure (now all converted to 128x128):
        Random fields - Regular (4 files). In each file:
            Vx:           (10000, 21, 128, 128)
            Vy:           (10000, 21, 128, 128)
            density:      (10000, 21, 128, 128)
            pressure:     (10000, 21, 128, 128)
            t-coordinate: (22,)                    [0, 1.05]
            x-coordinate: (128,)                   (0, 1)
            y-coordinate: (128,)

        Random fields - Euler (2 files). In each file:
            Vx:           (1000, 21, 512, 512)
            Vy:           (1000, 21, 512, 512)
            density:      (1000, 21, 512, 512)
            pressure:     (1000, 21, 512, 512)
            t-coordinate: (22,)
            x-coordinate: (512,)                   (0, 1)   (average every 4 points gives the previous 128 grid)
            y-coordinate: (512,)

        Turbulence (2 files). In each file::
            Vx:           (1000, 21, 512, 512)
            Vy:           (1000, 21, 512, 512)
            density:      (1000, 21, 512, 512)
            pressure:     (1000, 21, 512, 512)
            t-coordinate: (22,)
            x-coordinate: (512,)                   (0, 1)
            y-coordinate: (512,)
    """

    def __init__(self, params, symbol_env, split="train", train=True, file_idx=-1):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "com_ns"

        self.t_step = params.data.com_ns.t_step
        self.x_step = params.data.com_ns.x_num // params.data.x_num

        if params.data.com_ns.type == "all":
            folders = params.data.com_ns.folders.values()
        else:
            folders = [params.data.com_ns.folders[params.data.com_ns.type]]

        files = []
        for folder in folders:
            files += [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".hdf5")]

        self.data_files = sorted(files)
        if file_idx >= 0:
            self.data_files = self.data_files[file_idx : file_idx + 1]

        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def __iter__(self):
        self.init_rng()

        data_paths = self.data_files
        if self.train:
            data_paths = self.rng.permutation(data_paths)

        for data_path in data_paths:

            with h5py.File(data_path, "r") as hf:
                iter_range = self.get_iter_range(len(hf["Vx"]))[self.local_rank :: self.n_gpu_per_node]
                if self.train:
                    iter_range = self.rng.permutation(iter_range)

                for i in iter_range:
                    t0 = self.sample_initial_time(hf["Vx"].shape[1]) if self.random_start else 0

                    Vx = hf["Vx"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)
                    Vy = hf["Vy"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)
                    density = hf["density"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)
                    pressure = hf["pressure"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)

                    data = np.stack([Vx, Vy, density, pressure], axis=-1)  # (t_num, x_num, x_num, 4)
                    data = self.augment_data(data)
                    data = torch.from_numpy(data).float()

                    d = {}
                    # d["type"] = self.type_label

                    if not self.params.data.tie_fields:
                        d["data_mask"] = self.c_mask
                        nt, nx, ny, _ = data.size()
                        tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                        tmp[..., self.c_mask_bool] = data
                        data = tmp

                    d["data"] = data

                    if self.params.use_raw_time:
                        t_grid = hf["t-coordinate"][t0 : (t0 + self.t_num * self.t_step) : self.t_step]  # (t_num, )
                        t_grid = torch.from_numpy(t_grid).float()
                        d["t"] = t_grid

                    if self.params.symbol.symbol_input:
                        # tree = self.symbol_env.generator.get_tree(self.type_label, {"eta": 0.1, "zeta": 0.1})
                        # tree_encoded = self.symbol_env.equation_encoder.encode(tree)
                        # symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
                        # d["symbol_input"] = symbol_input

                        d["symbol_input"] = self.symbol_ids

                    # x_grid = hf["x-coordinate"][:: self.x_step]  # (x_num, )
                    # y_grid = hf["y-coordinate"][:: self.x_step]  # (x_num, )

                    yield d


class IncomNS2DArena(myIterDp):
    """
    PDEArena 2D incompressible navier-stokes dataset (conditioned).
        size: 2496/608/608       train/val/test
        t_num: 56                [18, 102] dt=1.5
        x_num: (128, 128)        [0, 32]
        data_dim: 2+1
        bc: dirichlet for velocity, neumann for scalar

    Dataset structure:
    train/val/test: 78/19/19 files. In each file:
        train/valid/test:
            vx:    (32, 56, 128, 128)
            vy:    (32, 56, 128, 128)
            u:     (32, 56, 128, 128)
            buo_y: (32,)
            t:     (32, 56)
            x:     (32, 128)
            y:     (32, 128)
            dt:    (32,)
            dx:    (32,)
            dy:    (32,)
    """

    split_to_name = {"train": "train", "val": "valid", "test": "test"}

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "incom_ns_arena"

        self.t_step = params.data.incom_ns_arena.t_step
        self.x_step = params.data.incom_ns_arena.x_num // params.data.x_num

        self.folder = params.data.incom_ns_arena.folder
        self.data_files = sorted(
            [f for f in os.listdir(self.folder) if self.split_to_name[self.split] in f and f.endswith(".h5")]
        )
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        # if self.params.symbol.symbol_input:
        #     tree = self.symbol_env.generator.get_tree(self.type_label)
        #     tree_encoded = self.symbol_env.equation_encoder.encode(tree)
        #     symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
        #     self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def get_iter_range(self, total_len):
        # split number of workers (files already split based on train/val/test)

        start = 0
        end = total_len

        if self.num_workers <= 1:
            # return start, end
            return np.arange(start, end)
        else:
            # subdivide based on number of workers
            return np.arange(start + self.worker_id, end, self.num_workers)

    def __iter__(self):
        self.init_rng()
        iter_range = self.get_iter_range(len(self.data_files))

        if self.train:
            iter_range = self.rng.permutation(iter_range)

        for file_idx in iter_range:
            data_path = os.path.join(self.folder, self.data_files[file_idx])

            with h5py.File(data_path, "r") as f:
                hf = f[self.split_to_name[self.split]]
                file_size = len(hf["vx"])

                file_iter_range = np.arange(self.local_rank, file_size, self.n_gpu_per_node)
                if self.train:
                    file_iter_range = self.rng.permutation(file_iter_range)

                for i in file_iter_range:
                    t0 = self.sample_initial_time(hf["vx"].shape[1]) if self.random_start else 0

                    vx = hf["vx"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)
                    vy = hf["vy"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)
                    u = hf["u"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)

                    data = np.stack([vx, vy, u], axis=-1)  # (t_num, x_num, x_num, 3)
                    data = self.augment_data(data)
                    data = torch.from_numpy(data).float()

                    d = {}
                    # d["type"] = self.type_label

                    if not self.params.data.tie_fields:
                        d["data_mask"] = self.c_mask
                        nt, nx, ny, _ = data.size()
                        tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                        tmp[..., self.c_mask_bool] = data
                        data = tmp

                    d["data"] = data

                    if self.params.use_raw_time:
                        t = hf["t"][i, t0 : (t0 + self.t_num * self.t_step) : self.t_step]  # (t_num, )
                        t = torch.from_numpy(t).float()
                        d["t"] = t

                    if self.params.symbol.symbol_input:
                        buo_y = hf["buo_y"][i]
                        tree = self.symbol_env.generator.get_tree(self.type_label, {"F": buo_y})
                        tree_encoded = self.symbol_env.equation_encoder.encode(tree)
                        symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
                        d["symbol_input"] = symbol_input

                        # d["symbol_input"] = self.symbol_ids

                    # x = hf["x"][i, :: self.x_step]  # (x_num, )
                    # y = hf["y"][i, :: self.x_step]  # (x_num, )

                    yield d


class IncomNS2DArenaU(myIterDp):
    """
    PDEArena 2D incompressible navier-stokes dataset (unconditioned).
        size: 1664/1664/1664     train/val/test
        t_num: 14                [18, 102] dt=6.46
        x_num: (128, 128)        [0, 32]
        data_dim: 2+1
        bc: dirichlet for velocity, neumann for scalar

    Dataset structure:
    train/val/test: 52/52/52 files. In each file:
        train/valid/test:
            vx:    (100, 14, 128, 128)
            vy:    (100, 14, 128, 128)
            u:     (100, 14, 128, 128)
            buo_y: (100,)
            t:     (100, 14)
            x:     (100, 128)
            y:     (100, 128)
            dt:    (100,)
            dx:    (100,)
            dy:    (100,)
    """

    split_to_name = {"train": "train", "val": "valid", "test": "test"}

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "incom_ns_arena_u"

        self.t_step = params.data.incom_ns_arena_u.t_step
        self.x_step = params.data.incom_ns_arena_u.x_num // params.data.x_num

        self.folder = params.data.incom_ns_arena_u.folder
        self.data_files = sorted(
            [f for f in os.listdir(self.folder) if self.split_to_name[self.split] in f and f.endswith(".h5")]
        )
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

    def get_iter_range(self, total_len):
        # split number of workers (files already split based on train/val/test)

        start = 0
        end = total_len

        if self.num_workers <= 1:
            # return start, end
            return np.arange(start, end)
        else:
            # subdivide based on number of workers
            return np.arange(start + self.worker_id, end, self.num_workers)

    def __iter__(self):
        self.init_rng()
        iter_range = self.get_iter_range(len(self.data_files))

        if self.train:
            iter_range = self.rng.permutation(iter_range)

        for file_idx in iter_range:

            data_path = os.path.join(self.folder, self.data_files[file_idx])
            with h5py.File(data_path, "r") as f:
                hf = f[self.split_to_name[self.split]]
                file_size = len(hf["vx"])
                file_iter_range = np.arange(self.local_rank, file_size, self.n_gpu_per_node)
                if self.train:
                    file_iter_range = self.rng.permutation(file_iter_range)

                for i in file_iter_range:
                    t0 = self.sample_initial_time(hf["vx"].shape[1]) if self.random_start else 0

                    vx = hf["vx"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)
                    vy = hf["vy"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)
                    u = hf["u"][
                        i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                    ]  # (t_num, x_num, x_num)

                    data = np.stack([vx, vy, u], axis=-1)  # (t_num, x_num, x_num, 3)
                    data = self.augment_data(data)
                    data = torch.from_numpy(data).float()

                    d = {}
                    # d["type"] = self.type_label

                    if not self.params.data.tie_fields:
                        d["data_mask"] = self.c_mask
                        nt, nx, ny, _ = data.size()
                        tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                        tmp[..., self.c_mask_bool] = data
                        data = tmp

                    d["data"] = data

                    if self.params.use_raw_time:
                        t = hf["t"][i, t0 : (t0 + self.t_num * self.t_step) : self.t_step]  # (t_num, )
                        t = torch.from_numpy(t).float()
                        d["t"] = t

                    if self.params.symbol.symbol_input:
                        # buo_y = hf["buo_y"][i]
                        # tree = self.symbol_env.generator.get_tree(self.type_label, {"F": buo_y})
                        # tree_encoded = self.symbol_env.equation_encoder.encode(tree)
                        # symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
                        # d["symbol_input"] = symbol_input

                        d["symbol_input"] = self.symbol_ids

                    # x = hf["x"][i, :: self.x_step]  # (x_num, )
                    # y = hf["y"][i, :: self.x_step]  # (x_num, )
                    yield d


class CFDBench2D(myIterDp):
    """
    CFDBench 2D incompressible navier-stokes dataset.
        size: 8774/1026/1150       train/val/test
        t_num: 20                ?
        x_num: (128, 128)        ? (interpolated from 64x64)
        data_dim: 2+1            (last dimension is boundary information)
        bc: (non-homogeneous) dirichlet

    Dataset structure:
    ns2d_cdb_train.hdf5:
        data: (8774, 20, 128, 128, 3)

    ns2d_cdb_val.hdf5:
        data: (1026, 20, 128, 128, 3)

    ns2d_cdb_test.hdf5:
        data: (1150, 20, 128, 128, 3)
    """

    def __init__(self, params, symbol_env, split="train", train=True):
        super().__init__(params, symbol_env, split, train)

        # dataset specific initialization

        self.type_label = "cfdbench"

        self.t_step = params.data.cfdbench.t_step
        self.x_step = params.data.cfdbench.x_num // params.data.x_num

        self.data_path = params.data.cfdbench.data_path[split]
        self.fully_shuffled = True  # no need to shuffle since we shuffle in __iter__

        if self.params.symbol.symbol_input:
            tree = self.symbol_env.generator.get_tree(self.type_label)
            tree_encoded = self.symbol_env.equation_encoder.encode(tree)
            symbol_input = self.symbol_env.word_to_idx([tree_encoded], float_input=False)[0]
            self.symbol_ids = symbol_input

        if not self.params.data.tie_fields:
            self.c_mask = torch.Tensor(params.data[self.type_label].c_mask)
            self.c_mask_bool = self.c_mask.bool()

        assert not params.use_raw_time

    def get_iter_range(self, total_len):
        # split number of workers (files already split based on train/val/test)

        start = 0
        end = total_len

        if self.num_workers <= 1:
            # return start, end
            return np.arange(start, end)
        else:
            # subdivide based on number of workers
            return np.arange(start + self.worker_id, end, self.num_workers)

    def __iter__(self):
        self.init_rng()

        with h5py.File(self.data_path, "r") as f:
            iter_range = self.get_iter_range(len(f["data"]))[self.local_rank :: self.n_gpu_per_node]
            if self.train:
                iter_range = self.rng.permutation(iter_range)

            for i in iter_range:
                t0 = self.sample_initial_time(f["data"].shape[1]) if self.random_start else 0

                data = f["data"][
                    i, t0 : (t0 + self.t_num * self.t_step) : self.t_step, :: self.x_step, :: self.x_step
                ]  # (t_num, x_num, x_num, 3)

                if not self.params.data.cfdbench.get("use_boundary", 1):
                    data = data[..., :2]

                data = self.augment_data(data)
                data = torch.from_numpy(data).float()

                d = {}
                # d["type"] = self.type_label

                if not self.params.data.tie_fields:
                    d["data_mask"] = self.c_mask
                    nt, nx, ny, _ = data.size()
                    tmp = torch.zeros(nt, nx, ny, self.params.data.max_output_dimension, dtype=data.dtype)
                    tmp[..., self.c_mask_bool] = data
                    data = tmp

                d["data"] = data

                if self.params.symbol.symbol_input:
                    d["symbol_input"] = self.symbol_ids

                yield d
