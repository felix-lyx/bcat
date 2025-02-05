from pathlib import Path
from typing import Tuple, List, Dict, Union, Any
from bisect import bisect_right
import random

import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm

from .base import CfdDataset, CfdAutoDataset
from .utils import load_json, normalize_bc, normalize_physics_props


def load_case_data(case_dir: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Load from the file that I have preprocessed, and pad the boundary conditions,
    turn into a numpy array of features.

    The shape of both u and v is (time steps, height, width)
    """
    case_params = load_json(case_dir / "case.json")
    # print(case_params)

    u_file = case_dir / "u.npy"
    v_file = case_dir / "v.npy"
    u = np.load(u_file)
    v = np.load(v_file)
    # The shape of u and v is (time steps, height, width)

    mask = np.ones_like(u)

    features = np.stack([u, v, mask], axis=1)  # (T, 3, h, w)
    return features, case_params


class CavityFlowDataset(CfdDataset):
    """
    Auto-regressive dataset for Lid-driven cavity flow problem.

    ```
        u_top
        --->
    -------------
    |           |
    |           |
    |           |
    -------------
    ```

    There should be a clock-wise vortex in the middle, and maybe some small
    vortices on the corners. Delta time between frames is 0.1s.

    The dataset is generated by FLUENT.

    Each example is:
    - input:
        - input to branch net: condition/setting parameters
        - input to trunk net: x, y, t
    - output:
        - u(x, y, t)
    """

    # The time between two consecutive frames in the data
    data_delta_time = 0.1
    # Predefined order of case parameter, used to construct the input to branch net
    case_params_keys = [
        "vel_top",
        "density",
        "viscosity",
        "height",
        "width",
    ]

    def __init__(
        self,
        case_dirs: List[Path],
        norm_props: bool,
        norm_bc: bool,
        sample_point_by_point: bool = False,
        stable_state_diff: float = 0.001,
    ):
        """
        Args:
        - data_dir:
        - norm_props: whether to normalize physics properties.
        - sample_point_by_point: If True, each example is a feature point
            (x, y, t) and the corresponding output function value u(x, y, t).
            If False, each example is an entire frame.
        - stable_state_diff: The mean relative difference between two consecutive
            frames that indicates the system has reached a stable state.
        """
        self.case_dirs = case_dirs
        self.norm_props = norm_props
        self.norm_bc = norm_bc
        self.sample_point_by_point = sample_point_by_point
        self.stable_state_diff = stable_state_diff

        self.load_data(case_dirs)

    def load_data(self, case_dirs: List[Path]):
        """
        This will set the following attributes:
            self.case_params: List[dict]
            self.features: List[Tensor]  # (N, T, 2, h, w)
            self.case_ids: List[int]  # Each sample's case ID
        where N is the number of cases.
        """
        self.case_params: List[Tensor] = []
        self.num_features = 0
        self.num_frames: List[int] = []
        features: List[Tensor] = []
        case_ids: List[int] = []  # The case ID of each example
        self.all_features: List[np.ndarray] = []

        # Loop each frame in each case, create features labels
        for case_id, case_dir in enumerate(tqdm(case_dirs)):
            # (T, c, h, w), dict
            this_case_features, this_case_params = load_case_data(case_dir)
            if self.norm_props:
                normalize_physics_props(this_case_params)
            if self.norm_bc:
                normalize_bc(this_case_params, "vel_top")

            T, c, h, w = this_case_features.shape
            self.num_features += T * h * w
            params_tensor = torch.tensor(
                [this_case_params[key] for key in self.case_params_keys],
                dtype=torch.float32,
            )
            self.all_features.append(this_case_features)
            self.case_params.append(params_tensor)
            features.append(torch.tensor(this_case_features, dtype=torch.float32))
            case_ids.append(case_id)
            self.num_frames.append(T)

        self.features = features  # N * (T, c, h, w)
        self.case_ids = torch.tensor(case_ids)

        # get the no. frames up until this case (inclusive), used for evaluation.
        self.num_frames_before: List[int] = [sum(self.num_frames[: i + 1]) for i in range(len(self.num_frames))]

    def idx_to_case_id_and_frame_idx(self, idx: int) -> Tuple[int, int]:
        """
        Given an index, return the case ID of the corresponding example.
        Will be using `self.num_frames_before`.

        For instance, if the number of frames in the first three cases are
        [10, 12, 11], then:
        - 0~9 should map to case_id = 0
        - 10~21 should map to case_id = 1
        - 22~32 should map to case_id = 2
        In this case, `num_frames_before` should be [10, 22, 33].
        """
        case_id = bisect_right(self.num_frames_before, idx)
        if case_id == 0:
            frame_idx = idx
        else:
            frame_idx = idx - self.num_frames_before[case_id - 1]
        return case_id, frame_idx

    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Returns (
            output func. query location (input to trunk net),
            output func. value,
            input function (BC, physics properties, etc., input to branch net)
        )
        all tensors.
        """
        if self.sample_point_by_point:
            height = self.features[0].shape[2]
            width = self.features[0].shape[3]
            num_pixels = height * width
            case_id, t = self.idx_to_case_id_and_frame_idx(idx // num_pixels)
            frame_idx = idx % num_pixels
            y = frame_idx // width
            x = frame_idx % width
            case_features = self.features[case_id]  # (T, c, h, w)
            case_params = self.case_params[case_id]
            query_point = torch.tensor([t, x, y]).float()
            # Get the output function value
            # print(case_features.shape, t, y, x)
            label = case_features[t, :, y, x]  # (1, c)
            label = label.squeeze().float()
            print("label.dtype:", label.dtype)
            return case_params, query_point, label

        # During evaluation, we need an entire frame
        # So each example returns (case_params, frame)
        # The number of examples is
        case_id, frame_idx = self.idx_to_case_id_and_frame_idx(idx)
        t = torch.tensor([frame_idx]).float()
        frame = self.features[case_id][frame_idx]  # (T, c, h, w)
        case_params = self.case_params[case_id]
        return case_params, t, frame

    def __len__(self) -> int:
        if self.sample_point_by_point:
            return self.num_features
        else:
            # During evaluation, each example is an entire frame
            # So the number of examples is (# frames) * (# data points per frame)
            num_frames = self.num_frames_before[-1]
            # num_rows = self.features[0].shape[2]
            # num_cols = self.features[0].shape[3]
            return num_frames


class CavityFlowAutoDataset(CfdAutoDataset):
    """
    Auto-regressive dataset for Lid-driven cavity flow problem.

        u_top
        --->
    -------------
    |           |
    |           |
    |           |
    -------------

    There should be a clock-wise vortex in the middle, and maybe some small
    vortices on the corners. Delta time between frames is 0.1s.

    The dataset is generated by FLUENT.

    Each example is (u_{t-1} -> u_{t}), is used for auto-regressive generation.
    """

    data_delta_time = 0.1  # The time between two consecutive frames in the data

    def __init__(
        self,
        case_dirs: List[Path],
        norm_props: bool,
        norm_bc: bool,
        delta_time: float = 0.1,
        stable_state_diff: float = 0.001,
    ):
        """
        Number of cases:
        - Different u_top: 50 cases
        - Different density and viscosity: 84 cases
        - Different geometry: 25 cases

        Args:
            case_dirs: Path to the each case data directory.
            delta_time: Time step size in the data.
            stable_state_diff: The mean difference between two consecutive
                frames that indicates the system has reached a stable state.
        """
        self.case_dirs = case_dirs
        self.norm_props = norm_props
        self.norm_bc = norm_bc
        self.delta_time = delta_time
        self.stable_state_diff = stable_state_diff

        # The difference between input and output in number of frames.
        self.time_step_size = int(self.delta_time / self.data_delta_time)
        self.load_data(case_dirs, self.time_step_size)

    def load_data(self, case_dirs: List[Path], time_step_size: int):
        """
        This will set the following attributes:
            self.case_dirs: List[Path]
            self.case_params: List[dict]
            self.inputs: List[Tensor]  # (2, h, w)
            self.labels: List[Tensor]  # (2, h, w)
            self.case_ids: List[int]  # Each sample's case ID
        """
        self.case_params: List[dict] = []
        all_inputs: List[Tensor] = []
        all_labels: List[Tensor] = []
        all_case_ids: List[int] = []  # The case ID of each feature
        self.all_features: List[np.ndarray] = []  # (# case, # frames, 3, h, w)

        # Loop through each frame in each case, create features labels
        for case_id, case_dir in enumerate(case_dirs):
            case_features, this_case_params = load_case_data(case_dir)  # (T, c, h, w)
            self.all_features.append(case_features)
            inputs = case_features[:-time_step_size, :]  # (T, 3, h, w)
            outputs = case_features[time_step_size:, :]  # (T, 3, h, w)
            assert len(inputs) == len(outputs)

            if self.norm_props:
                normalize_physics_props(this_case_params)
            if self.norm_bc:
                normalize_bc(this_case_params, "vel_top")

            self.case_params.append(this_case_params)
            num_steps = len(outputs)
            # Loop frames, get input-output pairs
            # Stop when converged
            for i in range(num_steps):
                inp = torch.tensor(inputs[i], dtype=torch.float32)  # (3, h, w)
                out = torch.tensor(outputs[i], dtype=torch.float32)  # (3, h, w)

                # Check for convergence
                inp_magn = torch.sqrt(inp[0] ** 2 + inp[1] ** 2)
                out_magn = torch.sqrt(out[0] ** 2 + out[1] ** 2)
                diff = torch.abs(inp_magn - out_magn).mean()
                if diff < self.stable_state_diff:
                    print(f"Converged at {i} out of {num_steps}, {this_case_params}")
                    break
                assert not torch.isnan(inp).any()
                assert not torch.isnan(out).any()
                all_inputs.append(inp)  # (3, h, w)
                all_labels.append(out)  # (3, h, w)
                all_case_ids.append(case_id)
        self.inputs = torch.stack(all_inputs)  # (# cases, 3, h, w)
        self.labels = torch.stack(all_labels)  # (# cases, 3, h, w)
        self.case_ids = np.array(all_case_ids)  # (# cases,)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Return:
            feat: (2, h, w)
            label: (2, h, w)
            case_params: dict, e.g. {"density": 1000, "viscosity": 0.01}
        c=2
        """
        inputs = self.inputs[idx]  # (3, h, w)
        label = self.labels[idx]  # (1, h, w)
        case_id = self.case_ids[idx]
        case_params = self.case_params[case_id]
        case_params = {k: torch.tensor(v, dtype=torch.float32) for k, v in case_params.items()}
        return inputs, label, case_params

    def __len__(self):
        return len(self.inputs)


def get_cavity_datasets(
    data_dir: Path,
    case_name: str,
    norm_props: bool,
    norm_bc: bool,
    seed: int = 0,
) -> Tuple[CavityFlowDataset, CavityFlowDataset, CavityFlowDataset]:
    """
    Returns: (train_data, dev_data, test_data)
    """
    case_dirs = []
    for name in ["prop", "bc", "geo"]:
        if name in case_name:
            case_dir = data_dir / name
            this_case_dirs = sorted(case_dir.glob("case*"), key=lambda x: int(x.name[4:]))
            case_dirs += this_case_dirs
    assert len(case_dirs) > 0
    random.seed(seed)
    random.shuffle(case_dirs)
    # Split into train, dev, test
    num_cases = len(case_dirs)
    num_train = round(num_cases * 0.8)
    num_dev = round(num_cases * 0.1)
    train_case_dirs = case_dirs[:num_train]
    dev_case_dirs = case_dirs[num_train : num_train + num_dev]
    test_case_dirs = case_dirs[num_train + num_dev :]
    train_data = CavityFlowDataset(train_case_dirs, norm_props=norm_props, norm_bc=norm_bc)
    dev_data = CavityFlowDataset(dev_case_dirs, norm_props=norm_props, norm_bc=norm_bc)
    test_data = CavityFlowDataset(test_case_dirs, norm_props=norm_props, norm_bc=norm_bc)
    return train_data, dev_data, test_data


def get_cavity_auto_datasets(
    data_dir: Path,
    case_name: str,
    norm_props: bool,
    norm_bc: bool,
    delta_time: float = 0.1,
    stable_state_diff: float = 0.001,
    seed: int = 0,
):
    print(data_dir, case_name)
    case_dirs = []
    for name in ["prop", "bc", "geo"]:
        if name in case_name:
            case_dir = data_dir / name
            print(f"Getting cases from: {case_dir}")
            this_case_dirs = sorted(case_dir.glob("case*"), key=lambda x: int(x.name[4:]))
            case_dirs += this_case_dirs

    assert case_dirs != []

    random.seed(seed)
    random.shuffle(case_dirs)

    # Split into train, dev, test
    num_cases = len(case_dirs)
    num_train = round(num_cases * 0.8)
    num_dev = round(num_cases * 0.1)
    train_case_dirs = case_dirs[:num_train]
    dev_case_dirs = case_dirs[num_train : num_train + num_dev]
    test_case_dirs = case_dirs[num_train + num_dev :]
    print("==== Number of cases in different splits ====")
    print(f"train: {len(train_case_dirs)}, " f"dev: {len(dev_case_dirs)}, " f"test: {len(test_case_dirs)}")
    print("=============================================")
    kwargs: dict[str, Any] = dict(
        delta_time=delta_time,
        stable_state_diff=stable_state_diff,
        norm_props=norm_props,
        norm_bc=norm_bc,
    )
    train_data = CavityFlowAutoDataset(train_case_dirs, **kwargs)
    dev_data = CavityFlowAutoDataset(dev_case_dirs, **kwargs)
    test_data = CavityFlowAutoDataset(test_case_dirs, **kwargs)
    return train_data, dev_data, test_data


if __name__ == "__main__":
    data_dir = Path("../../data/large/cfdbench/cavity")
    delta_time = 0.5
    # train_data, dev_data, test_data = get_cavity_datasets(
    #     data_dir,
    #     case_name="prop",
    #     norm_props=True,
    #     norm_bc=True,
    # )
    train_data, dev_data, test_data = get_cavity_auto_datasets(
        data_dir, case_name="prop", norm_props=True, norm_bc=True
    )
    print(train_data[0])  # [5], [1], [3,64,64]  bc: 3729, geo: 1273, prop: 4770
