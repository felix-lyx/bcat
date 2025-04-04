import os
import numpy as np
from logging import getLogger
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import get_dataset
from utils.misc import sync_tensor, to_cuda, set_worker_sharing_strategy
from utils.metrics import compute_metrics
from utils.plot import plot_2d_pde, plot_2d_pde_formal
from data_utils.collate import custom_collate
from tabulate import tabulate
import wandb


logger = getLogger()

metric_to_header = {
    "_l2_error": "rel l2",
    "_mse": "mse",
    "_rmse": "rmse",
    "_l2_error_first_half": "rel l2 1st_half",
    "_l2_error_second_half": "rel l2 2nd_half",
    "_l2_error_step_1": "rel l2 step 1",
    "_l2_error_step_5": "rel l2 step 5",
    "_l2_error_step_10": "rel l2 step 10",
    "_l2_error_int": "rel l2 interior",
}


def get_boundary_mask(patch_size: int, patch_num: int, boundary_width=1):
    """
    Output:
        mask: Tensor (patch_size*patch_num=x_num, x_num)
    """
    single_mask = torch.ones(patch_size, patch_size)
    single_mask[0:boundary_width] = 0
    single_mask[patch_size - boundary_width :] = 0
    single_mask[:, 0:boundary_width] = 0
    single_mask[:, patch_size - boundary_width :] = 0
    mask = (
        single_mask[None, :, None]
        .expand(patch_num, patch_size, patch_num, patch_size)
        .reshape(patch_num * patch_size, -1)
        .contiguous()
    )
    return mask


class Evaluator(object):

    def __init__(self, trainer, symbol_env):
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.symbol_env = symbol_env
        self.datasets: dict = get_dataset(self.params, symbol_env, split="test" if self.params.eval_only else "val")

        self.collate_fn = custom_collate(
            self.params.data.max_output_dimension,
            self.symbol_env.pad_index,
            self.params.data.tie_fields,
            pad_right=self.params.symbol.pad_right,
        )
        self.dataloaders = {
            k: DataLoader(
                v,
                batch_size=self.params.batch_size_eval,
                num_workers=self.params.num_workers_eval,
                worker_init_fn=set_worker_sharing_strategy,
                pin_memory=False,
                collate_fn=custom_collate(
                    self.params.data.max_output_dimension,
                    symbol_env.pad_index,
                    self.params.data.tie_fields,
                    pad_right=self.params.symbol.pad_right,
                ),
            )
            for k, v in self.datasets.items()
        }
        self.types = self.datasets.keys()

        self.validation_metrics = self.params.validation_metrics_print.split(",")
        self.output_len = self.params.data.t_num - self.params.input_len
        if self.output_len < 10:
            self.validation_metrics.remove("_l2_error_step_10")
        if self.output_len < 5:
            self.validation_metrics.remove("_l2_error_step_5")

        if "_l2_error_int" in self.validation_metrics and "patch_num" in self.params.model:
            patch_num = self.params.model.patch_num
            patch_size = self.params.data.x_num // patch_num
            self.boundary_mask = get_boundary_mask(patch_size, patch_num, boundary_width=1)[None, None, :, :, None]
            if not self.params.cpu:
                self.boundary_mask = self.boundary_mask.cuda()
        else:
            self.validation_metrics.remove("_l2_error_int")
            self.boundary_mask = None

    @torch.inference_mode()
    def evaluate(self):

        params = self.params

        model = self.modules["model"]
        model.eval()

        if params.print_outputs:
            save_folder = os.path.join(params.eval_dump_path, "figures/")
            os.makedirs(save_folder, exist_ok=True)

        if params.log_eval_plots > 0:
            plot_folder = os.path.join(params.eval_dump_path, f"epoch_{self.trainer.epoch}_{self.params.local_rank}")
            os.makedirs(plot_folder, exist_ok=True)

        if params.save_outputs:
            output_folder = os.path.join(params.eval_dump_path, "outputs/")
            os.makedirs(output_folder, exist_ok=True)

        if params.eval_only or self.trainer.epoch == params.max_epoch - 1:
            # plot everything during testing and last epoch
            input_plot_len = output_plot_len = -1
        else:
            input_plot_len = 1
            output_plot_len = min(2, self.output_len)

        all_results = {}
        more_metrics = []

        for type, loader in self.dataloaders.items():
            num_plotted = 0
            eval_size = 0
            results = defaultdict(list)

            if self.params.debug:
                logger.info(f"Evaluating {type}")

            for idx, samples in enumerate(loader):
                bs = len(samples["data"])
                eval_size += bs
                model_input, d = self.trainer.prepare_data(samples, train=False)

                if type in ["incom_ns_arena_u"]:
                    # NOTE: currently hardcoded
                    if "output_times" in model_input:
                        model_input["output_times"] = model_input["output_times"][:, : 14 - params.input_len]
                    if "times" in model_input:
                        model_input["times"] = model_input["times"][:, :14]

                if type in ["cfdbench"] and self.params.model.name.endswith("auto"):
                    # NOTE: currently hardcoded
                    model_input["carry_over_c"] = 2

                if self.params.rollout:
                    mode = "rollout"
                    model_input["n_total_steps"] = self.params.data.t_num
                    model_input["normalizer"] = self.trainer.normalize_data
                else:
                    mode = "generate"

                with torch.amp.autocast(
                    "cpu" if params.cpu else "cuda", enabled=bool(params.amp), dtype=torch.bfloat16
                ):
                    data_output = model(mode, **model_input)  # (bs, output_len, x_num, x_num, data_dim)

                # computing eval metrics

                if self.params.normalize:
                    if self.params.denormalize_for_loss:
                        # denormalize data, loss in original space
                        data_output = data_output * d["std"] + d["mean"]

                        data_output = data_output * d["data_mask"]
                        data_loss = self.data_loss_fn(data_output, d["data_label"], d["data_mask"])
                    else:
                        # loss in normalized space, then denormalize for other metrics
                        data_output = data_output * d["data_mask"]
                        data_loss = self.data_loss_fn(data_output, d["data_label"], d["data_mask"])

                        data_output = data_output * d["std"] + d["mean"]
                        d["data_label"] = d["data_label"] * d["std"] + d["mean"]
                else:
                    # no normalization, directly compute data loss
                    data_output = data_output * d["data_mask"]
                    data_loss = self.data_loss_fn(data_output, d["data_label"], d["data_mask"])

                results["data_loss"].extend(data_loss)

                if self.params.data.tie_fields:
                    dim = params.data[type.split(":")[0]].dim
                    data_output = data_output[..., :dim]
                    d["data_label"] = d["data_label"][..., :dim]
                else:
                    # remove padding based on data_mask
                    valid_mask = d["data_mask"][0, 0, 0, 0].bool()  # (dim, )
                    data_output = data_output[..., valid_mask]
                    d["data_label"] = d["data_label"][..., valid_mask]

                cur_result = compute_metrics(
                    data_output,
                    d["data_label"],
                    mask=self.boundary_mask,
                    metrics=self.validation_metrics,
                    batched=True,
                )

                for k in cur_result.keys():
                    results[k].extend(cur_result[k])

                if params.save_outputs:
                    # (b, t, x, y, c)
                    output = data_output[..., : params.data[type.split(":")[0]].dim].float().numpy(force=True)
                    target = samples["data"][..., : params.data[type.split(":")[0]].dim].numpy(force=True)
                    errors = np.array(cur_result["_l2_error"])
                    np.savez(os.path.join(output_folder, f"{type}.npz"), output=output, target=target, errors=errors)

                if params.print_outputs:
                    # plot all outputs
                    data_output = data_output.float().numpy(force=True)  # (bs, output_len, x_num, x_num, data_dim)
                    if params.data.tie_fields:
                        data_all = samples["data"].numpy(
                            force=True
                        )  # (bs, input_len+output_len, x_num, x_num, data_dim)
                    else:
                        data_all = samples["data"][..., valid_mask.cpu()].numpy(force=True)
                    t = (
                        samples["t"].numpy(force=True)
                        if "t" in samples
                        else self.trainer.t.expand(bs, -1).numpy(force=True)
                    )

                    for i in range(bs):
                        index = idx * params.batch_size_eval + i
                        plot_title = "Type {} | Idx {} | {:.4f}".format(type, index, cur_result["_l2_error"][i])
                        plot_2d_pde_formal(
                            data_output[i],
                            data_all[i],
                            t[i],
                            params.input_len,
                            plot_title,
                            filename=f"{type}_plot_{index}",
                            folder=save_folder,
                            dim=params.data[type.split(":")[0]].dim,
                        )

                if params.log_eval_plots > 0 and num_plotted < params.log_eval_plots:
                    # only plot the first element
                    if isinstance(data_output, np.ndarray):
                        # already converted to numpy
                        output = data_output[0]
                        cur_data = data_all[0]
                        t = t[0]
                    else:
                        output = data_output[0].float().numpy(force=True)
                        if params.data.tie_fields:
                            cur_data = samples["data"][0].numpy(force=True)
                        else:
                            cur_data = samples["data"][0, ..., valid_mask.cpu()].numpy(force=True)
                        t = samples["t"][0].numpy(force=True) if "t" in samples else self.trainer.t[0].numpy(force=True)

                    index = idx * params.batch_size_eval
                    plot_title = "Epoch {} | Type {} | idx {} | {:.4f}".format(
                        self.trainer.epoch, type, index, cur_result["_l2_error"][0]
                    )
                    path = plot_2d_pde(
                        output,
                        cur_data,
                        t,
                        params.input_len,
                        plot_title,
                        filename=f"{type}_plot_{index}",
                        folder=plot_folder,
                        input_plot_len=input_plot_len,
                        output_plot_len=output_plot_len,
                        dim=params.data[type.split(":")[0]].dim,
                    )

                    if params.use_wandb:
                        if self.trainer.epoch <= 4 or (self.trainer.epoch + 1) % 5 == 0:
                            wandb.log(
                                {"val": {"epoch": self.trainer.epoch, f"{type}_plot_{num_plotted}": wandb.Image(path)}}
                            )

                    num_plotted += 1

                if params.eval_size > 0 and eval_size >= params.eval_size:
                    break

            for k, v in results.items():
                arr = np.array(v)
                results[k] = np.sum(arr)
                if k == "_l2_error":
                    more_metrics.append(
                        [
                            type,
                            len(arr),
                            np.mean(arr),
                            np.std(arr),
                            np.min(arr),
                            np.max(arr),
                            np.median(arr),
                        ]
                    )

            results["size"] = eval_size
            all_results[type] = results

        if params.multi_gpu:
            # sync results on all gpus
            sorted_keys = None
            for type, results in all_results.items():

                if sorted_keys is None:
                    sorted_keys = sorted(results.keys())

                stats = torch.Tensor([results[k] for k in sorted_keys])
                stats = sync_tensor(stats)
                results = {k: stats[i].item() for i, k in enumerate(sorted_keys)}
                results["size"] = int(results["size"])
                all_results[type] = results

        # aggregate results and compute averages

        total_size = 0
        results_per_type = {}
        stats = defaultdict(list)
        for type, results in all_results.items():
            res_mean_type = {}
            for k, v in results.items():
                if k == "size":
                    res_mean_type[k] = v
                    total_size += v
                else:
                    res_mean_type[k] = v / results["size"]
                    stats[k].append(res_mean_type[k])
            results_per_type[type] = res_mean_type
        stats = {k: np.nanmean(np.array(v)) for k, v in stats.items()}

        # report metrics per equation type as a table

        headers = ["type", "dim", "size", "data_loss"] + self.validation_metrics
        table = []
        for type, results in results_per_type.items():
            row = [type, self.params.data[type.split(":")[0]].dim]
            for j, k in enumerate(headers[2:]):
                row.append(results[k])
            table.append(row)
        table.append(["AVE_BY_CLASS", "-", "-", stats["data_loss"]] + [stats[k] for k in self.validation_metrics])

        headers = list(map(lambda s: metric_to_header[s] if s in metric_to_header else s, headers))
        logger.info(
            "Evaluation Stats (total size = {})\n{}".format(
                total_size,
                tabulate(
                    table,
                    headers=headers,
                    tablefmt="grid",
                    floatfmt=[".6f"] * 4 + [".5f"] * len(self.validation_metrics),
                ),
            )
        )

        logger.info(
            "Additional Stats for Rel L2 Error:\n{}".format(
                tabulate(
                    more_metrics,
                    headers=["type", "size", "mean", "std", "min", "max", "median"],
                    tablefmt="grid",
                    floatfmt=".5f",
                )
            )
        )

        return stats, results_per_type

    def data_loss_fn(self, data_output, data_label, data_mask):
        # copy of trainer data_loss_fn, by batch

        # prepare weights for loss function
        if self.params.loss_weight == "l2":
            weight = torch.linalg.vector_norm(data_label, dim=(2, 3), keepdim=True)  # (bs, output_len, 1, 1, dim)
        elif self.params.loss_weight == "linfty":
            weight, _ = torch.max(torch.abs(data_label), dim=(2, 3), keepdim=True)  # (bs, output_len, 1, 1, dim)
        else:
            weight = None

        if weight is None:
            # no re-weighting, loss is just regular MSE
            loss = F.mse_loss(data_output, data_label, reduction="none")
            loss = (loss * data_mask).flatten(1).sum(1) / torch.count_nonzero(
                data_mask.expand_as(loss).flatten(1), dim=1
            )

        else:
            # reweight by weight
            eps = 1e-8
            if self.params.square_loss:
                loss = F.mse_loss(data_output, data_label, reduction="none")
                loss = ((loss * data_mask) / (weight**2 + eps)).flatten(1).sum(1)
            else:
                loss = torch.linalg.vector_norm(data_output - data_label, dim=(2, 3), keepdim=True)
                loss = (loss / (weight + eps)).flatten(1).sum(1)
        return loss.tolist()  # (bs, )
