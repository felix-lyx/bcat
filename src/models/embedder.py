import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from neuralop.models import FNO2d

try:
    from .attention_utils import get_embeddings, get_activation
except:
    from attention_utils import get_embeddings, get_activation
from logging import getLogger

logger = getLogger()


def get_embedder(config, x_num, max_output_dim):
    match config.type:
        case "linear":
            embedder = LinearEmbedder
        case "conv":
            embedder = ConvEmbedder
        case "patch":
            embedder = PatchEmbedder
        case _:
            raise ValueError(f"Unknown embedder type: {config.type}")

    return embedder(config, x_num, max_output_dim)


def layer_initialize(layer, mode="zero", gamma=0.01):
    # re-initialize given layer to have small outputs
    if mode == "zero":
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif mode == "uniform":
        nn.init.uniform_(layer.weight, -gamma, gamma)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, -gamma, gamma)
    else:
        raise ValueError(f"Unknown mode {mode}")


class LinearEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim
        act = get_activation("gelu")

        assert (
            x_num % config.patch_num == 0
        ), f"x_num must be divisible by patch_num, x_num: {x_num}, patch_num: {config.patch_num}"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        fold_params = dict(kernel_size=self.patch_resolution, stride=self.patch_resolution)
        self.patchify = nn.Unfold(**fold_params)
        self.unpatchify = nn.Fold(output_size=x_num, **fold_params)

        # position embeddings
        self.patch_position_embeddings = get_embeddings((1, 1, config.patch_num * config.patch_num, self.dim))
        self.time_embeddings = get_embeddings((1, config.get("max_time_len", 20), 1, self.dim))

        self.in_proj = nn.Linear(self.patch_dim, self.dim)
        self.pre_proj = nn.Sequential(
            act(),
            nn.Linear(self.dim, self.dim),
        )

        self.conv_dim = config.get("conv_dim", self.dim // 4)
        self.post_proj = nn.Sequential(
            nn.Linear(self.dim, self.conv_dim),
            act(),
            nn.Linear(self.conv_dim, self.conv_dim),
            act(),
        )
        self.head = nn.Linear(self.conv_dim, self.patch_dim)

    def get_pos_embeddings(self, t_len):
        return (self.time_embeddings[:, :t_len] + self.patch_position_embeddings).view(1, -1, self.dim)  # (1, t*p*p, d)

    def encode(self, data, proj=True):
        """
        (b, t, x_num, x_num, data_dim) -> (b, t*p*p, d)
        """
        b = data.size(0)
        data = rearrange(data, "b t h w c -> (b t) c h w")
        data = self.patchify(data)  # (b*t, d, p*p)
        data = rearrange(data, "(b t) d pp -> b (t pp) d", b=b)

        if proj:
            return self.pre_proj(self.in_proj(data))
        else:
            return data

    def decode(self, data, proj=True):
        """
        (b, t*p*p, d) -> (b, t, x_num, x_num, data_dim)
        """
        if proj:
            data = self.head(self.post_proj(data))  # (b, t*p*p, d)

        b = data.size(0)
        data = rearrange(data, "b (t pp) d -> (b t) d pp", pp=self.config.patch_num**2)
        data = self.unpatchify(data)  # (b*t, data_dim, x_num, x_num)
        data = rearrange(data, "(b t) c h w -> b t h w c", b=b)

        return data


class ConvEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim
        act = get_activation("gelu")

        assert (
            x_num % config.patch_num == 0
        ), f"x_num must be divisible by patch_num, x_num: {x_num}, patch_num: {config.patch_num}"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        assert (
            x_num % config.patch_num_output == 0
        ), f"x_num must be divisible by patch_num_output, x_num: {x_num}, patch_num_output: {config.patch_num_output}"
        self.patch_resolution_output = (
            x_num // config.patch_num_output
        )  # resolution of one space dimension for each patch in output
        self.patch_dim_output = (
            data_dim * self.patch_resolution_output * self.patch_resolution_output
        )  # dimension per patch in output

        ## for encoder part

        self.patch_position_embeddings = get_embeddings((1, 1, config.patch_num * config.patch_num, self.dim))

        self.time_embed_type = config.get("time_embed", "continuous")
        match self.time_embed_type:
            case "continuous":
                self.time_proj = nn.Sequential(
                    nn.Linear(1, self.dim),
                    act(),
                    nn.Linear(self.dim, self.dim),
                )
            case "learnable":
                self.time_embeddings = get_embeddings((1, config.get("max_time_len", 10), 1, self.dim))

        if config.get("early_conv", 0):
            n_conv_layers = math.log2(self.patch_resolution)
            assert n_conv_layers.is_integer(), f"patch_resolution {self.patch_resolution} must be a power of 2"
            n_conv_layers = int(n_conv_layers)
            kernel_size = [3] * n_conv_layers + [1]
            stride = [2] * n_conv_layers + [1]
            padding = [1] * n_conv_layers + [0]
            channels = [data_dim] + [self.dim // (2**i) for i in range(n_conv_layers - 1, 0, -1)] + [self.dim, self.dim]

            self.conv_proj = nn.Sequential()
            for i in range(len(kernel_size)):
                self.conv_proj.append(
                    nn.Conv2d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_size[i],
                        stride=stride[i],
                        padding=padding[i],
                    )
                )
                if i < len(kernel_size) - 1:
                    self.conv_proj.append(act())
        else:
            # regular vit patch embedding
            self.in_proj = nn.Conv2d(
                in_channels=data_dim,
                out_channels=self.dim,
                kernel_size=self.patch_resolution,
                stride=self.patch_resolution,
            )
            self.conv_proj = nn.Sequential(
                act(),
                nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1),
            )

        ## for decoder part

        self.conv_dim = config.get("conv_dim", self.dim // 4)

        if config.get("deep", 0):
            self.post_proj = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=self.dim),
                act(),
                nn.Linear(in_features=self.dim, out_features=self.dim),
                act(),
                Rearrange("b (t h w) d -> (b t) d h w", h=self.config.patch_num_output, w=self.config.patch_num_output),
                nn.ConvTranspose2d(
                    in_channels=self.dim,
                    out_channels=self.conv_dim,
                    kernel_size=self.patch_resolution_output,
                    stride=self.patch_resolution_output,
                ),
                act(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                act(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                act(),
            )
        else:
            self.post_proj = nn.Sequential(
                Rearrange("b (t h w) d -> (b t) d h w", h=self.config.patch_num_output, w=self.config.patch_num_output),
                nn.ConvTranspose2d(
                    in_channels=self.dim,
                    out_channels=self.conv_dim,
                    kernel_size=self.patch_resolution_output,
                    stride=self.patch_resolution_output,
                ),
                act(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                act(),
            )
        self.head = nn.Conv2d(in_channels=self.conv_dim, out_channels=self.data_dim, kernel_size=1, stride=1)

        if config.get("initialize_small_output", 0):
            layer_initialize(self.head, mode=config.initialize_small_output)

    def encode(self, data, times, skip_len=0):
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs, input_len, 1)
        Output:
            data:           Tensor (bs, data_len, dim)      data_len = input_len * patch_num * patch_num
                            embedded data + time embeddings + patch position embeddings
        """

        bs = data.size(0)
        data = rearrange(data[:, skip_len:], "b t h w c -> (b t) c h w")
        data = self.in_proj(data)
        data = self.conv_proj(data)  # (bs*input_len, d, patch_num, patch_num)
        data = rearrange(data, "(b t) d h w -> b t (h w) d", b=bs)  # (bs, input_len, p*p, dim)

        match self.time_embed_type:
            case "continuous":
                time_embeddings = self.time_proj(times[:, skip_len:])[:, :, None]  # (bs, input_len, 1, dim)
                data = data + time_embeddings
            case "learnable":
                time_embeddings = self.time_embeddings[:, skip_len : times.size(1)]  # (bs, input_len, 1, dim)
                data = data + time_embeddings

        data = data + self.patch_position_embeddings  # (b, input_len, p*p, d)

        data = data.reshape(bs, -1, self.dim)
        return data

    def decode(self, data_output):
        """
        Input:
            data_output:     Tensor (bs, query_len, dim)
                             query_len = output_len * patch_num * patch_num
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)
        data_output = self.post_proj(data_output)  # (bs*output_len, data_dim, x_num, x_num)
        data_output = self.head(data_output)
        data_output = rearrange(data_output, "(b t) c h w -> b t h w c", b=bs)
        return data_output


class PatchEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim
        act = get_activation("gelu")

        assert (
            x_num % config.patch_num == 0
        ), f"x_num must be divisible by patch_num, x_num: {x_num}, patch_num: {config.patch_num}"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        assert (
            x_num % config.patch_num_output == 0
        ), f"x_num must be divisible by patch_num_output, x_num: {x_num}, patch_num_output: {config.patch_num_output}"
        self.patch_resolution_output = (
            x_num // config.patch_num_output
        )  # resolution of one space dimension for each patch in output
        self.patch_dim_output = (
            data_dim * self.patch_resolution_output * self.patch_resolution_output
        )  # dimension per patch in output

        ## for encoder part

        self.patch_position_embeddings = get_embeddings((1, 1, config.patch_num, config.patch_num, self.dim))

        self.time_embed_type = config.get("time_embed", "continuous")
        match self.time_embed_type:
            case "continuous":
                self.time_proj = nn.Sequential(
                    nn.Linear(1, self.dim),
                    act(),
                    nn.Linear(self.dim, self.dim),
                )
            case "learnable":
                self.time_embeddings = get_embeddings((1, config.get("max_time_len", 20), 1, 1, self.dim))

        # regular vit patch embedding
        self.in_proj = nn.Conv2d(
            in_channels=data_dim,
            out_channels=self.dim,
            kernel_size=self.patch_resolution,
            stride=self.patch_resolution,
        )
        self.conv_proj = nn.Sequential(
            act(),
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1),
        )

        ## for decoder part

        self.conv_dim = config.get("conv_dim", self.dim // 4)

        self.post_proj = nn.Sequential(
            # Rearrange("b (t h w) d -> (b t) d h w", h=self.config.patch_num_output, w=self.config.patch_num_output),
            nn.ConvTranspose2d(
                in_channels=self.dim,
                out_channels=self.conv_dim,
                kernel_size=self.patch_resolution_output,
                stride=self.patch_resolution_output,
            ),
            act(),
            nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
            act(),
        )
        self.head = nn.Conv2d(in_channels=self.conv_dim, out_channels=self.data_dim, kernel_size=1, stride=1)

    def encode(self, data, times, mode="none"):
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs, input_len, 1)
        Output:
            data:   embedded data + time embeddings + patch position embeddings
                mode:   flatten -> Tensor (bs, input_len*patch_num*patch_num, dim)
                        st      -> Tensor (bs, input_len, patch_num*patch_num, dim)
                        none    -> Tensor (bs, input_len, patch_num, patch_num, dim)
        """

        bs = data.size(0)
        data = rearrange(data, "b t h w c -> (b t) c h w")
        data = self.in_proj(data)
        data = self.conv_proj(data)  # (bs*input_len, d, patch_num, patch_num)
        data = rearrange(data, "(b t) d h w -> b t h w d", b=bs)  # (bs, input_len, p, p, dim)

        match self.time_embed_type:
            case "continuous":
                time_embeddings = self.time_proj(times)[:, :, None, None]  # (bs, input_len, 1, 1, dim)
                data = data + time_embeddings
            case "learnable":
                time_embeddings = self.time_embeddings[:, : times.size(1)]  # (bs, input_len, 1, 1, dim)
                data = data + time_embeddings

        data = data + self.patch_position_embeddings  # (b, input_len, p*p, d)

        match mode:
            case "flatten":
                return data.reshape(bs, -1, self.dim)
            case "st":
                # space time
                return rearrange(data, "b t h w c -> b t (h w) c")
            case _:
                return data

    def decode(self, data_output, mode="none"):
        """
        Input:
            data_output:
                mode:   flatten -> Tensor (bs, output_len*patch_num*patch_num, dim)
                        st      -> Tensor (bs, output_len, patch_num*patch_num, dim)
                        none    -> Tensor (bs, output_len, patch_num, patch_num, dim)
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)

        match mode:
            case "flatten":
                data_output = rearrange(
                    data_output,
                    "b (t h w) d -> (b t) d h w",
                    h=self.config.patch_num_output,
                    w=self.config.patch_num_output,
                )
            case "st":
                data_output = rearrange(
                    data_output,
                    "b t (h w) d -> (b t) d h w",
                    h=self.config.patch_num_output,
                    w=self.config.patch_num_output,
                )
            case _:
                data_output = rearrange(data_output, "b t h w d -> (b t) d h w")

        data_output = self.post_proj(data_output)  # (bs*output_len, data_dim, x_num, x_num)
        data_output = self.head(data_output)
        data_output = rearrange(data_output, "(b t) c h w -> b t h w c", b=bs)
        return data_output
