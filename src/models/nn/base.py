import math
from typing import Sequence

import torch

from zuko.flows import NSF as _NSF
from zuko.nn import LayerNorm

from src.utils import parse_torch_nn_class
from src.models.nn.icnn.convex_init import (
    TraditionalInitialiser,
    ConvexInitialiser,
    ConvexBiasCorrectionInitialiser,
)
from src.models.nn.icnn.convex_modules import *


class NSF(_NSF):
    def __init__(self, shape: Sequence[int], n_stacked=None, **kwargs):
        assert (
            len(shape) == 1
        ), "Neural Spline Flow (NSF) requires flat parameters!"
        super().__init__(
            features=shape[0],
            context=0,
            activation=parse_torch_nn_class(
                kwargs.pop("activation", None), torch.nn.ReLU
            ),
            **kwargs,
        )


class MLP(torch.nn.Module):
    # Based on zuko.nn.MLP

    def __init__(
        self,
        input_dim,
        out_dim,
        hidden_features=(64, 64),
        activation=None,
        normalize: bool = False,
        dropout: float = 0.0,
        inplace: bool = True,
        spectral_norm: bool = False,
        spectral_norm_config: dict = None,
        **kwargs,
    ):
        if spectral_norm_config is None:
            spectral_norm_config = dict()
        super().__init__()
        activation = parse_torch_nn_class(activation, torch.nn.ReLU)

        normalization = LayerNorm if normalize else lambda: None
        dropout_layer = lambda: (
            torch.nn.Dropout(dropout, inplace=inplace) if dropout > 0 else None
        )

        layers = []

        for before, after in zip(
            (input_dim, *hidden_features),
            (*hidden_features, out_dim),
        ):
            layers.extend(
                [
                    dropout_layer(),
                    (
                        torch.nn.Linear(before, after, **kwargs)
                        if not spectral_norm
                        else torch.nn.utils.parametrizations.spectral_norm(
                            torch.nn.Linear(before, after, **kwargs),
                            **spectral_norm_config,
                        )
                    ),
                    activation(inplace=inplace),
                    normalization(),
                ]
            )

        layers = layers[1:-2]
        layers = filter(lambda l: l is not None, layers)

        self.layers = torch.nn.Sequential(*layers)

        self.in_features = input_dim
        self.out_features = out_dim
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)


class EBMMLP(MLP):
    def __init__(
        self,
        shape: Sequence[int],
        n_stacked=None,
        **kwargs,
    ):
        assert len(shape) == 1, "MLP requires flat parameter input!"
        super().__init__(input_dim=shape[0], out_dim=1, **kwargs)

    def forward(self, x):
        return super().forward(x).squeeze(-1)


class DistanceToPoint(torch.nn.Module):
    def __init__(
        self, shape: Sequence[int], point: Sequence[float] = None, norm=2
    ):
        assert (
            len(shape) == 1
        ), "DistanceToPoint requires flat parameter input!"
        super().__init__()
        if point is not None:
            assert (
                len(point) == shape[0]
            ), "`point` has to be the same length as input shape!"
            self.register_buffer("point", torch.tensor(point))
        else:
            self.register_buffer("point", torch.zeros(shape[0]))
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.linalg.vector_norm(
            x - self.point.unsqueeze(0), ord=self.norm, dim=-1
        ).squeeze(-1)


class ICNNMLP(torch.nn.Module):
    """
    Source: https://github.com/ml-jku/convex-init"

    Create Input-Convex neural network based on MLP.

    Parameters
    ----------
    shape : torch.Size
        The array shape of a single input image.
    num_hidden : int, optional
        The number of hidden layers in the network.
    positivity : str, optional
        The function to use to make weighs positive (for input-convex nets):
         - ``"exp"`` uses the exponential function to obtain positive weights
         - ``"clip"`` clips values at zero to obtain positive weights
         - ``"icnn"`` clips values at zero after each update
         - ``""`` or ``None`` results in a NON-convex network
    better_init : bool, optional
        Use principled initialisation for convex layers instead of default (He et al., 2015).
    rand_bias : bool, optional
        Use random bias initialisation instead of constants for convex nets.
    corr : float, optional
        The correlation fixed point to aim for in the better initialisation.
    skip : bool, optional
        Wrap layer in skip-connection.
    bias_init_only: bool, optional
        Only apply principled initialisation for bias parameters.
        Weight parameters are initialised using the default (He et al., 2015) initialisation.
    """

    def __init__(
        self,
        shape: Sequence[int],
        num_hidden: int = 3,
        hidden_size: int = -1,
        positivity: str = None,
        better_init: bool = True,
        rand_bias: bool = False,
        corr: float = 0.5,
        skip: bool = False,
        bias_init_only: bool = False,
        activation=None,
    ):
        super().__init__()
        assert len(shape) == 1, "ICNN requires flat parameter input!"

        width = shape[0]
        widths = (
            width if hidden_size == -1 else hidden_size,
        ) * num_hidden + (1,)

        if positivity is None or positivity == "":
            positivity = NoPositivity()
        elif positivity == "exp":
            positivity = ExponentialPositivity()
        elif positivity == "negexp":
            positivity = NegExpPositivity()
        elif positivity == "clip":
            positivity = ClippedPositivity()
        elif positivity == "icnn":
            positivity = LazyClippedPositivity()
        elif positivity is not None:
            raise ValueError(f"unknown value for positivity: '{positivity}'")

        # first layer can be regular
        layer1 = torch.nn.Linear(width, widths[0])
        activation = parse_torch_nn_class(activation, torch.nn.ReLU)
        layers = [
            layer1,
            *(
                torch.nn.Sequential(
                    activation(),
                    ConvexLinear(n_in, n_out, positivity=positivity),
                )
                for n_in, n_out in zip(widths[:-1], widths[1:])
            ),
        ]

        # initialisation
        lecun_init = TraditionalInitialiser(gain=1.0)
        if better_init and not isinstance(positivity, NoPositivity):
            if bias_init_only:
                init = ConvexBiasCorrectionInitialiser(positivity, gain=2.0)
            else:
                init = ConvexInitialiser(
                    var=1.0,
                    corr=corr,
                    bias_noise=0.5 if rand_bias else 0.0,
                )
        else:
            init = TraditionalInitialiser(gain=2.0)

        lecun_init(layer1.weight, layer1.bias)
        for _, convex_layer in layers[1:]:
            init(convex_layer.weight, convex_layer.bias)

        if skip:
            skipped = LinearSkip(
                width, widths[1], torch.nn.Sequential(*layers[:2])
            )
            for layer, num_out in zip(layers[2:], widths[2:]):
                skipped = LinearSkip(
                    width, num_out, torch.nn.Sequential(skipped, layer)
                )
            layers = [skipped]

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.net(x).squeeze(-1)


class Noise(torch.nn.Module):
    def __init__(
        self,
        distribution,
        shape,
    ):
        super().__init__()
        self.distribution = getattr(torch.distributions, distribution["type"])
        self.distribution_params = torch.nn.ParameterDict(
            {
                n: torch.nn.Parameter(
                    torch.tensor(v),
                    requires_grad=False,
                )
                for n, v in distribution["parameters"].items()
            }
        )
        self.shape = shape

    def forward(self, shape: Sequence[int]):
        return self.distribution(**self.distribution_params).sample(
            (*shape, *self.shape)
        )


class GenerativeMLP(torch.nn.Module):
    def __init__(
        self,
        shape: Sequence[int],
        latent_dim: int,
        hidden_features,
        noise=None,
        activation="ReLU",
        output_activation=None,
        n_stacked=None,
        **kwargs,
    ):
        super(GenerativeMLP, self).__init__()
        self.net = MLP(
            input_dim=latent_dim,
            out_dim=math.prod(shape),
            hidden_features=hidden_features,
            activation=activation,
            **kwargs,
        )
        if noise is not None:
            noise = {
                "type": "Normal",
                "parameters": {"loc": 0.0, "scale": 1.0},
            }
        self.noise = Noise(shape=(latent_dim,), distribution=noise)
        output_activation = parse_torch_nn_class(output_activation, None)
        if output_activation is not None:
            self.net.layers.append(output_activation())
        self.shape = shape

    def forward(self, shape=None, noise=None):
        if noise is None:
            return self.net(self.noise(shape)).view(*shape, *self.shape)
        else:
            return self.net(noise).view(
                *noise.shape[: -len(self.noise.shape)], *self.shape
            )

    def sample(self, shape):
        return self(shape)

    def rsample(self, shape):
        return self(shape)


class View(torch.nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)


class MNISTCritic(torch.nn.Module):
    def __init__(self, shape, hidden_dim=64, activation=None, n_stacked=1):
        super(MNISTCritic, self).__init__()
        activation = parse_torch_nn_class(activation, torch.nn.ReLU)
        layers = [View((n_stacked, *shape))]
        layers.append(
            torch.nn.Conv2d(
                n_stacked,
                hidden_dim,
                kernel_size=5,
                stride=2,
                padding=2,
            )
        )
        layers.append(activation())
        layers.append(
            torch.nn.Conv2d(
                hidden_dim,
                2 * hidden_dim,
                kernel_size=5,
                stride=2,
                padding=2,
            )
        )
        layers.append(activation())
        layers.append(
            torch.nn.Conv2d(
                2 * hidden_dim,
                4 * hidden_dim,
                kernel_size=5,
                stride=2,
                padding=2,
            )
        )
        layers.append(activation())
        layers.append(View((4 * 4 * 4 * hidden_dim,)))
        layers.append(torch.nn.Linear(4 * 4 * 4 * hidden_dim, 1))
        self.net = torch.nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.net(x).squeeze(dim=(-1, -2))


class MNISTGenerator(torch.nn.Module):
    def __init__(
        self,
        shape,
        hidden_dim,
        latent_dim,
        noise=None,
        activation=None,
        n_stacked=1,
    ):
        super(MNISTGenerator, self).__init__()
        if n_stacked != 1:
            self.shape = (n_stacked, *shape)
        else:
            self.shape = shape
        activation = parse_torch_nn_class(activation, torch.nn.ReLU)
        if noise is not None:
            noise = {
                "type": "Normal",
                "parameters": {"loc": 0.0, "scale": 1.0},
            }
        self.noise = Noise(shape=(latent_dim,), distribution=noise)
        layers = [
            torch.nn.Linear(latent_dim, 4 * 4 * 4 * hidden_dim),
            activation(),
            View((4 * hidden_dim, 4, 4)),
            torch.nn.ConvTranspose2d(
                4 * hidden_dim,
                2 * hidden_dim,
                kernel_size=5,
                stride=2,
                dilation=1,
                padding=2,
                output_padding=1,
            ),
            activation(),
        ]
        self.net1 = torch.nn.Sequential(*layers)
        layers = [
            torch.nn.ConvTranspose2d(
                2 * hidden_dim,
                hidden_dim,
                kernel_size=5,
                stride=2,
                dilation=1,
                padding=2,
                output_padding=1,
            ),
            activation(),
            torch.nn.ConvTranspose2d(
                hidden_dim,
                n_stacked,
                kernel_size=5,
                stride=2,
                dilation=1,
                padding=2,
                output_padding=1,
            ),
            torch.nn.Sigmoid(),
            View(self.shape),
        ]
        self.net2 = torch.nn.Sequential(*layers)
        self.apply(self._init_weights)

    def net(self, x):
        return self.net2(self.net1(x)[..., :7, :7])

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.ConvTranspose2d):
            torch.nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, shape=None, noise=None):
        if noise is None:
            return self.net(self.noise(shape)).view(*shape, *self.shape)
        else:
            return self.net(noise).view(
                *noise.shape[: -len(self.noise.shape)], *self.shape
            )

    def sample(self, shape):
        return self(shape)

    def rsample(self, shape):
        return self(shape)


class CIFAR10CriticSimple(torch.nn.Module):
    def __init__(
        self,
        shape,
        hidden_dim=64,
        activation=None,
        batch_norm=False,
        n_stacked=None,
    ):
        super(CIFAR10CriticSimple, self).__init__()
        activation = parse_torch_nn_class(activation, torch.nn.ReLU)
        layers = [
            torch.nn.Conv2d(
                3,
                hidden_dim,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            activation(),
            torch.nn.Conv2d(
                hidden_dim,
                2 * hidden_dim,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            (
                torch.nn.BatchNorm2d(2 * hidden_dim, track_running_stats=False)
                if batch_norm
                else None
            ),
            activation(),
            torch.nn.Conv2d(
                2 * hidden_dim,
                4 * hidden_dim,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            (
                torch.nn.BatchNorm2d(4 * hidden_dim, track_running_stats=False)
                if batch_norm
                else None
            ),
            activation(),
            View((4 * 4 * 4 * hidden_dim,)),
            torch.nn.Linear(4 * 4 * 4 * hidden_dim, 1),
        ]
        self.net = torch.nn.Sequential(
            *filter(lambda l: l is not None, layers)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.net(x).squeeze(dim=(-1, -2))


class CIFAR10GeneratorSimple(torch.nn.Module):
    def __init__(
        self,
        shape,
        hidden_dim,
        latent_dim,
        noise=None,
        activation=None,
        n_stacked=None,
    ):
        super(CIFAR10GeneratorSimple, self).__init__()
        self.shape = shape
        activation = parse_torch_nn_class(activation, torch.nn.ReLU)
        if noise is not None:
            noise = {
                "type": "Normal",
                "parameters": {"loc": 0.0, "scale": 1.0},
            }
        self.noise = Noise(shape=(latent_dim,), distribution=noise)
        layers = [
            torch.nn.Linear(latent_dim, 4 * 4 * 4 * hidden_dim),
            torch.nn.BatchNorm1d(
                4 * 4 * 4 * hidden_dim, track_running_stats=False
            ),
            activation(),
            View((4 * hidden_dim, 4, 4)),
            torch.nn.ConvTranspose2d(
                4 * hidden_dim,
                2 * hidden_dim,
                kernel_size=5,
                stride=2,
                dilation=1,
                padding=2,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(2 * hidden_dim, track_running_stats=False),
            activation(),
            torch.nn.ConvTranspose2d(
                2 * hidden_dim,
                hidden_dim,
                kernel_size=5,
                stride=2,
                dilation=1,
                padding=2,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(hidden_dim, track_running_stats=False),
            activation(),
            torch.nn.ConvTranspose2d(
                hidden_dim,
                3,
                kernel_size=5,
                stride=2,
                dilation=1,
                padding=2,
                output_padding=1,
            ),
            torch.nn.Tanh(),
            View(self.shape),
        ]
        self.net = torch.nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.ConvTranspose2d):
            torch.nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, shape=None, noise=None):
        if noise is None:
            return self.net(self.noise(shape)).view(*shape, *self.shape)
        else:
            return self.net(noise).view(
                *noise.shape[: -len(self.noise.shape)], *self.shape
            )

    def sample(self, shape):
        return self(shape)

    def rsample(self, shape):
        return self(shape)


### BEGINNING: copied from https://github.com/jalola/improved-wgan-pytorch ###


class MyConvo2d(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        he_init=True,
        stride=1,
        bias=True,
    ):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1) / 2)
        self.conv = torch.nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride=1,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, input):
        output = self.conv(input)
        return output


class ConvMeanPool(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(
            input_dim, output_dim, kernel_size, he_init=self.he_init
        )

    def forward(self, input):
        output = self.conv(input)
        output = (
            output[..., ::2, ::2]
            + output[..., 1::2, ::2]
            + output[..., ::2, 1::2]
            + output[..., 1::2, 1::2]
        ) / 4
        return output


class MeanPoolConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(
            input_dim, output_dim, kernel_size, he_init=self.he_init
        )

    def forward(self, input):
        output = input
        output = (
            output[..., ::2, ::2]
            + output[..., 1::2, ::2]
            + output[..., ::2, 1::2]
            + output[..., 1::2, 1::2]
        ) / 4
        output = self.conv(output)
        return output


class DepthToSpace(torch.nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(
            batch_size,
            input_height,
            input_width,
            self.block_size_sq,
            output_depth,
        )
        spl = t_1.split(self.block_size, 3)
        stacks = [
            t_t.reshape(batch_size, input_height, output_width, output_depth)
            for t_t in spl
        ]
        output = (
            torch.stack(stacks, 0)
            .transpose(0, 1)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch_size, output_height, output_width, output_depth)
        )
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, kernel_size, he_init=True, bias=True
    ):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(
            input_dim, output_dim, kernel_size, he_init=self.he_init, bias=bias
        )
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        resample=None,
        hw=64,
        track_running_stats=False,
    ):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        if resample == "down":
            self.bn1 = torch.nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = torch.nn.LayerNorm([input_dim, hw, hw])
        elif resample == "up":
            self.bn1 = torch.nn.BatchNorm2d(
                input_dim, track_running_stats=track_running_stats
            )
            self.bn2 = torch.nn.BatchNorm2d(
                output_dim, track_running_stats=track_running_stats
            )
        elif resample == None:
            # TODO: ????
            self.bn1 = torch.nn.BatchNorm2d(
                output_dim, track_running_stats=track_running_stats
            )
            self.bn2 = torch.nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception("invalid resample value")

        if resample == "down":
            self.conv_shortcut = MeanPoolConv(
                input_dim, output_dim, kernel_size=1, he_init=False
            )
            self.conv_1 = MyConvo2d(
                input_dim, input_dim, kernel_size=kernel_size, bias=False
            )
            self.conv_2 = ConvMeanPool(
                input_dim, output_dim, kernel_size=kernel_size
            )
        elif resample == "up":
            self.conv_shortcut = UpSampleConv(
                input_dim, output_dim, kernel_size=1, he_init=False
            )
            self.conv_1 = UpSampleConv(
                input_dim, output_dim, kernel_size=kernel_size, bias=False
            )
            self.conv_2 = MyConvo2d(
                output_dim, output_dim, kernel_size=kernel_size
            )
        elif resample == None:
            self.conv_shortcut = MyConvo2d(
                input_dim, output_dim, kernel_size=1, he_init=False
            )
            self.conv_1 = MyConvo2d(
                input_dim, input_dim, kernel_size=kernel_size, bias=False
            )
            self.conv_2 = MyConvo2d(
                input_dim, output_dim, kernel_size=kernel_size
            )
        else:
            raise Exception("invalid resample value")

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output


class CIFAR10GeneratorResNet(torch.nn.Module):
    def __init__(
        self,
        shape,
        latent_dim,
        noise=None,
        activation=None,
        n_stacked=None,
    ):
        super(CIFAR10GeneratorResNet, self).__init__()
        self.shape = shape
        activation = parse_torch_nn_class(activation, torch.nn.ReLU)
        if noise is not None:
            noise = {
                "type": "Normal",
                "parameters": {"loc": 0.0, "scale": 1.0},
            }
        self.noise = Noise(shape=(latent_dim,), distribution=noise)

        self.dim = shape[-1]

        self.ssize = self.dim // 16
        self.ln1 = torch.nn.Linear(
            latent_dim, self.ssize * self.ssize * 8 * self.dim
        )
        self.rb1 = ResidualBlock(
            8 * self.dim,
            8 * self.dim,
            3,
            resample="up",
            track_running_stats=True,
        )
        self.rb2 = ResidualBlock(
            8 * self.dim,
            4 * self.dim,
            3,
            resample="up",
            track_running_stats=True,
        )
        self.rb3 = ResidualBlock(
            4 * self.dim,
            2 * self.dim,
            3,
            resample="up",
            track_running_stats=True,
        )
        self.rb4 = ResidualBlock(
            2 * self.dim,
            1 * self.dim,
            3,
            resample="up",
            track_running_stats=True,
        )
        self.bn = torch.nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1 * self.dim, 3, 3)
        self.relu = activation()
        self.tanh = torch.nn.Tanh()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, MyConvo2d):
            if module.conv.weight is not None:
                if module.he_init:
                    torch.nn.init.kaiming_uniform_(module.conv.weight)
                else:
                    torch.nn.init.xavier_uniform_(module.conv.weight)
            if module.conv.bias is not None:
                torch.nn.init.constant_(module.conv.bias, 0.0)
        if isinstance(module, torch.nn.Linear):
            if module.weight is not None:
                torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def net(self, input):
        output = self.ln1(input)
        output = output.view(-1, 8 * self.dim, self.ssize, self.ssize)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        return output

    def forward(self, shape=None, noise=None):
        if noise is None:
            return self.net(self.noise(shape)).view(*shape, *self.shape)
        else:
            return self.net(noise).view(
                *noise.shape[: -len(self.noise.shape)], *self.shape
            )

    def sample(self, shape):
        return self(shape)

    def rsample(self, shape):
        return self(shape)


class CIFAR10CriticResNet(torch.nn.Module):
    def __init__(
        self,
        shape,
        n_stacked=None,
    ):
        super(CIFAR10CriticResNet, self).__init__()

        self.dim = shape[-1]

        self.ssize = self.dim // 16
        self.conv1 = MyConvo2d(3, self.dim, 3, he_init=False)
        self.rb1 = ResidualBlock(
            self.dim, 2 * self.dim, 3, resample="down", hw=self.dim
        )
        self.rb2 = ResidualBlock(
            2 * self.dim,
            4 * self.dim,
            3,
            resample="down",
            hw=int(self.dim / 2),
        )
        self.rb3 = ResidualBlock(
            4 * self.dim,
            8 * self.dim,
            3,
            resample="down",
            hw=int(self.dim / 4),
        )
        self.rb4 = ResidualBlock(
            8 * self.dim,
            8 * self.dim,
            3,
            resample="down",
            hw=int(self.dim / 8),
        )
        self.ln1 = torch.nn.Linear(self.ssize * self.ssize * 8 * self.dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, MyConvo2d):
            if module.conv.weight is not None:
                if module.he_init:
                    torch.nn.init.kaiming_uniform_(module.conv.weight)
                else:
                    torch.nn.init.xavier_uniform_(module.conv.weight)
            if module.conv.bias is not None:
                torch.nn.init.constant_(module.conv.bias, 0.0)
        if isinstance(module, torch.nn.Linear):
            if module.weight is not None:
                torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, input):
        output = self.conv1(input.contiguous())
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, self.ssize * self.ssize * 8 * self.dim)
        output = self.ln1(output)
        return output.squeeze(dim=(-1, -2))


### END: copied from https://github.com/jalola/improved-wgan-pytorch ###
