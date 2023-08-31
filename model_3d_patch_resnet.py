import torch

class ResConv3DBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, bottleneck_factor=1,
                 squeeze_excitation=False, squeeze_excitation_bottleneck_factor=4):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param downsample: whether to downsample the input 2x2
        :param squeeze_excitation: whether to use squeeze and excitation
        :param squeeze_excitation_bottleneck_factor: factor by which to reduce the number of channels in the squeeze and excitation block
        """
        super(ResConv3DBlock, self).__init__()
        assert in_channels <= out_channels
        if bottleneck_factor > 1:
            assert out_channels % bottleneck_factor == 0, "out_channels must be divisible by bottleneck_factor"
            assert out_channels % (bottleneck_factor * 4) == 0, "out_channels must be divisible by bottleneck_factor * 4"
        bottleneck_channels = out_channels // bottleneck_factor

        if downsample:
            self.avgpool = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        if bottleneck_factor > 1:
            self.conv1 = torch.nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1, bias=False,
                                         padding="same", padding_mode="replicate")
            self.batchnorm1 = torch.nn.InstanceNorm3d(bottleneck_channels, affine=True)
            self.nonlin1 = torch.nn.GELU()

            num_groups = bottleneck_channels // 4
            if downsample:
                self.conv2 = torch.nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=(1, 2, 2),
                                             bias=False, padding=0, groups=num_groups) # 3d conv
            else:
                self.conv2 = torch.nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=3, bias=False,
                                             padding=(0, 1, 1), padding_mode="replicate", groups=num_groups) # 3d conv
            self.batchnorm2 = torch.nn.InstanceNorm3d(bottleneck_channels, affine=True)
            self.nonlin2 = torch.nn.GELU()

            self.conv3 = torch.nn.Conv3d(bottleneck_channels, out_channels, kernel_size=1, bias=False,
                                         padding="same", padding_mode="replicate")
            self.batchnorm3 = torch.nn.InstanceNorm3d(out_channels, affine=True)
            self.nonlin3 = torch.nn.GELU()
        else:
            self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), bias=False,
                                         padding="same", padding_mode="replicate") # depth preserving conv
            self.batchnorm1 = torch.nn.InstanceNorm3d(out_channels, affine=True)
            self.nonlin1 = torch.nn.GELU()

            if downsample:
                self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=(1, 2, 2),
                                             bias=False, padding=0) # 3d conv
            else:
                self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, bias=False,
                                             padding=(0, 1, 1), padding_mode="replicate") # 3d conv
            self.batchnorm2 = torch.nn.InstanceNorm3d(out_channels, affine=True)
            self.nonlin2 = torch.nn.GELU()
        if squeeze_excitation:
            assert out_channels % squeeze_excitation_bottleneck_factor == 0, "out_channels must be divisible by squeeze_excitation_bottleneck_factor"
            self.se_pool = torch.nn.AdaptiveAvgPool3d(1)
            self.se_conv1 = torch.nn.Conv3d(out_channels, out_channels // squeeze_excitation_bottleneck_factor,
                                            kernel_size=1, bias=True, padding="same", padding_mode="replicate")
            self.se_relu = torch.nn.ReLU()
            self.se_conv2 = torch.nn.Conv3d(out_channels // squeeze_excitation_bottleneck_factor, out_channels,
                                            kernel_size=1, bias=True, padding="same", padding_mode="replicate")
            self.se_sigmoid = torch.nn.Sigmoid()

        self.bottleneck_factor = bottleneck_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.squeeze_excitation = squeeze_excitation

    def forward(self, x):
        N, C, D, H, W = x.shape
        assert C == self.in_channels

        if self.in_channels < self.out_channels:
            x_init = torch.nn.functional.pad(x, (
            0, 0, 0, 0, 0, 0, 0, self.out_channels - self.in_channels), "constant", 0.0)
        else:
            x_init = x

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.nonlin1(x)
        assert x.shape == (N, self.out_channels // self.bottleneck_factor, D, H, W)

        if self.downsample:
            x = self.conv2(torch.nn.functional.pad(x, (1, 0, 1, 0, 0, 0), "reflect"))
            assert x.shape == (N, self.out_channels // self.bottleneck_factor, D - 2, H // 2, W // 2)
        else:
            x = self.conv2(x)
            assert x.shape == (N, self.out_channels // self.bottleneck_factor, D - 2, H, W)
        x = self.batchnorm2(x)

        if self.bottleneck_factor > 1:
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = self.batchnorm3(x)

        if self.squeeze_excitation:
            x_se = self.se_pool(x)
            x_se = self.se_conv1(x_se)
            x_se = self.se_relu(x_se)
            x_se = self.se_conv2(x_se)
            x_se = self.se_sigmoid(x_se)
            x = x * x_se

        x_init = x_init[:, :, 1:-1, :, :]
        if self.downsample:
            x_init = self.avgpool(x_init)
            assert x_init.shape == (N, self.out_channels, D - 2, H // 2, W // 2)
        else:
            assert x_init.shape == (N, self.out_channels, D - 2, H, W)
        if self.bottleneck_factor > 1:
            result = self.nonlin3(x_init + x)
        else:
            result = self.nonlin2(x_init + x)
        return result

class ResConv2DBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False,
                 bottleneck_factor=1, squeeze_excitation=False, squeeze_excitation_bottleneck_factor=4):
        """
        Only applies 2D convolutions to a 3D block
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param normalization_type: "batchnorm" or "instancenorm"
        :param downsample: whether to downsample the input 2x2
        :param bottleneck_factor: how much to expand the number of channels in the bottleneck
        :param squeeze_excitation: whether to use squeeze and excitation
        :param squeeze_excitation_bottleneck_factor: how much to reduce the number of channels in the squeeze and excitation
        """
        super(ResConv2DBlock, self).__init__()
        assert in_channels <= out_channels
        if bottleneck_factor > 1:
            assert out_channels % bottleneck_factor == 0, "out_channels must be divisible by bottleneck_factor"
            assert out_channels % (bottleneck_factor * 4) == 0, "out_channels must be divisible by bottleneck_factor * 4"

        bottleneck_channels = out_channels // bottleneck_factor
        if downsample:
            self.avgpool = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        if bottleneck_factor > 1:
            # 1 -> 3 -> 1 kernel, with bottleneck and capacity
            self.conv1 = torch.nn.Conv3d(in_channels, bottleneck_channels, 1, bias=False, padding="same", padding_mode="replicate")
            self.batchnorm1 = torch.nn.InstanceNorm3d(bottleneck_channels, affine=True)
            self.nonlin1 = torch.nn.GELU()

            num_groups = bottleneck_channels // 4
            if downsample:
                self.conv2 = torch.nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                             bias=False, padding=0, groups=num_groups)  # x4d, meaning 4 channels in each "capacity" connection
            else:
                self.conv2 = torch.nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=(1, 3, 3), bias=False,
                                             padding="same", padding_mode="replicate", groups=num_groups)
            self.batchnorm2 = torch.nn.InstanceNorm3d(bottleneck_channels, affine=True)
            self.nonlin2 = torch.nn.GELU()

            self.conv3 = torch.nn.Conv3d(bottleneck_channels, out_channels, 1, bias=False, padding="same",
                                         padding_mode="replicate")
            self.batchnorm3 = torch.nn.InstanceNorm3d(out_channels, affine=True)
            self.nonlin3 = torch.nn.GELU()
        else:
            # 3 -> 3 kernel, without bottleneck or capacity
            self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), bias=False, padding="same",
                                         padding_mode="replicate")
            self.batchnorm1 = torch.nn.InstanceNorm3d(out_channels, affine=True)
            self.nonlin1 = torch.nn.GELU()

            if downsample:
                self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                             bias=False, padding=0)
            else:
                self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), bias=False,
                                             padding="same", padding_mode="replicate")
            self.batchnorm2 = torch.nn.InstanceNorm3d(out_channels, affine=True)
            self.nonlin2 = torch.nn.GELU()

        if squeeze_excitation:
            assert out_channels % squeeze_excitation_bottleneck_factor == 0, "out_channels must be divisible by squeeze_excitation_bottleneck_factor"
            self.se_pool = torch.nn.AdaptiveAvgPool3d(1)
            self.se_conv1 = torch.nn.Conv3d(out_channels, out_channels // squeeze_excitation_bottleneck_factor, kernel_size=1, bias=True,
                                            padding="same", padding_mode="replicate")
            self.se_relu = torch.nn.ReLU()
            self.se_conv2 = torch.nn.Conv3d(out_channels // squeeze_excitation_bottleneck_factor, out_channels, kernel_size=1, bias=True,
                                            padding="same", padding_mode="replicate")
            self.se_sigmoid = torch.nn.Sigmoid()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.bottleneck_factor = bottleneck_factor
        self.squeeze_excitation = squeeze_excitation

    def forward(self, x):
        N, C, D, H, W = x.shape

        if self.in_channels < self.out_channels:
            x_init = torch.nn.functional.pad(x, (
            0, 0, 0, 0, 0, 0, 0, self.out_channels - self.in_channels), "constant", 0.0)
        else:
            x_init = x

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.nonlin1(x)
        assert x.shape == (N, self.out_channels // self.bottleneck_factor, D, H, W)

        if self.downsample:
            x = self.conv2(torch.nn.functional.pad(x, (1, 0, 1, 0, 0, 0), "reflect"))
        else:
            x = self.conv2(x)
        x = self.batchnorm2(x)

        if self.bottleneck_factor > 1:
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = self.batchnorm3(x)

        if self.squeeze_excitation:
            x_se = self.se_pool(x)
            x_se = self.se_conv1(x_se)
            x_se = self.se_relu(x_se)
            x_se = self.se_conv2(x_se)
            x_se = self.se_sigmoid(x_se)
            x = x * x_se

        if self.downsample:
            x_init = self.avgpool(x_init)
        if self.bottleneck_factor > 1:
            result = self.nonlin3(x_init + x)
        else:
            result = self.nonlin2(x_init + x)
        return result

class ResConvPure2DBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False,
                 bottleneck_factor=1, squeeze_excitation=False):
        """
        2D convolutions to a 2D block
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param downsample: whether to downsample the input 2x2
        :param bottleneck_factor: how much to expand the number of channels in the bottleneck
        :param squeeze_excitation: whether to use squeeze and excitation
        """
        super(ResConvPure2DBlock, self).__init__()
        assert in_channels <= out_channels
        if bottleneck_factor > 1:
            assert out_channels % bottleneck_factor == 0, "out_channels must be divisible by bottleneck_factor"
            assert out_channels % (bottleneck_factor * 4) == 0, "out_channels must be divisible by bottleneck_factor * 4"

        bottleneck_channels = out_channels // bottleneck_factor
        if downsample:
            self.avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        if bottleneck_factor > 1:
            # 1 -> 3 -> 1 kernel, with bottleneck and capacity
            self.conv1 = torch.nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False, padding="same", padding_mode="replicate")
            self.batchnorm1 = torch.nn.InstanceNorm2d(bottleneck_channels, affine=True)
            self.nonlin1 = torch.nn.GELU()

            num_groups = bottleneck_channels // 4
            if downsample:
                self.conv2 = torch.nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=2,
                                             bias=False, padding=0, groups=num_groups)  # x4d, meaning 4 channels in each "capacity" connection
            else:
                self.conv2 = torch.nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, bias=False,
                                             padding="same", padding_mode="replicate", groups=num_groups)
            self.batchnorm2 = torch.nn.InstanceNorm2d(bottleneck_channels, affine=True)
            self.nonlin2 = torch.nn.GELU()

            self.conv3 = torch.nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False, padding="same",
                                         padding_mode="replicate")
            self.batchnorm3 = torch.nn.InstanceNorm2d(out_channels, affine=True)
            self.nonlin3 = torch.nn.GELU()
        else:
            # 3 -> 3 kernel, without bottleneck or capacity
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding="same",
                                         padding_mode="replicate")
            self.batchnorm1 = torch.nn.InstanceNorm2d(out_channels, affine=True)
            self.nonlin1 = torch.nn.GELU()

            if downsample:
                self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2,
                                             bias=False, padding=0)
            else:
                self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False,
                                             padding="same", padding_mode="replicate")
            self.batchnorm2 = torch.nn.InstanceNorm2d(out_channels, affine=True)
            self.nonlin2 = torch.nn.GELU()

        if squeeze_excitation:
            assert out_channels % 4 == 0, "out_channels must be divisible by 4"
            self.se_pool = torch.nn.AdaptiveAvgPool2d(1)
            self.se_conv1 = torch.nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=True,
                                            padding="same", padding_mode="replicate")
            self.se_relu = torch.nn.ReLU()
            self.se_conv2 = torch.nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=True,
                                            padding="same", padding_mode="replicate")
            self.se_sigmoid = torch.nn.Sigmoid()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.bottleneck_factor = bottleneck_factor
        self.squeeze_excitation = squeeze_excitation

    def forward(self, x):
        N, C, H, W = x.shape

        if self.in_channels < self.out_channels:
            x_init = torch.nn.functional.pad(x, (
            0, 0, 0, 0, 0, self.out_channels - self.in_channels), "constant", 0.0)
        else:
            x_init = x

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.nonlin1(x)
        assert x.shape == (N, self.out_channels // self.bottleneck_factor, H, W)

        if self.downsample:
            x = self.conv2(torch.nn.functional.pad(x, (1, 0, 1, 0), "reflect"))
        else:
            x = self.conv2(x)
        x = self.batchnorm2(x)

        if self.bottleneck_factor > 1:
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = self.batchnorm3(x)

        if self.squeeze_excitation:
            x_se = self.se_pool(x)
            x_se = self.se_conv1(x_se)
            x_se = self.se_relu(x_se)
            x_se = self.se_conv2(x_se)
            x_se = self.se_sigmoid(x_se)
            x = x * x_se

        if self.downsample:
            x_init = self.avgpool(x_init)
        if self.bottleneck_factor > 1:
            result = self.nonlin3(x_init + x)
        else:
            result = self.nonlin2(x_init + x)
        return result

def get_highest_divisor(num: int):
    for k in range(4, 1, -1):
        if num % k == 0:
            return k

class ResConv3D(torch.nn.Module):
    def __init__(self, in_channels: int, blocks_2d_channels: int, downsample=False, blocks_2d: int=3, blocks_3d_channels=[],
                 bottleneck_factor: int=1, squeeze_excitation=False):
        super(ResConv3D, self).__init__()

        assert in_channels <= blocks_2d_channels
        if len(blocks_3d_channels) > 0:
            assert blocks_2d_channels <= blocks_3d_channels[0]
            for k in range(1, len(blocks_3d_channels)):
                assert blocks_3d_channels[k - 1] <= blocks_3d_channels[k]

        self.conv_res = torch.nn.ModuleList()
        num_convs = 1
        self.conv_res.append(
            ResConv2DBlock(in_channels, blocks_2d_channels, downsample=downsample,
                         bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                           squeeze_excitation_bottleneck_factor=get_highest_divisor(blocks_2d_channels)))
        for k in range(1, blocks_2d):
            self.conv_res.append(
                ResConv2DBlock(blocks_2d_channels, blocks_2d_channels,
                             bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                               squeeze_excitation_bottleneck_factor=get_highest_divisor(blocks_2d_channels)))
            num_convs += 1
        for k in range(len(blocks_3d_channels)):
            if k == 0:
                self.conv_res.append(
                    ResConv3DBlock(blocks_2d_channels, blocks_3d_channels[k],
                                 bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                                   squeeze_excitation_bottleneck_factor=get_highest_divisor(blocks_3d_channels[k])))
            else:
                self.conv_res.append(
                    ResConv3DBlock(blocks_3d_channels[k - 1], blocks_3d_channels[k],
                                 bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                                   squeeze_excitation_bottleneck_factor=get_highest_divisor(blocks_3d_channels[k])))
            num_convs += 1

        self.num_convs = num_convs

    def forward(self, x):
        for k in range(self.num_convs):
            x = self.conv_res[k](x)
        return x

class ResConvPure2D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int, downsample=False,
                 bottleneck_factor: int=1, squeeze_excitation=False):
        super(ResConvPure2D, self).__init__()

        assert in_channels <= out_channels

        self.conv_res = torch.nn.ModuleList()
        self.conv_res.append(
            ResConvPure2DBlock(in_channels, out_channels, downsample=downsample,
                         bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation))
        for k in range(1, blocks):
            self.conv_res.append(
                ResConvPure2DBlock(out_channels, out_channels,
                             bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation))

        self.blocks = blocks

    def forward(self, x):
        for k in range(self.blocks):
            x = self.conv_res[k](x)
        return x


class ResNet3DBackbone(torch.nn.Module):
    def __init__(self, in_channels: int, channel_progression: list[int]=[2, 3, 6, 9, 15, 32, 128, 256, 512, 1024], res_conv3d_blocks=[1, 2, 1, 0, 0, 0],
                 res_conv_blocks=[1, 2, 6, 8, 23, 8], bottleneck_factor=1, squeeze_excitation=False):
        super(ResNet3DBackbone, self).__init__()

        assert len(res_conv_blocks) == len(res_conv3d_blocks), "res_conv_blocks and res_conv3d_blocks must have the same length"
        assert len(channel_progression) == sum(res_conv3d_blocks) + len(res_conv_blocks), "channel_progression must have the same length as the sum of res_conv_blocks and res_conv3d_blocks"
        assert res_conv3d_blocks[-1] == 0, "last ResConv must not contain any 3D elements"

        self.convs = torch.nn.ModuleList()

        self.initial_conv = torch.nn.Conv3d(in_channels, channel_progression[0], kernel_size=(1, 7, 7),
                                            bias=False, padding="same", padding_mode="replicate")
        self.initial_batchnorm = torch.nn.InstanceNorm3d(channel_progression[0], affine=True)
        self.initial_nonlin = torch.nn.GELU()

        cp_index = 0
        prev_channels = channel_progression[0]
        remaining_3d = sum(res_conv3d_blocks)

        self.first2d = None
        for i in range(len(res_conv_blocks)):
            if remaining_3d > 0:
                self.convs.append(ResConv3D(prev_channels, channel_progression[cp_index], downsample=(i > 0),
                            blocks_2d=res_conv_blocks[i], blocks_3d_channels=channel_progression[cp_index + 1:cp_index + 1 + res_conv3d_blocks[i]],
                            bottleneck_factor=bottleneck_factor if (i > 1) else 1, squeeze_excitation=squeeze_excitation))
            else:
                if self.first2d is None:
                    self.first2d = i
                self.convs.append(ResConvPure2D(prev_channels, channel_progression[cp_index], downsample=(i > 0),
                            blocks=res_conv_blocks[i], bottleneck_factor=bottleneck_factor if (i > 1) else 1,
                            squeeze_excitation=squeeze_excitation))
            prev_channels = channel_progression[cp_index + res_conv3d_blocks[i]]
            cp_index = cp_index + 1 + res_conv3d_blocks[i]
            remaining_3d -= res_conv3d_blocks[i]

        if self.first2d is None:
            self.first2d = -1
        self.pyr_height = len(res_conv_blocks)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_batchnorm(x)
        x = self.initial_nonlin(x)

        # contracting path
        ret = []
        for i in range(self.pyr_height):
            if i == self.first2d:
                x = x.squeeze(2)
            x = self.convs[i](x)
            ret.append(x)

        return ret

class LocalizedROINet(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, num_channels: int):
        super(LocalizedROINet, self).__init__()
        self.backbone = backbone
        self.num_channels = num_channels

        self.outconv = torch.nn.Conv2d(num_channels, 4, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.outconv(x)
        return x
