import torch

class ResConv3DBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 downsample=False, depth_average=False,
                 bottleneck_factor=1,
                 squeeze_excitation=False, squeeze_excitation_bottleneck_factor=4):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param downsample: whether to downsample the input 2x2
        :param depth_average: whether to average over the depth dimension or restrict to middle
        :param bottleneck_factor: factor by which to reduce the number of channels in the bottleneck
        :param squeeze_excitation: whether to use squeeze and excitation
        :param squeeze_excitation_bottleneck_factor: factor by which to reduce the number of channels in the squeeze and excitation block
        """
        super(ResConv3DBlock, self).__init__()
        assert in_channels <= out_channels
        if bottleneck_factor > 1:
            assert out_channels % bottleneck_factor == 0, "out_channels must be divisible by bottleneck_factor"
            assert out_channels % (bottleneck_factor * 4) == 0, "out_channels must be divisible by bottleneck_factor * 4"
        bottleneck_channels = out_channels // bottleneck_factor

        if downsample and depth_average:
            self.avgpool = torch.nn.AvgPool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=0)
        elif downsample:
            self.avgpool = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        elif depth_average:
            self.avgpool = torch.nn.AvgPool3d(kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=0)
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
        self.depth_average = depth_average
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

        if self.downsample and self.depth_average:
            x_init = self.avgpool(x_init)
            assert x_init.shape == (N, self.out_channels, D - 2, H // 2, W // 2)
        elif self.downsample:
            x_init = x_init[:, :, 1:-1, :, :]
            x_init = self.avgpool(x_init)
            assert x_init.shape == (N, self.out_channels, D - 2, H // 2, W // 2)
        elif self.depth_average:
            x_init = self.avgpool(x_init)
            assert x_init.shape == (N, self.out_channels, D - 2, H, W)
        else:
            x_init = x_init[:, :, 1:-1, :, :]
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
                 bottleneck_factor: int=1, squeeze_excitation=False, avg_pool_last_3d=False, return_last2=False):
        super(ResConv3D, self).__init__()

        assert in_channels <= blocks_2d_channels
        if len(blocks_3d_channels) > 0:
            assert blocks_2d_channels <= blocks_3d_channels[0]
            for k in range(1, len(blocks_3d_channels)):
                assert blocks_3d_channels[k - 1] <= blocks_3d_channels[k]
        assert (not avg_pool_last_3d) or len(blocks_3d_channels) > 0, "The option to use average pooling for depth dimension for the last 3D block requires at least one 3D block"

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
                                   squeeze_excitation_bottleneck_factor=get_highest_divisor(blocks_3d_channels[k]),
                                   depth_average=((k == len(blocks_3d_channels) - 1) and avg_pool_last_3d) ))
            num_convs += 1

        self.num_convs = num_convs
        self.return_last2 = return_last2

    def forward(self, x):
        if self.return_last2:
            ret = []
        for k in range(self.num_convs):
            x = self.conv_res[k](x)
            if self.return_last2 and k >= self.num_convs - 2:
                ret.append(x)
        if self.return_last2:
            return ret
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
                 res_conv_blocks=[1, 2, 6, 8, 23, 8], bottleneck_factor=1, squeeze_excitation=False, return_3d_features=False):
        super(ResNet3DBackbone, self).__init__()

        assert len(res_conv_blocks) == len(res_conv3d_blocks), "res_conv_blocks and res_conv3d_blocks must have the same length"
        assert len(channel_progression) == sum(res_conv3d_blocks) + len(res_conv_blocks), "channel_progression must have the same length as the sum of res_conv_blocks and res_conv3d_blocks"
        assert res_conv3d_blocks[-1] == 0, "last ResConv must not contain any 3D elements"
        if return_3d_features:
            assert res_conv3d_blocks == [1, 2, 1, 0, 0, 0], "return_3d_features only supported for the default channel progression"

        self.convs = torch.nn.ModuleList()

        self.initial_conv = torch.nn.Conv3d(in_channels, channel_progression[0], kernel_size=(1, 7, 7),
                                            bias=False, padding="same", padding_mode="replicate")
        self.initial_batchnorm = torch.nn.InstanceNorm3d(channel_progression[0], affine=True)
        self.initial_nonlin = torch.nn.GELU()

        cp_index = 0
        prev_channels = channel_progression[0]
        remaining_3d = sum(res_conv3d_blocks)

        self.first2d = None
        if return_3d_features:
            self.deep3d_channels = []
        for i in range(len(res_conv_blocks)):
            if i == 1 and return_3d_features:
                current_3d_blocks_channels = channel_progression[cp_index + 1:cp_index + 1 + res_conv3d_blocks[i]]
                self.deep3d_channels.append(current_3d_blocks_channels[-2])
                self.deep3d_channels.append(current_3d_blocks_channels[-1])

            if remaining_3d > 0:
                return_last2 = (i == 1) and return_3d_features
                avg_pool_last_3d = (i > 0) and return_3d_features
                self.convs.append(ResConv3D(prev_channels, channel_progression[cp_index], downsample=(i > 0),
                            blocks_2d=res_conv_blocks[i], blocks_3d_channels=channel_progression[cp_index + 1:cp_index + 1 + res_conv3d_blocks[i]],
                            bottleneck_factor=bottleneck_factor if (i > 1) else 1, squeeze_excitation=squeeze_excitation,
                            return_last2=return_last2, avg_pool_last_3d=avg_pool_last_3d))
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
        self.return_3d_features = return_3d_features

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_batchnorm(x)
        x = self.initial_nonlin(x)

        if self.return_3d_features:
            ret = []
        for i in range(self.pyr_height):
            if i == self.first2d:
                x = x.squeeze(2)
            x = self.convs[i](x)
            if self.return_3d_features and i == 1:
                ret.append(x[0])
                ret.append(x[1])
                x = x[1]
        if self.return_3d_features:
            ret.append(x)
            return ret

        return x

    def get_deep3d_channels(self):
        return self.deep3d_channels


class LocalizedROINet(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, num_channels: int):
        super(LocalizedROINet, self).__init__()
        self.backbone = backbone
        self.num_channels = num_channels

        self.outconv = torch.nn.Conv2d(num_channels, 4, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.outconv(x)
        return x

class SupervisedAttentionLinearLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SupervisedAttentionLinearLayer, self).__init__()

        self.attention = torch.nn.Conv2d(in_channels, 4, kernel_size=1, bias=True)
        self.midconv = torch.nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=False)
        self.mid_norm = torch.nn.InstanceNorm2d(out_channels * 4, affine=True)
        self.mid_nonlin = torch.nn.GELU()

        self.mid_bias = torch.nn.Parameter(torch.zeros(4, out_channels), requires_grad=True)

    def forward(self, x):
        N, C, H, W = x.shape

        roi_logits = self.attention(x)
        feature_attention = torch.sigmoid(roi_logits)

        x = self.midconv(x)
        x = self.mid_norm(x)
        x = self.mid_nonlin(x).view(N, 4, -1, H, W)
        x = ((torch.sum(x * feature_attention.unsqueeze(2), dim=(-1, -2)) + self.mid_bias) /
                torch.sum(feature_attention, dim=(-1, -2)).unsqueeze(2) + 1.0)

        return x, roi_logits


class ResConv2DDoubleDownsampleBlock(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        """
        Only applies 2D convolutions to a 3D block
        :param in_channels: number of input channels
        :param mid_channels: number of intermediate channels
        :param out_channels: number of output channels
        """
        super(ResConv2DDoubleDownsampleBlock, self).__init__()
        assert in_channels <= mid_channels <= out_channels, "in_channels <= mid_channels <= out_channels"

        self.avgpool = torch.nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=0)

        # 6 -> 6 kernel for aggressive downsampling
        self.conv1 = torch.nn.Conv3d(in_channels, mid_channels, kernel_size=(1, 6, 6), stride=(1, 2, 2),
                                     bias=False, padding=(0, 2, 2), padding_mode="replicate")
        self.batchnorm1 = torch.nn.InstanceNorm3d(mid_channels, affine=True)
        self.nonlin1 = torch.nn.GELU()

        self.conv2 = torch.nn.Conv3d(mid_channels, out_channels, kernel_size=(1, 6, 6), stride=(1, 2, 2),
                                     bias=False, padding=(0, 2, 2), padding_mode="replicate")
        self.batchnorm2 = torch.nn.InstanceNorm3d(out_channels, affine=True)
        self.nonlin2 = torch.nn.GELU()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

    def forward(self, x):
        N, C, D, H, W = x.shape

        if self.in_channels < self.out_channels:
            x_init = torch.nn.functional.pad(x, (
            0, 0, 0, 0, 0, 0, 0, self.out_channels - self.in_channels), "constant", 0.0)
        else:
            x_init = x
        x_init = self.avgpool(x_init)

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.nonlin1(x)
        assert x.shape == (N, self.mid_channels, D, H // 2, W // 2)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        assert x.shape == (N, self.out_channels, D, H // 4, W // 4)

        result = self.nonlin2(x_init + x)
        return result

class ResConvQuadDownsample(torch.nn.Module):
    def __init__(self, in_channels):
        super(ResConvQuadDownsample, self).__init__()
        self.downsample1 = ResConv2DDoubleDownsampleBlock(in_channels, in_channels * 2, in_channels * 4)
        self.downsample2 = ResConv2DDoubleDownsampleBlock(in_channels * 4, in_channels * 8, in_channels * 16)

    def forward(self, x):
        x = self.downsample1(x)
        x = self.downsample2(x)
        return x

class NeighborhoodROINet(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, first_channels:int, mid_channels:int, last_channels:int,
                       feature_width:int, feature_height:int):
        super(NeighborhoodROINet, self).__init__()
        self.backbone = backbone
        self.head = NeighborhoodROILayer(first_channels, mid_channels, last_channels, feature_width, feature_height)
        self.outconv = torch.nn.Conv3d(first_channels * 32, 4, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = self.outconv(x)
        return x

class NeighborhoodROILayer(torch.nn.Module):
    def __init__(self, first_channels:int, mid_channels:int, last_channels:int,
                       feature_width:int, feature_height:int):
        super(NeighborhoodROILayer, self).__init__()
        self.first_channels = first_channels
        self.mid_channels = mid_channels
        self.last_channels = last_channels
        self.feature_width = feature_width
        self.feature_height = feature_height

        self.spatial_contraction1 = ResConvQuadDownsample(first_channels)
        self.spatial_contraction2 = ResConvQuadDownsample(mid_channels)

        self.downconv1 = torch.nn.Conv3d(last_channels, mid_channels * 16, kernel_size=(1, 3, 3), bias=False,
                                         padding=(0, 1, 1), padding_mode="replicate")
        self.batchnorm1 = torch.nn.InstanceNorm3d(mid_channels * 32, affine=True)
        self.nonlin1 = torch.nn.GELU()

        self.downconv2 = torch.nn.Conv3d(mid_channels * 32, first_channels * 16, kernel_size=(1, 3, 3), bias=False,
                                         padding=(0, 1, 1), padding_mode="replicate")
        self.upsample2 = torch.nn.Upsample(size=(5, feature_height, feature_width), mode="trilinear", align_corners=False)
        self.batchnorm2 = torch.nn.InstanceNorm3d(first_channels * 32, affine=True)
        self.nonlin2 = torch.nn.GELU()

        self.downconv3 = torch.nn.Conv3d(first_channels * 32, first_channels * 16, kernel_size=(1, 3, 3), bias=False,
                                         padding=(0, 1, 1), padding_mode="replicate")
        self.batchnorm3 = torch.nn.InstanceNorm3d(first_channels * 16, affine=True)
        self.nonlin3 = torch.nn.GELU()

    def forward(self, x):
        x_first = self.spatial_contraction1(x[0])
        x_mid = self.spatial_contraction2(x[1])
        x_last = x[2]

        assert x_first.shape[-1] == x_mid.shape[-1] == x_last.shape[-1], "The width of the feature maps must be the same"
        assert x_first.shape[-2] == x_mid.shape[-2] == x_last.shape[-2], "The height of the feature maps must be the same"
        assert x_first.shape[-1] == self.feature_width, "The width of the feature maps must be the same as the specified width"
        assert x_first.shape[-2] == self.feature_height, "The height of the feature maps must be the same as the specified height"

        x = self.downconv1(x_last.unsqueeze(2))
        x = torch.concat([x.expand(-1, -1, 3, -1, -1), x_mid], dim=1)
        x = self.batchnorm1(x)
        x = self.nonlin1(x)

        x = self.downconv2(x)
        x = torch.concat([self.upsample2(x), x_first], dim=1)
        x = self.batchnorm2(x)
        x = self.nonlin2(x)

        x = self.downconv3(x)
        x = self.batchnorm3(x)
        x = self.nonlin3(x)

        return x

class SupervisedAttentionLinearLayer3D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SupervisedAttentionLinearLayer3D, self).__init__()

        self.attention = torch.nn.Conv3d(in_channels, 4, kernel_size=1, bias=True)
        self.midconv = torch.nn.Conv3d(in_channels, out_channels * 4, kernel_size=1, bias=False)
        self.mid_norm = torch.nn.InstanceNorm3d(out_channels * 4, affine=True)
        self.mid_nonlin = torch.nn.GELU()

        self.mid_bias = torch.nn.Parameter(torch.zeros(4, out_channels), requires_grad=True)

    def forward(self, x):
        N, C, D, H, W = x.shape

        roi_logits = self.attention(x)
        feature_attention = torch.sigmoid(roi_logits)

        x = self.midconv(x)
        x = self.mid_norm(x)
        x = self.mid_nonlin(x).view(N, 4, -1, D, H, W)
        x = ((torch.sum(x * feature_attention.unsqueeze(2), dim=(-1, -2, -3)) + self.mid_bias) /
                torch.sum(feature_attention, dim=(-1, -2, -3)).unsqueeze(2) + 1.0)

        return x, roi_logits

class ClassifierNeck(torch.nn.Module):
    def __init__(self, in_channels: int, classification_levels: list[int]):
        super(ClassifierNeck, self).__init__()
        assert len(classification_levels) == 4, "There must be 4 classification heads, for liver, spleen, kidney, bowel"
        for k in classification_levels:
            assert k in [1, 2], "The classification levels must be between 1 and 2"

        self.outconvs = torch.nn.ModuleList()
        for k in classification_levels:
            if k == 1:
                self.outconvs.append(torch.nn.Linear(in_channels, 1, bias=True))
            else:
                self.outconvs.append(torch.nn.Linear(in_channels, 3, bias=True))

    def forward(self, x):
        outputs = []
        for k in range(4):
            outputs.append(self.outconvs[k](x[:, k, :]))
        return outputs

class MeanProbaReductionHead(torch.nn.Module):
    def __init__(self, classification_levels: list[int]):
        super(MeanProbaReductionHead, self).__init__()
        assert len(classification_levels) == 4, "There must be 4 classification heads, for liver, spleen, kidney, bowel"
        for k in classification_levels:
            assert k in [1, 2], "The classification levels must be between 1 and 2"

        self.classification_levels = classification_levels

    def forward(self, x):
        outputs = []
        for k in range(4):
            if self.classification_levels[k] == 1:
                outputs.append(torch.mean(torch.sigmoid(x[k]), dim=0))
            else:
                outputs.append(torch.mean(torch.softmax(x[k], dim=1), dim=0))
        return outputs

class UnionProbaReductionHead(torch.nn.Module):
    def __init__(self, classification_levels: list[int]):
        super(UnionProbaReductionHead, self).__init__()
        assert len(classification_levels) == 4, "There must be 4 classification heads, for liver, spleen, kidney, bowel"
        for k in classification_levels:
            assert k in [1, 2], "The classification levels must be between 1 and 2"

        self.classification_levels = classification_levels

    def forward(self, x):
        outputs = []
        for k in range(4):
            if self.classification_levels[k] == 1:
                outputs.append(1 - torch.prod(1 - torch.sigmoid(x[k]), dim=0))
            else:
                
                outputs.append(torch.softmax(torch.mean(x[k], dim=0), dim=0))
        return outputs

class SupervisedAttentionClassifier(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, backbone_out_channels: int, conv_hidden_channels: int,
                 classification_levels: list[int], reduction="mean"):
        super(SupervisedAttentionClassifier, self).__init__()

        self.backbone = backbone
        self.backbone_out_channels = backbone_out_channels
        self.conv_hidden_channels = conv_hidden_channels

        self.attention_layer = SupervisedAttentionLinearLayer(backbone_out_channels, conv_hidden_channels)
        self.classifier_neck = ClassifierNeck(conv_hidden_channels, classification_levels)

    def forward(self, x):
        x = self.backbone(x)
        x, roi_logits = self.attention_layer(x)
        x = self.classifier_neck(x)

        return x, roi_logits

class SupervisedAttentionClassifier3D(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, backbone_first_channels:int, backbone_mid_channels:int, backbone_last_channels:int,
                       backbone_feature_width:int, backbone_feature_height:int, conv_hidden_channels: int,
                 classification_levels: list[int]):
        super(SupervisedAttentionClassifier3D, self).__init__()

        self.backbone = backbone

        self.ushaped_neck = NeighborhoodROILayer(backbone_first_channels, backbone_mid_channels, backbone_last_channels,
                                                 backbone_feature_width, backbone_feature_height)
        self.attention_layer = SupervisedAttentionLinearLayer3D(backbone_first_channels * 16, conv_hidden_channels)
        self.classifier_neck = ClassifierNeck(conv_hidden_channels, classification_levels)

    def forward(self, x):
        x = self.backbone(x)
        x = self.ushaped_neck(x)
        x, roi_logits = self.attention_layer(x)
        x = self.classifier_neck(x)

        return x, roi_logits
