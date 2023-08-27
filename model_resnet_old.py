import torch


class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False):
        super(Conv, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm1 = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels)
        self.elu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, bias=True, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm2 = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels)
        self.elu2 = torch.nn.ReLU(inplace=True)

        torch.nn.init.constant_(self.conv1.bias, 0.0)
        torch.nn.init.constant_(self.conv2.bias, 0.0)

        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.batchnorm1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.batchnorm2(x)
        x = self.elu2(x)
        return x


class ResConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False, downsample=False, bottleneck_expansion=1,
                 squeeze_excitation=False):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels without bottleneck expansion. The actual number of output channels is out_channels * bottleneck_expansion
        :param use_batch_norm: whether to use batch (instance) normalization
        :param downsample: whether to downsample the input 2x2
        :param bottleneck_expansion: the expansion factor of the bottleneck
        """
        super(ResConvBlock, self).__init__()
        assert in_channels <= out_channels * bottleneck_expansion

        if downsample:
            self.avgpool = torch.nn.AvgPool2d(2)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 1, bias=False, padding="same", padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm1 = torch.nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)  # instance norm
        self.elu1 = torch.nn.ReLU(inplace=True)

        num_groups = out_channels // 8
        if (num_groups == 0) or ((out_channels % num_groups) != 0):
            num_groups = out_channels // 4
        if downsample:
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, bias=False, padding=0,
                                         groups=num_groups)  # x8d, meaning 8 channels in each "capacity" connection
        else:
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding="same",
                                         padding_mode="replicate", groups=num_groups)
        if use_batch_norm:
            self.batchnorm2 = torch.nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)  # instance norm
        self.elu2 = torch.nn.ReLU(inplace=True)

        self.conv3 = torch.nn.Conv2d(out_channels, out_channels * bottleneck_expansion, 1, bias=False, padding="same",
                                     padding_mode="replicate")
        if use_batch_norm:
            self.batchnorm3 = torch.nn.GroupNorm(num_groups=out_channels * bottleneck_expansion,
                                                 num_channels=out_channels * bottleneck_expansion)  # instance norm
        self.elu3 = torch.nn.ReLU(inplace=True)

        if squeeze_excitation:
            self.se_pool = torch.nn.AdaptiveAvgPool2d(1)
            self.se_conv1 = torch.nn.Conv2d(out_channels * bottleneck_expansion, out_channels // 4, 1, bias=True,
                                            padding="same", padding_mode="replicate")
            self.se_relu = torch.nn.ReLU(inplace=True)
            self.se_conv2 = torch.nn.Conv2d(out_channels // 4, out_channels * bottleneck_expansion, 1, bias=True,
                                            padding="same", padding_mode="replicate")
            self.se_sigmoid = torch.nn.Sigmoid()

        self.use_batch_norm = use_batch_norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.bottleneck_expansion = bottleneck_expansion
        self.squeeze_excitation = squeeze_excitation

    def forward(self, x):
        if self.in_channels < self.out_channels * self.bottleneck_expansion:
            x_init = torch.nn.functional.pad(x, (
            0, 0, 0, 0, 0, self.out_channels * self.bottleneck_expansion - self.in_channels), "constant", 0.0)
        else:
            x_init = x

        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.batchnorm1(x)
        x = self.elu1(x)

        if self.downsample:
            x = self.conv2(torch.nn.functional.pad(x, (1, 0, 1, 0), "reflect"))
        else:
            x = self.conv2(x)
        if self.use_batch_norm:
            x = self.batchnorm2(x)
        x = self.elu2(x)

        x = self.conv3(x)
        if self.use_batch_norm:
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
        result = self.elu3(x_init + x)
        return result


class ResConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False, downsample=False, blocks=3,
                 bottleneck_expansion=1, squeeze_excitation=False):
        # bottleneck expansion means how many times the number of channels is increased in the ultimate outputs of resconvs.
        super(ResConv, self).__init__()

        self.conv_res = torch.nn.ModuleList()
        self.conv_res.append(
            ResConvBlock(in_channels, out_channels, use_batch_norm=use_batch_norm, downsample=downsample,
                         bottleneck_expansion=bottleneck_expansion, squeeze_excitation=squeeze_excitation))
        for k in range(1, blocks):
            self.conv_res.append(
                ResConvBlock(out_channels * bottleneck_expansion, out_channels, use_batch_norm=use_batch_norm,
                             bottleneck_expansion=bottleneck_expansion, squeeze_excitation=squeeze_excitation))

        self.blocks = blocks

    def forward(self, x):
        for k in range(self.blocks):
            x = self.conv_res[k](x)

        return x


class ResNetBackbone(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, use_batch_norm=False, use_res_conv=False, pyr_height=4,
                 res_conv_blocks=[2, 3, 4, 6, 10, 15, 15], bottleneck_expansion=1, squeeze_excitation=False,
                 use_initial_conv=False):
        super(ResNetBackbone, self).__init__()
        self.pyr_height = pyr_height
        self.conv_down = torch.nn.ModuleList()

        if use_initial_conv:
            self.initial_conv = torch.nn.Conv2d(in_channels, hidden_channels * bottleneck_expansion, kernel_size=7,
                                                bias=False, padding="same", padding_mode="replicate")
            if use_batch_norm:
                self.initial_batch_norm = torch.nn.GroupNorm(num_groups=hidden_channels * bottleneck_expansion,
                                                             num_channels=hidden_channels * bottleneck_expansion)  # instance norm
            self.initial_elu = torch.nn.ReLU(inplace=True)

        if use_res_conv:
            self.conv0 = ResConv((hidden_channels * bottleneck_expansion) if use_initial_conv else in_channels,
                                 hidden_channels, use_batch_norm=use_batch_norm, blocks=res_conv_blocks[0],
                                 bottleneck_expansion=bottleneck_expansion, squeeze_excitation=squeeze_excitation)
            for i in range(pyr_height - 1):
                self.conv_down.append(
                    ResConv(bottleneck_expansion * hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1),
                            use_batch_norm=use_batch_norm, downsample=True, blocks=res_conv_blocks[i + 1],
                            bottleneck_expansion=bottleneck_expansion, squeeze_excitation=squeeze_excitation))
        else:
            self.conv0 = Conv(hidden_channels, hidden_channels, use_batch_norm=use_batch_norm)
            for i in range(pyr_height - 1):
                self.conv_down.append(
                    Conv(hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1), use_batch_norm=use_batch_norm))
        if use_res_conv:
            self.conv_down.append(ResConv(bottleneck_expansion * hidden_channels * 2 ** (pyr_height - 1),
                                          hidden_channels * 2 ** pyr_height, use_batch_norm=use_batch_norm,
                                          downsample=True, blocks=res_conv_blocks[pyr_height],
                                          bottleneck_expansion=bottleneck_expansion,
                                          squeeze_excitation=squeeze_excitation))
        else:
            self.conv_down.append(Conv(hidden_channels * 2 ** (pyr_height - 1), hidden_channels * 2 ** pyr_height,
                                       use_batch_norm=use_batch_norm))
        self.maxpool = torch.nn.MaxPool2d(2)
        self.use_res_conv = use_res_conv
        self.use_batch_norm = use_batch_norm
        self.use_initial_conv = use_initial_conv

    def forward(self, x):
        if self.use_initial_conv:
            x = self.initial_conv(x)
            if self.use_batch_norm:
                x = self.initial_batch_norm(x)
            x = self.initial_elu(x)

        # contracting path
        ret = []
        x = self.conv0(x)
        ret.append(x)
        for i in range(self.pyr_height - 1):
            if self.use_res_conv:
                x = self.conv_down[i](x)
            else:
                x = self.conv_down[i](self.maxpool(x))
            ret.append(x)
        if not self.use_res_conv:
            x = self.conv_down[-1](self.maxpool(x))
        else:
            x = self.conv_down[-1](x)
        ret.append(x)

        return ret

class FullClassifier(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, neck: torch.nn.Module, head: torch.nn.Module):
        super(FullClassifier, self).__init__()

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        x = x.squeeze(2)
        x = self.backbone(x)
        x = [z.unsqueeze(2) for z in x]
        attn, outs = self.neck(x)
        probas = self.head(outs)
        return probas
