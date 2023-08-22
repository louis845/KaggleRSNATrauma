import torch

BATCH_NORM_MOMENTUM = 0.1

class ResConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, normalization_type="batchnorm",
                 downsample=False, bottleneck_factor=1, squeeze_excitation=False):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param normalization_type: "batchnorm" or "instancenorm"
        :param downsample: whether to downsample the input 2x2
        :param bottleneck_factor: how much to expand the number of channels in the bottleneck
        :param squeeze_excitation: whether to use squeeze and excitation
        """
        super(ResConvBlock, self).__init__()
        assert in_channels <= out_channels
        assert out_channels % bottleneck_factor == 0, "out_channels must be divisible by bottleneck_factor"
        assert out_channels % (bottleneck_factor * 4) == 0, "out_channels must be divisible by bottleneck_factor * 4"

        bottleneck_channels = out_channels // bottleneck_factor
        if downsample:
            self.avgpool = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        self.conv1 = torch.nn.Conv3d(in_channels, bottleneck_channels, 1, bias=False, padding="same", padding_mode="replicate")
        if normalization_type == "batchnorm":
            self.batchnorm1 = torch.nn.BatchNorm3d(bottleneck_channels, momentum=BATCH_NORM_MOMENTUM)
        elif normalization_type == "instancenorm":
            self.batchnorm1 = torch.nn.InstanceNorm3d(bottleneck_channels)
        self.elu1 = torch.nn.ReLU(inplace=True)

        num_groups = bottleneck_channels // 4
        if downsample:
            self.conv2 = torch.nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                         bias=False, padding=0, groups=num_groups)  # x4d, meaning 4 channels in each "capacity" connection
        else:
            self.conv2 = torch.nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=(1, 3, 3), bias=False,
                                         padding="same", padding_mode="replicate", groups=num_groups)
        if normalization_type == "batchnorm":
            self.batchnorm2 = torch.nn.BatchNorm3d(bottleneck_channels, momentum=BATCH_NORM_MOMENTUM)
        elif normalization_type == "instancenorm":
            self.batchnorm2 = torch.nn.InstanceNorm3d(bottleneck_channels)
        self.elu2 = torch.nn.ReLU(inplace=True)

        self.conv3 = torch.nn.Conv3d(bottleneck_channels, out_channels, 1, bias=False, padding="same",
                                     padding_mode="replicate")
        if normalization_type == "batchnorm":
            self.batchnorm3 = torch.nn.BatchNorm3d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        elif normalization_type == "instancenorm":
            self.batchnorm3 = torch.nn.InstanceNorm3d(out_channels)
        self.elu3 = torch.nn.ReLU(inplace=True)

        if squeeze_excitation:
            assert out_channels % 4 == 0, "out_channels must be divisible by 4"
            self.se_pool = torch.nn.AdaptiveAvgPool2d(1)
            self.se_conv1 = torch.nn.Conv3d(out_channels, out_channels // 4, kernel_size=1, bias=True,
                                            padding="same", padding_mode="replicate")
            self.se_relu = torch.nn.ReLU(inplace=True)
            self.se_conv2 = torch.nn.Conv3d(out_channels // 4, out_channels, kernel_size=1, bias=True,
                                            padding="same", padding_mode="replicate")
            self.se_sigmoid = torch.nn.Sigmoid()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.bottleneck_factor = bottleneck_factor
        self.squeeze_excitation = squeeze_excitation

    def forward(self, x):
        if self.in_channels < self.out_channels:
            x_init = torch.nn.functional.pad(x, (
            0, 0, 0, 0, 0, 0, 0, self.out_channels - self.in_channels), "constant", 0.0)
        else:
            x_init = x

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.elu1(x)

        if self.downsample:
            x = self.conv2(torch.nn.functional.pad(x, (1, 0, 1, 0), "reflect"))
        else:
            x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu2(x)

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
        result = self.elu3(x_init + x)
        return result


class ResConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, normalization_type="batchnorm", downsample=False, blocks=3,
                 bottleneck_factor=1, squeeze_excitation=False):
        super(ResConv, self).__init__()

        if out_channels < 16:
            bottleneck_factor = 1

        self.conv_res = torch.nn.ModuleList()
        self.conv_res.append(
            ResConvBlock(in_channels, out_channels, normalization_type=normalization_type, downsample=downsample,
                         bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation))
        for k in range(1, blocks):
            self.conv_res.append(
                ResConvBlock(out_channels, out_channels, normalization_type=normalization_type,
                             bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation))

        self.blocks = blocks

    def forward(self, x):
        for k in range(self.blocks):
            x = self.conv_res[k](x)

        return x


class ResNetBackbone(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, normalization_type="batchnorm", pyr_height=5,
                 res_conv_blocks=[2, 6, 8, 23, 8], bottleneck_factor=1, squeeze_excitation=False,
                 hidden_channels_override: list[int]=None):
        super(ResNetBackbone, self).__init__()

        if hidden_channels_override is None:
            hidden_channels_override = [hidden_channels * 2 ** i for i in range(pyr_height - 1)]

        self.pyr_height = pyr_height
        self.convs = torch.nn.ModuleList()

        self.initial_conv = torch.nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 7, 7),
                                            bias=False, padding="same", padding_mode="replicate")
        if normalization_type == "batchnorm":
            self.initial_batchnorm = torch.nn.BatchNorm3d(hidden_channels, momentum=BATCH_NORM_MOMENTUM)
        elif normalization_type == "instancenorm":
            self.initial_batchnorm = torch.nn.InstanceNorm3d(hidden_channels)
        self.initial_elu = torch.nn.ReLU(inplace=True)

        self.convs.append(ResConv(hidden_channels, hidden_channels, normalization_type=normalization_type,
                                      blocks=res_conv_blocks[0], bottleneck_factor=bottleneck_factor,
                                      squeeze_excitation=squeeze_excitation))
        for i in range(pyr_height - 1):
            self.convs.append(ResConv(hidden_channels_override[i], hidden_channels_override[i] * 2,
                        normalization_type=normalization_type, downsample=True, blocks=res_conv_blocks[i + 1],
                        bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation))

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_batch_norm(x)
        x = self.initial_elu(x)

        # contracting path
        ret = []
        for i in range(self.pyr_height):
            x = self.convs[i](x)
            ret.append(x)

        return ret

class PatchAttnClassifierNeck(torch.nn.Module):
    def __init__(self, channels, out_classes=[3], key_dim=8):
        super(PatchAttnClassifierNeck, self).__init__()

        self.channels = channels
        self.out_classes = out_classes
        self.num_outs = len(out_classes)
        self.key_dim = key_dim

        self.query = torch.nn.Conv3d(channels, key_dim * self.num_outs, kernel_size=1, bias=False)

        self.spatch_pool = torch.nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1, padding=0)
        self.key = torch.nn.Conv3d(channels, key_dim * self.num_outs, kernel_size=1, bias=False)
        self.value = torch.nn.Conv3d(channels, channels * self.num_outs, kernel_size=1, bias=False)

        self.nonlinearity = torch.nn.ReLU(inplace=True)
        self.outconvs = torch.nn.ModuleList()
        for i in range(self.num_outs):
            self.outconvs.append(torch.nn.Conv3d(channels, out_classes[i], kernel_size=1, bias=False))

        self.temperature = torch.sqrt(torch.tensor(1.0 / key_dim))

    def forward(self, x):
        x = x[-1]
        query = self.query(torch.mean(x, dim=(-1, -2), keepdim=True))
        assert query.shape == (x.shape[0], self.num_outs * self.key_dim, x.shape[2], 1, 1)

        x = self.spatch_pool(x)
        N = x.shape[0]
        D = x.shape[2]
        H = x.shape[3]
        W = x.shape[4]

        key = self.key(x)
        value = self.value(x)
        assert key.shape == (N, self.num_outs * self.key_dim, D, H, W)
        assert value.shape == (N, self.num_outs * self.channels, D, H, W)

        query = query.view(N, self.num_outs, self.key_dim, D, 1)
        key = key.view(N, self.num_outs, self.key_dim, D, H * W)
        value = value.view(N, self.num_outs, self.channels, D, H * W)

        dot = torch.sum(query * key, dim=2, keepdim=True) * self.temperature
        attn = torch.softmax(dot, dim=-1)
        out = self.nonlinearity(torch.sum(value * attn, dim=-1))
        out = out.view(N, self.num_outs, self.channels, D, 1, 1)

        outs = []
        for i in range(self.num_outs):
            outs.append(self.outconvs[i](out[:, i, ...]))

        # output shape: (N, self.num_outs, D, H, W), [(N, sum(self.out_classes), D, 1, 1)]
        return attn.view(N, self.num_outs, D, H, W), outs

class MeanProbaReductionHead(torch.nn.Module):
    def __init__(self, channels, out_classes=[3]):
        super(MeanProbaReductionHead, self).__init__()

        self.channels = channels
        self.out_classes = out_classes
        self.num_outs = len(out_classes)

    def forward(self, outs):
        probas = []
        for k in range(self.num_outs):
            out = outs[k]
            out = out.squeeze(-1).squeeze(-1)
            if self.out_classes[k] == 1:
                out = torch.sigmoid(out).squeeze(1)
            else:
                out = torch.softmax(out, dim=1)
            probas.append(torch.mean(out, dim=-1))
        return probas

class UnionProbaReductionHead(torch.nn.Module):
    def __init__(self, channels, out_classes=[1]):
        super(UnionProbaReductionHead, self).__init__()

        for k in out_classes:
            assert k == 1, "UnionProbaReductionHead only supports binary classification"

        self.channels = channels
        self.out_classes = out_classes
        self.num_outs = len(out_classes)

    def forward(self, outs):
        probas = []
        for out in outs:
            out = out.squeeze(-1).squeeze(-1).squeeze(1)
            out = torch.sigmoid(out)
            probas.append(1 - torch.prod(1 - out, dim=2))
        return probas

class FullClassifier(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, neck: torch.nn.Module, head: torch.nn.Module):
        super(FullClassifier, self).__init__()

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        attn, outs = self.neck(x)
        probas = self.head(outs)
        return probas