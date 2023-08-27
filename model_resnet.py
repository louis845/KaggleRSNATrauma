import torch

BATCH_NORM_MOMENTUM = 0.1

BATCH_NORM = "batchnorm"
INSTANCE_NORM = "instancenorm"

class ResConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, normalization_type=INSTANCE_NORM,
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
        assert normalization_type in [BATCH_NORM, INSTANCE_NORM]

        bottleneck_channels = out_channels // bottleneck_factor

        if normalization_type == BATCH_NORM:
            if downsample:
                self.avgpool = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

            self.conv1 = torch.nn.Conv3d(in_channels, bottleneck_channels, 1, bias=False, padding="same", padding_mode="replicate")
            self.batchnorm1 = torch.nn.BatchNorm3d(bottleneck_channels, momentum=BATCH_NORM_MOMENTUM)
            self.elu1 = torch.nn.GELU()

            num_groups = bottleneck_channels // 4
            if downsample:
                self.conv2 = torch.nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                             bias=False, padding=0, groups=num_groups)  # x4d, meaning 4 channels in each "capacity" connection
            else:
                self.conv2 = torch.nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=(1, 3, 3), bias=False,
                                             padding="same", padding_mode="replicate", groups=num_groups)
            self.batchnorm2 = torch.nn.BatchNorm3d(bottleneck_channels, momentum=BATCH_NORM_MOMENTUM)
            self.elu2 = torch.nn.GELU()

            self.conv3 = torch.nn.Conv3d(bottleneck_channels, out_channels, 1, bias=False, padding="same",
                                         padding_mode="replicate")
            self.batchnorm3 = torch.nn.BatchNorm3d(out_channels, momentum=BATCH_NORM_MOMENTUM)
            self.elu3 = torch.nn.GELU()

            if squeeze_excitation:
                assert out_channels % 4 == 0, "out_channels must be divisible by 4"
                self.se_conv1 = torch.nn.Conv3d(out_channels, out_channels // 4, kernel_size=1, bias=True,
                                                padding="same", padding_mode="replicate")
                self.se_relu = torch.nn.GELU()
                self.se_conv2 = torch.nn.Conv3d(out_channels // 4, out_channels, kernel_size=1, bias=True,
                                                padding="same", padding_mode="replicate")
                self.se_sigmoid = torch.nn.Sigmoid()
        else:
            if downsample:
                self.avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

            self.conv1 = torch.nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False, padding="same", padding_mode="replicate")
            self.batchnorm1 = torch.nn.InstanceNorm2d(bottleneck_channels, affine=True)
            self.elu1 = torch.nn.GELU()

            num_groups = bottleneck_channels // 4
            if downsample:
                self.conv2 = torch.nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=2,
                                             bias=False, padding=0, groups=num_groups)  # x4d, meaning 4 channels in each "capacity" connection
            else:
                self.conv2 = torch.nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, bias=False,
                                             padding="same", padding_mode="replicate", groups=num_groups)
            self.batchnorm2 = torch.nn.InstanceNorm2d(bottleneck_channels, affine=True)
            self.elu2 = torch.nn.GELU()

            self.conv3 = torch.nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False, padding="same",
                                         padding_mode="replicate")
            self.batchnorm3 = torch.nn.InstanceNorm2d(out_channels, affine=True)
            self.elu3 = torch.nn.GELU()

            if squeeze_excitation:
                assert out_channels % 4 == 0, "out_channels must be divisible by 4"
                self.se_pool = torch.nn.AdaptiveAvgPool2d(1)
                self.se_conv1 = torch.nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=True,
                                                padding="same", padding_mode="replicate")
                self.se_relu = torch.nn.ReLU()
                self.se_conv2 = torch.nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=True,
                                                padding="same", padding_mode="replicate")
                self.se_sigmoid = torch.nn.Sigmoid()

        self.normalization_type = normalization_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.bottleneck_factor = bottleneck_factor
        self.squeeze_excitation = squeeze_excitation

    def forward(self, x):
        if self.normalization_type == BATCH_NORM:
            if self.in_channels < self.out_channels:
                x_init = torch.nn.functional.pad(x, (
                0, 0, 0, 0, 0, 0, 0, self.out_channels - self.in_channels), "constant", 0.0)
            else:
                x_init = x

            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.elu1(x)

            if self.downsample:
                x = self.conv2(torch.nn.functional.pad(x, (1, 0, 1, 0, 0, 0), "reflect"))
            else:
                x = self.conv2(x)
            x = self.batchnorm2(x)
            x = self.elu2(x)

            x = self.conv3(x)
            x = self.batchnorm3(x)

            if self.squeeze_excitation:
                x_se = torch.mean(x, dim=(3, 4), keepdim=True)
                x_se = self.se_conv1(x_se)
                x_se = self.se_relu(x_se)
                x_se = self.se_conv2(x_se)
                x_se = self.se_sigmoid(x_se)
                x = x * x_se

            if self.downsample:
                x_init = self.avgpool(x_init)
            result = self.elu3(x_init + x)
        else:
            if self.in_channels < self.out_channels:
                x_init = torch.nn.functional.pad(x, (
                0, 0, 0, 0, 0, self.out_channels - self.in_channels), "constant", 0.0)
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
    def __init__(self, in_channels, out_channels, normalization_type=INSTANCE_NORM, downsample=False, blocks=3,
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
    def __init__(self, in_channels, hidden_channels, normalization_type=INSTANCE_NORM, pyr_height=5,
                 res_conv_blocks=[2, 6, 8, 23, 8], bottleneck_factor=1, squeeze_excitation=False,
                 hidden_channels_override: list[int]=None):
        super(ResNetBackbone, self).__init__()
        assert normalization_type in [INSTANCE_NORM, BATCH_NORM]

        if hidden_channels_override is None:
            hidden_channels_override = [hidden_channels * 2 ** i for i in range(pyr_height - 1)]

        self.pyr_height = pyr_height
        self.convs = torch.nn.ModuleList()

        if normalization_type == BATCH_NORM:
            self.initial_conv = torch.nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 7, 7),
                                                bias=False, padding="same", padding_mode="replicate")
            self.initial_batchnorm = torch.nn.BatchNorm3d(hidden_channels, momentum=BATCH_NORM_MOMENTUM)
            self.initial_elu = torch.nn.GELU()
        else:
            self.initial_conv = torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=7,
                                                bias=False, padding="same", padding_mode="replicate")
            self.initial_batchnorm = torch.nn.InstanceNorm2d(hidden_channels, affine=True)
            self.initial_elu = torch.nn.GELU()


        self.convs.append(ResConv(hidden_channels, hidden_channels, normalization_type=normalization_type,
                                      blocks=res_conv_blocks[0], bottleneck_factor=bottleneck_factor,
                                      squeeze_excitation=squeeze_excitation))
        for i in range(pyr_height - 1):
            self.convs.append(ResConv(hidden_channels_override[i], hidden_channels_override[i] * 2,
                        normalization_type=normalization_type, downsample=True, blocks=res_conv_blocks[i + 1],
                        bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation))

        self.normalization_type = normalization_type

    def wrap_to_2d(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        return x

    def unwrap_2d(self, x, batch_size):
        x = x.view(batch_size, -1, x.shape[1], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        return x

    def forward(self, x):
        batch_size = x.shape[0]
        depth = x.shape[2]

        if self.normalization_type == INSTANCE_NORM:
            x = self.wrap_to_2d(x)
        x = self.initial_conv(x)
        x = self.initial_batchnorm(x)
        x = self.initial_elu(x)

        # contracting path
        ret = []
        for i in range(self.pyr_height):
            x = self.convs[i](x)
            if self.normalization_type == BATCH_NORM:
                ret.append(x)
            else: # INSTANCE_NORM
                assert x.shape[0] == batch_size * depth
                ret.append(self.unwrap_2d(x, batch_size))

        return ret

class AveragePoolClassifierNeck(torch.nn.Module):
    def __init__(self, channels, out_classes={"label": 1}):
        super(AveragePoolClassifierNeck, self).__init__()

        self.channels = channels
        self.out_classes = out_classes
        self.num_outs = len(out_classes)
        self.out_id_to_key = []

        self.outconvs = torch.nn.ModuleList()
        for key in out_classes:
            self.outconvs.append(torch.nn.Conv3d(channels, out_classes[key], kernel_size=1, bias=True))
            self.out_id_to_key.append(key)

    def forward(self, x):
        x = x[-1]
        x = torch.mean(x, dim=(3, 4), keepdim=True)

        outs = {}
        for i in range(self.num_outs):
            outs[self.out_id_to_key[i]] = self.outconvs[i](x)

        return outs

class PatchAttnClassifierNeck(torch.nn.Module):
    def __init__(self, channels, out_classes={"label": 1}, key_dim=8, normalization_type=INSTANCE_NORM):
        super(PatchAttnClassifierNeck, self).__init__()

        self.channels = channels
        self.out_classes = out_classes
        self.num_outs = len(out_classes)
        self.out_id_to_key = []
        self.key_dim = key_dim

        self.query = torch.nn.Conv3d(channels, key_dim * self.num_outs, kernel_size=1, bias=False)

        self.spatch_pool = torch.nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1, padding=0)
        self.key = torch.nn.Conv3d(channels, key_dim * self.num_outs, kernel_size=1, bias=False)
        self.value = torch.nn.Conv3d(channels, channels * self.num_outs, kernel_size=1, bias=False)
        if normalization_type == BATCH_NORM:
            self.value_norm = torch.nn.BatchNorm3d(channels * self.num_outs, momentum=BATCH_NORM_MOMENTUM)
        else:
            self.value_norm = torch.nn.InstanceNorm2d(channels * self.num_outs, affine=True)

        self.nonlinearity = torch.nn.GELU()
        self.outconvs = torch.nn.ModuleList()
        for key in out_classes:
            self.outconvs.append(torch.nn.Conv3d(channels, out_classes[key], kernel_size=1, bias=True))
            self.out_id_to_key.append(key)

        self.temperature = torch.sqrt(torch.tensor(1.0 / key_dim))
        self.normalization_type = normalization_type

    def forward(self, x):
        x = x[-1]
        query = self.query(torch.mean(x, dim=(-1, -2), keepdim=True))
        assert query.shape == (x.shape[0], self.num_outs * self.key_dim, x.shape[2], 1, 1)
        N = x.shape[0]
        D = x.shape[2]

        x = self.spatch_pool(x)
        H = x.shape[3]
        W = x.shape[4]

        key = self.key(x)
        value = self.value(x)
        assert key.shape == (N, self.num_outs * self.key_dim, D, H, W)
        assert value.shape == (N, self.num_outs * self.channels, D, H, W)

        if self.normalization_type == BATCH_NORM:
            value = self.value_norm(value)
        else:
            value = value.permute(0, 2, 1, 3, 4)
            value = value.view(N * D, self.num_outs * self.channels, H, W)
            value = self.value_norm(value)
            value = value.view(N, D, self.num_outs * self.channels, H, W)
            value = value.permute(0, 2, 1, 3, 4)

        query = query.view(N, self.num_outs, self.key_dim, D, 1)
        key = key.view(N, self.num_outs, self.key_dim, D, H * W)
        value = value.view(N, self.num_outs, self.channels, D, H * W)

        dot = torch.sum(query * key, dim=2, keepdim=True) * self.temperature
        attn = torch.softmax(dot, dim=-1)
        out = self.nonlinearity(torch.sum(value * attn, dim=-1))
        out = out.view(N, self.num_outs, self.channels, D, 1, 1)

        outs = {}
        for i in range(self.num_outs):
            outs[self.out_id_to_key[i]] = self.outconvs[i](out[:, i, ...])

        # output shape: (N, self.num_outs, D, H, W), [(N, sum(self.out_classes), D, 1, 1)]
        return attn.view(N, self.num_outs, D, H, W), outs

class MeanProbaReductionHead(torch.nn.Module):
    def __init__(self, channels, out_classes={"label": 1}):
        super(MeanProbaReductionHead, self).__init__()

        self.channels = channels
        self.out_classes = out_classes
        self.out_id_to_key = []
        self.num_outs = len(out_classes)

        for key in out_classes:
            self.out_id_to_key.append(key)

    def forward(self, outs):
        probas = {}
        for k in range(self.num_outs):
            key = self.out_id_to_key[k]
            out = outs[key]
            out = out.squeeze(-1).squeeze(-1)
            if self.out_classes[key] == 1:
                out = torch.sigmoid(out).squeeze(1)
            else:
                out = torch.softmax(out, dim=1)
            probas[key] = torch.mean(out, dim=-1)
        return probas

class UnionProbaReductionHead(torch.nn.Module):
    def __init__(self, channels, out_classes={"label": 1}):
        super(UnionProbaReductionHead, self).__init__()

        for k in out_classes:
            assert out_classes[k] == 1, "UnionProbaReductionHead only supports binary classification"

        self.channels = channels
        self.out_classes = out_classes
        self.out_id_to_key = []
        self.num_outs = len(out_classes)

        for key in out_classes:
            self.out_id_to_key.append(key)

    def forward(self, outs):
        probas = {}
        for k in range(self.num_outs):
            key = self.out_id_to_key[k]
            out = outs[key]
            out = out.squeeze(-1).squeeze(-1).squeeze(1)
            out = torch.sigmoid(out)
            probas[key] = 1 - torch.prod(1 - out, dim=-1)
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
