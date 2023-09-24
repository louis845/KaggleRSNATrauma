import torch
import model_3d_patch_resnet
import math

class ResConv3DAttnBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False,
                 bottleneck_factor=1, attention_dim=32, depth=9,
                 pos_embeddings_input=False,
                 single_image=False, mix_images=False,
                 middle_layer_norm=False):
        """
        Only applies 2D convolutions to a 3D block
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param downsample: whether to downsample the input 2x2
        :param bottleneck_factor: how much to expand the number of channels in the bottleneck
        :param attention_dim: how many channels to use for the attention
        :param depth: the expected depth of the 3D volume
        :param pos_embeddings_input: whether positional embeddings are given to the input, or learned
        :param single_image: whether the input is a single image or two images. If true, mix_images is ignored.
        :param mix_images: by default, two images are mixed together in the input (concat by depth). If this is set to
            True, the information are mixed together with attention. If False, the information won't be mixed at all.
        """
        super(ResConv3DAttnBlock, self).__init__()
        assert in_channels <= out_channels
        if bottleneck_factor > 1:
            assert out_channels % bottleneck_factor == 0, "out_channels must be divisible by bottleneck_factor"
            assert out_channels % (bottleneck_factor * 4) == 0, "out_channels must be divisible by bottleneck_factor * 4"

        bottleneck_channels = out_channels // bottleneck_factor
        if downsample:
            self.avgpool = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

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
        if middle_layer_norm:
            self.batchnorm2 = torch.nn.GroupNorm(num_groups=1, num_channels=bottleneck_channels, affine=True)
        else:
            self.batchnorm2 = torch.nn.InstanceNorm3d(bottleneck_channels, affine=True)
        self.nonlin2 = torch.nn.GELU()

        self.projQ = torch.nn.Conv3d(bottleneck_channels, attention_dim, 1, bias=False, padding="same",
                                     padding_mode="replicate")
        self.projK = torch.nn.Conv3d(bottleneck_channels, attention_dim, 1, bias=False, padding="same",
                                     padding_mode="replicate")
        self.projV = torch.nn.Conv3d(bottleneck_channels, out_channels, 1, bias=False, padding="same",
                                     padding_mode="replicate")
        if pos_embeddings_input:
            self.projQ_embedding = torch.nn.Conv3d(attention_dim, attention_dim, 1, bias=False, padding="same")
            self.projK_embedding = torch.nn.Conv3d(attention_dim, attention_dim, 1, bias=False, padding="same")
        else:
            if single_image:
                self.Q_embedding = torch.nn.Parameter((torch.rand(size=(1, attention_dim, depth, 1, 1)) - 0.5) / 2)
                self.K_embedding = torch.nn.Parameter((torch.rand(size=(1, attention_dim, depth, 1, 1)) - 0.5) / 2)
            else:
                if mix_images:
                    # mix the images, the positional embeddings should be same for each image
                    self.Q_embedding = torch.nn.Parameter(torch.tile(
                        (torch.rand(size=(1, attention_dim, depth, 1, 1)) - 0.5) / 2, (1, 1, 2, 1, 1)))
                    self.K_embedding = torch.nn.Parameter(torch.tile(
                        (torch.rand(size=(1, attention_dim, depth, 1, 1)) - 0.5) / 2, (1, 1, 2, 1, 1)))
                else:
                    # positional embeddings should be same for each image (for the same position)
                    self.Q_embedding = torch.nn.Parameter((torch.rand(size=(1, attention_dim, 1, depth, 1, 1)) - 0.5) / 2)
                    self.K_embedding = torch.nn.Parameter((torch.rand(size=(1, attention_dim, 1, depth, 1, 1)) - 0.5) / 2)


        self.batchnormV = torch.nn.InstanceNorm3d(out_channels, affine=True)
        self.nonlinOut = torch.nn.GELU()

        self.depth = depth
        self.pos_embeddings_input = pos_embeddings_input
        self.single_image = single_image
        self.mix_images = mix_images
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention_dim = attention_dim
        self.downsample = downsample
        self.bottleneck_factor = bottleneck_factor

    def forward(self, x):
        if self.pos_embeddings_input:
            x, pos_embeddings = x # extract the input and given position embeddings
            if self.single_image:
                assert pos_embeddings.shape == (1, self.attention_dim, self.depth, 1, 1)
            else:
                assert pos_embeddings.shape == (1, self.attention_dim, self.depth * 2, 1, 1)
        N, C, D, H, W = x.shape
        if self.single_image:
            assert D == self.depth, "input depth must be the expected depth"
        else:
            assert D == self.depth * 2, "input depth must be twice the expected depth (for two images)"

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
            assert x.shape == (N, self.out_channels // self.bottleneck_factor, D, H // 2, W // 2)
        else:
            x = self.conv2(x)
            assert x.shape == (N, self.out_channels // self.bottleneck_factor, D, H, W)
        x = self.batchnorm2(x)
        x = self.nonlin2(x)
        N, _, _, H, W = x.shape

        # compute Q, K, V
        V = self.projV(x)
        V = self.batchnormV(V)

        if self.pos_embeddings_input:
            Q = self.projQ(x) + self.projQ_embedding(pos_embeddings)
            K = self.projK(x) + self.projK_embedding(pos_embeddings)
        else:
            if self.single_image or self.mix_images:
                Q = self.projQ(x) + self.Q_embedding
                K = self.projK(x) + self.K_embedding
            else:
                Q = self.projQ(x).view(N, self.attention_dim, 2, self.depth, H, W) + self.Q_embedding
                K = self.projK(x).view(N, self.attention_dim, 2, self.depth, H, W) + self.K_embedding
                V = V.view(N, self.out_channels, 2, self.depth, H, W)
        # depthwise and imagewise attention
        Q = Q.unsqueeze(-3)
        assert (Q.shape[-4:] == (self.depth, 1, H, W)) or (Q.shape[-4:] == (self.depth * 2, 1, H, W))
        K = K.unsqueeze(-4)
        assert (K.shape[-4:] == (1, self.depth, H, W)) or (K.shape[-4:] == (1, self.depth * 2, H, W))
        V = V.unsqueeze(-4)
        assert (V.shape[-4:] == (1, self.depth, H, W)) or (V.shape[-4:] == (1, self.depth * 2, H, W))
        attn = torch.softmax((Q * K).sum(dim=1, keepdim=True) / math.sqrt(self.attention_dim), dim=-3)
        x = (attn * V).sum(dim=-3)

        if self.single_image:
            assert x.shape == (N, self.out_channels, self.depth, H, W)
        else:
            if self.mix_images:
                assert x.shape == (N, self.out_channels, self.depth * 2, H, W)
            else:
                assert x.shape == (N, self.out_channels, 2, self.depth, H, W)
                x = x.view(N, self.out_channels, self.depth * 2, H, W)

        if self.downsample:
            x_init = self.avgpool(x_init)
        result = self.nonlinOut(x_init + x)
        return result

class ResConv3DAttn(torch.nn.Module):
    def __init__(self, blocks: int, num_2d_blocks: int, in_channels: int, out_channels: int, downsample=False, bottleneck_factor: int=1,
                 attention_dim=32, depth=9, pos_embeddings_input=False, single_image=False, mix_images=False):
        super(ResConv3DAttn, self).__init__()
        assert num_2d_blocks <= blocks, "num_2d_blocks must be smaller or equal to blocks"
        assert num_2d_blocks >= 0, "num_2d_blocks must be positive"
        assert in_channels <= out_channels, "in_channels must be smaller or equal to out_channels"

        self.num_2d_blocks = num_2d_blocks

        self.conv_res = torch.nn.ModuleList()
        if num_2d_blocks > 0:
            self.conv_res.append(model_3d_patch_resnet.ResConv2DBlock(in_channels, out_channels,
                                                                      downsample=downsample, bottleneck_factor=bottleneck_factor,
                                                                      squeeze_excitation=True,
                                                                      squeeze_excitation_bottleneck_factor=2))
            num_2d_blocks -= 1
        else:
            self.conv_res.append(ResConv3DAttnBlock(in_channels, out_channels, downsample=downsample,
                                                    bottleneck_factor=bottleneck_factor, attention_dim=attention_dim,
                                                    depth=depth, pos_embeddings_input=pos_embeddings_input,
                                                    single_image=single_image, mix_images=mix_images))
        for k in range(blocks - 1):
            if num_2d_blocks > 0:
                self.conv_res.append(model_3d_patch_resnet.ResConv2DBlock(out_channels, out_channels,
                                                                          downsample=False, bottleneck_factor=bottleneck_factor,
                                                                          squeeze_excitation=True,
                                                                          squeeze_excitation_bottleneck_factor=2))
                num_2d_blocks -= 1
            else:
                self.conv_res.append(ResConv3DAttnBlock(out_channels, out_channels, downsample=False,
                                                        bottleneck_factor=bottleneck_factor, attention_dim=attention_dim,
                                                        depth=depth, pos_embeddings_input=pos_embeddings_input,
                                                        single_image=single_image, mix_images=mix_images))
        self.blocks = blocks
        self.pos_embeddings_input = pos_embeddings_input

    def forward(self, x):
        if self.pos_embeddings_input:
            x, pos_embeddings = x # extract the input and given position embeddings
            num_2d_blocks = self.num_2d_blocks
            for k in range(self.blocks):
                if num_2d_blocks > 0:
                    x = self.conv_res[k](x)
                    num_2d_blocks -= 1
                else:
                    x = self.conv_res[k](x, pos_embeddings)
        else:
            for k in range(self.blocks):
                x = self.conv_res[k](x)
        return x

class TotalAttnBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, height, width, downsample=False,
                 bottleneck_factor=1, attention_dim=32, depth=9):
        """
        Applies both spatial and depthwise attention
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param height: height of the input (AFTER downsampling 2x2 if necessary)
        :param width: width of the input (AFTER downsampling 2x2 if necessary)
        :param downsample: whether to downsample the input 2x2
        :param attention_dim: dimension of the attention
        :param depth: depth of the input
        """
        super(TotalAttnBlock, self).__init__()
        assert in_channels <= out_channels
        assert out_channels % bottleneck_factor == 0, "out_channels must be divisible by bottleneck_factor"
        bottleneck_channels = out_channels // bottleneck_factor

        if downsample:
            self.avgpool = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        self.projQ1 = torch.nn.Conv3d(in_channels, attention_dim, kernel_size=1, bias=False)
        self.Q1_embedding = torch.nn.Parameter((torch.rand(1, attention_dim, depth, 1, 1) - 0.5) / 10)
        self.projK1 = torch.nn.Conv3d(in_channels, attention_dim, kernel_size=1, bias=False)
        self.K1_embedding = torch.nn.Parameter((torch.rand(1, attention_dim, depth, 1, 1) - 0.5) / 10)
        self.projV1 = torch.nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.batchnorm1 = torch.nn.GroupNorm(num_groups=1, num_channels=bottleneck_channels, affine=True)
        self.nonlin1 = torch.nn.GELU()

        self.projQ2 = torch.nn.Conv3d(bottleneck_channels, attention_dim * 2, kernel_size=1, bias=False)
        self.Q2_embedding = torch.nn.Parameter((torch.rand(1, attention_dim * 2, 1, height, width) - 0.5) / 10)
        self.projK2 = torch.nn.Conv3d(bottleneck_channels, attention_dim * 2, kernel_size=1, bias=False)
        self.K2_embedding = torch.nn.Parameter((torch.rand(1, attention_dim * 2, 1, height, width) - 0.5) / 10)
        self.projV2 = torch.nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.batchnorm2 = torch.nn.GroupNorm(num_groups=1, num_channels=bottleneck_channels, affine=True)
        self.nonlin2 = torch.nn.GELU()

        self.conv3 = torch.nn.Conv3d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.batchnorm3 = torch.nn.InstanceNorm3d(num_features=out_channels, affine=True)
        self.nonlin3 = torch.nn.GELU()

        self.downsample = downsample
        self.depth = depth
        self.height = height
        self.width = width
        self.attention_dim = attention_dim
        self.bottleneck_channels = bottleneck_channels

    def forward(self, x):
        N, C, D, H, W = x.shape
        assert D == self.depth, "input depth must be the expected depth"
        if self.downsample:
            assert H % 2 == 0 and W % 2 == 0, "input height and width must be divisible by 2 if downsample is True"

        if self.in_channels < self.out_channels:
            x_init = torch.nn.functional.pad(x, (
            0, 0, 0, 0, 0, 0, 0, self.out_channels - self.in_channels), "constant", 0.0)
        else:
            x_init = x

        # compute Q, K, V
        Q1 = self.projQ1(x) + self.Q1_embedding
        K1 = self.projK1(x) + self.K1_embedding
        V1 = self.projV1(x)
        V1 = self.batchnorm1(V1)
        assert V1.shape == (N, self.bottleneck_channels, D, H, W)

        # depthwise attention
        Q1 = Q1.unsqueeze(-3)
        assert Q1.shape[-4:] == (self.depth, 1, H, W)
        K1 = K1.unsqueeze(-4)
        assert K1.shape[-4:] == (1, self.depth, H, W)
        V1 = V1.unsqueeze(-4)
        assert V1.shape[-4:] == (1, self.depth, H, W)
        attn = torch.softmax((Q1 * K1).sum(dim=1, keepdim=True) / math.sqrt(self.attention_dim), dim=-3)
        x = (attn * V1).sum(dim=-3)
        assert x.shape == (N, self.bottleneck_channels, D, H, W)
        x = self.nonlin1(x)

        # downsample
        if self.downsample:
            x = self.avgpool(x)
            H = H // 2
            W = W // 2
        assert H == self.height, "input height must be the expected height (after downsampling)"
        assert W == self.width, "input width must be the expected width (after downsampling)"

        # compute Q, K, V
        Q2 = self.projQ2(x) + self.Q2_embedding
        K2 = self.projK2(x) + self.K2_embedding
        V2 = self.projV2(x)
        V2 = self.batchnorm2(V2)
        assert Q2.shape == (N, self.attention_dim * 2, D, H, W)
        assert K2.shape == (N, self.attention_dim * 2, D, H, W)
        assert V2.shape == (N, self.bottleneck_channels, D, H, W)

        # spatial attention
        Q2 = Q2.view(N, self.attention_dim * 2, D, H * W).unsqueeze(-1)
        assert Q2.shape == (N, self.attention_dim * 2, D, H * W, 1)
        K2 = K2.view(N, self.attention_dim * 2, D, H * W).unsqueeze(-2)
        assert K2.shape == (N, self.attention_dim * 2, D, 1, H * W)
        V2 = V2.view(N, self.bottleneck_channels, D, H * W).unsqueeze(-2)
        assert V2.shape == (N, self.bottleneck_channels, D, 1, H * W)
        attn = torch.softmax((Q2 * K2).sum(dim=1, keepdim=True) / math.sqrt(self.attention_dim * 2), dim=-1)
        x = (attn * V2).sum(dim=-1)
        assert x.shape == (N, self.bottleneck_channels, D, H * W, 1)
        x = x.view(N, self.bottleneck_channels, D, H, W)
        x = self.nonlin2(x)

        # 1x1x1 convolution
        x = self.conv3(x)
        x = self.batchnorm3(x)

        if self.downsample:
            x_init = self.avgpool(x_init)
        result = self.nonlin3(x_init + x)
        return result

class TotalAttn(torch.nn.Module):
    def __init__(self, blocks: int, in_channels: int, out_channels: int, height, width, depth=9, downsample=False,
                        bottleneck_factor=1):
        super(TotalAttn, self).__init__()
        assert in_channels <= out_channels, "in_channels must be smaller or equal to out_channels"

        self.height = height
        self.width = width
        if downsample:
            assert height % 2 == 0 and width % 2 == 0, "height and width must be divisible by 2 if downsample is True"
            self.downsampled_height = height // 2
            self.downsampled_width = width // 2
        else:
            self.downsampled_height = height
            self.downsampled_width = width

        self.conv_res = torch.nn.ModuleList()
        self.conv_res.append(TotalAttnBlock(in_channels, out_channels, self.downsampled_height, self.downsampled_width, downsample,
                                            bottleneck_factor=bottleneck_factor, attention_dim=32, depth=depth))
        for k in range(blocks):
            self.conv_res.append(TotalAttnBlock(in_channels, out_channels, self.downsampled_height, self.downsampled_width, False,
                                            bottleneck_factor=bottleneck_factor, attention_dim=32, depth=depth))
        self.blocks = blocks

    def forward(self, x):
        assert x.shape[-3:] == (self.height, self.width, self.depth)
        for k in range(self.blocks):
            x = self.conv_res[k](x)
        return x

class ResConv2D(torch.nn.Module):
    def __init__(self, blocks: int, in_channels: int, out_channels: int, downsample=False):
        super(ResConv2D, self).__init__()

        assert in_channels <= out_channels, "in_channels must be smaller or equal to out_channels"

        self.conv_res = torch.nn.ModuleList()
        self.conv_res.append(model_3d_patch_resnet.ResConv2DBlock(in_channels, out_channels,
                                                                  downsample=downsample, bottleneck_factor=1,
                                                                  squeeze_excitation=True,
                                                                  squeeze_excitation_bottleneck_factor=2))
        for k in range(blocks):
            self.conv_res.append(model_3d_patch_resnet.ResConv2DBlock(out_channels, out_channels,
                                                    downsample=False, bottleneck_factor=1,
                                                    squeeze_excitation=True, squeeze_excitation_bottleneck_factor=2))
        self.blocks = blocks

    def forward(self, x):
        for k in range(self.blocks):
            x = self.conv_res[k](x)
        return x

class ResNet3DClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, out_classes: int, channel_progression: list[int]=[4, 8, 16, 32, 64, 128],
                 conv3d_blocks=[0, 0, 0, 1, 1, 2], res_conv_blocks=[1, 2, 6, 8, 23, 8], bottleneck_factor=1,
                 input_depth=9, input_single_image=False, pos_embeddings_input=False,
                 initial_downsampling=False):
        super(ResNet3DClassifier, self).__init__()

        assert len(res_conv_blocks) == len(conv3d_blocks), "res_conv_blocks and res_conv3d_blocks must have the same length"
        assert len(channel_progression) == len(res_conv_blocks), "channel_progression and res_conv_blocks must have the same length"
        for k in conv3d_blocks:
            assert k in [0, 1, 2], "conv3d_blocks must only contain 0, 1 or 2"
        self.convs = torch.nn.ModuleList()

        if initial_downsampling:
            self.initial_conv = torch.nn.Conv3d(in_channels, channel_progression[0], kernel_size=(1, 10, 10),
                                                stride=(1, 2, 2), bias=False, padding=(0, 4, 4), padding_mode="replicate")
        else:
            self.initial_conv = torch.nn.Conv3d(in_channels, channel_progression[0], kernel_size=(1, 7, 7),
                                                bias=False, padding="same", padding_mode="replicate")
        self.initial_batchnorm = torch.nn.InstanceNorm3d(channel_progression[0], affine=True)
        self.initial_nonlin = torch.nn.GELU()

        prev_channels = channel_progression[0]
        for i in range(len(res_conv_blocks)):
            blocks = res_conv_blocks[i]
            if conv3d_blocks[i] == 0:
                self.convs.append(ResConv2D(blocks, prev_channels, channel_progression[i], downsample=(i > 0)))
            elif conv3d_blocks[i] == 1:
                self.convs.append(ResConv3DAttn(blocks, 0, prev_channels, channel_progression[i], downsample=(i > 0),
                                                bottleneck_factor=1, attention_dim=32,
                                                depth=input_depth, pos_embeddings_input=pos_embeddings_input,
                                                single_image=input_single_image, mix_images=True))
            elif conv3d_blocks[i] == 2:
                self.convs.append(ResConv3DAttn(blocks, 0, prev_channels, channel_progression[i], downsample=(i > 0),
                                                bottleneck_factor=bottleneck_factor, attention_dim=32,
                                                depth=input_depth, pos_embeddings_input=pos_embeddings_input,
                                                single_image=input_single_image, mix_images=True))
            prev_channels = channel_progression[i]

        self.pyr_height = len(res_conv_blocks)
        self.outpool = torch.nn.AdaptiveAvgPool2d(1)
        if input_single_image:
            self.outconv = torch.nn.Conv2d(channel_progression[-1] * input_depth,
                                           out_classes, kernel_size=1, bias=True)
        else:
            self.outconv = torch.nn.Conv2d(channel_progression[-1] * input_depth * 2,
                                           out_classes, kernel_size=1, bias=True)

        self.pos_embeddings_input = pos_embeddings_input
        self.conv3d_blocks = conv3d_blocks
        self.input_depth = input_depth
        self.last_channel = channel_progression[-1]
        self.input_single_image = input_single_image

    def forward(self, x):
        if self.pos_embeddings_input:
            x, pos_embeddings = x # extract the input and given position embeddings
        if self.input_single_image:
            assert x.shape[2] == self.input_depth
        else:
            assert x.shape[2] == self.input_depth * 2
        N = x.shape[0]
        x = self.initial_conv(x)
        x = self.initial_batchnorm(x)
        x = self.initial_nonlin(x)

        for i in range(self.pyr_height):
            if (self.conv3d_blocks[i] > 0) and self.pos_embeddings_input:
                x = self.convs[i](x, pos_embeddings)
            else:
                x = self.convs[i](x)

        if self.input_single_image:
            assert x.shape[:3] == (N, self.last_channel, self.input_depth)
        else:
            assert x.shape[:3] == (N, self.last_channel, self.input_depth * 2)
        H, W = x.shape[-2:]
        if self.input_single_image:
            x = self.outpool(x.view(N, self.last_channel * self.input_depth, H, W))
        else:
            x = self.outpool(x.view(N, self.last_channel * self.input_depth * 2, H, W))
        x = self.outconv(x).view(N, -1)
        return x

class ResNetTotalAttn3DClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, out_classes: int, input_height: int, input_width: int,
                 channel_progression: list[int]=[4, 8, 16, 32, 64, 128], conv3d_blocks=[0, 0, 0, 1, 1, 2],
                 res_conv_blocks=[1, 2, 6, 8, 23, 8], bottleneck_factor=1, input_depth=9):
        super(ResNetTotalAttn3DClassifier, self).__init__()

        assert len(res_conv_blocks) == len(conv3d_blocks), "res_conv_blocks and res_conv3d_blocks must have the same length"
        assert len(channel_progression) == len(res_conv_blocks), "channel_progression and res_conv_blocks must have the same length"
        for k in conv3d_blocks:
            assert k in [0, 1, 2], "conv3d_blocks must only contain 0, 1 or 2"
        self.convs = torch.nn.ModuleList()

        self.input_height = input_height
        self.input_width = input_width
        assert input_height % 2 == 0, "input_height must be even"
        assert input_width % 2 == 0, "input_width must be even"

        self.initial_conv = torch.nn.Conv3d(in_channels, channel_progression[0], kernel_size=(1, 10, 10),
                                            stride=(1, 2, 2), bias=False, padding=(0, 4, 4), padding_mode="replicate") # 2x2 downsample
        self.initial_batchnorm = torch.nn.InstanceNorm3d(channel_progression[0], affine=True)
        self.initial_nonlin = torch.nn.GELU()

        current_height = input_height // 2
        current_width = input_width // 2

        prev_channels = channel_progression[0]
        for i in range(len(res_conv_blocks)):
            current_is_downsampling = (i > 0)
            blocks = res_conv_blocks[i]
            if conv3d_blocks[i] == 0:
                self.convs.append(ResConv2D(blocks, prev_channels, channel_progression[i], downsample=current_is_downsampling))
            elif conv3d_blocks[i] == 1:
                assert blocks >= 4, "ResConv3DAttn (1) requires at least 4 blocks"
                num_2d_blocks = blocks - 3 # only 3 depthwise attention blocks
                self.convs.append(ResConv3DAttn(blocks, num_2d_blocks, prev_channels, channel_progression[i], downsample=current_is_downsampling,
                                                bottleneck_factor=1, attention_dim=32,
                                                depth=input_depth, pos_embeddings_input=False,
                                                single_image=True, mix_images=False))
            elif conv3d_blocks[i] == 2:
                self.convs.append(TotalAttn(blocks, prev_channels, channel_progression[i],
                                            height=current_height, width=current_width, depth=input_depth,
                                            downsample=current_is_downsampling, bottleneck_factor=bottleneck_factor))
            prev_channels = channel_progression[i]

            if current_is_downsampling:
                assert current_width % 2 == 0, "current_width must be even"
                assert current_height % 2 == 0, "current_height must be even"
                current_width = current_width // 2
                current_height = current_height // 2

        self.roi_outconv = torch.nn.Conv3d(channel_progression[4], 1, kernel_size=1, bias=True)

        self.pyr_height = len(res_conv_blocks)
        self.outpool = torch.nn.AdaptiveAvgPool3d(1)
        self.outconv = torch.nn.Conv3d(channel_progression[-1], out_classes, kernel_size=1, bias=True)

        self.conv3d_blocks = conv3d_blocks
        self.input_depth = input_depth
        self.last_channel = channel_progression[-1]

    def forward(self, x):
        assert x.shape[-3:] == (self.input_depth, self.input_height, self.input_width)
        N = x.shape[0]
        x = self.initial_conv(x)
        x = self.initial_batchnorm(x)
        x = self.initial_nonlin(x)

        deep_roi_out = None
        for i in range(self.pyr_height):
            x = self.convs[i](x)
            if i == 4:
                deep_roi_out = self.roi_outconv(x)

        assert x.shape[:3] == (N, self.last_channel, self.input_depth)
        x = self.outpool(x)
        x = self.outconv(x).view(N, -1)
        return x, deep_roi_out
