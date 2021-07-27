import torch
import torch.nn as nn

from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self):
        vgg = vgg19(pretrained = True)
        # Before activation (ReLU)
        self.vgg19_54 = nn.Sequential(*list(vgg.features.children())[: 35])

    def forward(self, img: torch.Tensor):
        return self.vgg19_54(img)


class DenseResidualBlock(nn.Module):
    """
    1) Remove all BN layer: increase performance and reduce computational complexity
    2) Dense block
    3) Residual block
    4) Residual scaling to prevent instability

    Inputs:
        - `img_tensor`: [batch, filters, height, width]
    Returns:
        - `feature map`: [batch, filters, height, width]
    """
    def __init__(self, filters: int, residual_scale: float = 0.2):
        super(DenseResidualBlock, self).__init__()
        self.residual_scale = residual_scale

        def block(in_features, non_linearity = True):
            layer = [nn.Conv2d(in_features, filters, 3, 1, 1, bias = True)]
            if non_linearity:
                layer.append(nn.ReLU(inplace = True))
            return nn.Sequential(*layer)

        self.block1 = block(filters)
        self.block2 = block(filters * 2)
        self.block3 = block(filters * 3)
        self.block4 = block(filters * 4)
        self.block5 = block(filters * 5, False)
        self.blocks = [self.block1, self.block2, self.block3, self.block4, self.block5]

    def forward(self, x: torch.Tensor):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out * self.residual_scale + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters: int, residual_scale: float = 0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.residual_scale = residual_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters, self.residual_scale),
            DenseResidualBlock(filters, self.residual_scale),
            DenseResidualBlock(filters, self.residual_scale)
        )

    def forward(self, x: torch.Tensor):
        return self.dense_blocks(x) * self.residual_scale + x


class Generator(nn.Module):
    """[Generator in ESRGAN]
    Arguments:
        in_channels {int} -- [Input channels of tensor]
        filters {int} -- [Number of feature maps]
        num_blocks {int} -- [Number of residual blocks in DenseResidualBlock]
        num_upsamples {int} -- [Number of upsampling layers (ratio = 2)]

    Inputs:
        - `img_tensor`: [batch, channels, height, width]
    Returns:
        - `out`: [batch, channels, height * 2^num_upsamples, width * 2^num_upsamples]
    """
    def __init__(self,
        in_channels: int = 3,
        filters: int = 32,
        num_blocks: int = 16,
        num_upsamples: int = 2,
        residual_scale: float = 0.2
    ):
        super().__init__()
        # First Convolution Layer
        self.first_conv = nn.Conv2d(in_channels, filters, 3, 1, 1)
        # Residual block
        self.residual_block = nn.Sequential(
            *[ResidualInResidualDenseBlock(filters, residual_scale) for _ in range(num_blocks)]
        )
        # Second Convolution Layer
        self.second_conv = nn.Conv2d(filters, filters, 3, 1, 1)
        # Upsampling Layers
        upsample_layer = []
        for _ in range(num_upsamples):
            upsample_layer.append(nn.Conv2d(filters, filters * 4, 3, 1, 1))
            upsample_layer.append(nn.LeakyReLU(0.2, inplace = True))
            upsample_layer.append(nn.PixelShuffle(2))
        self.upsample_layer = nn.Sequential(*upsample_layer)
        # Last Convolution Layer
        self.last_conv = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(filters, in_channels, 3, 1, 1)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std = 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        feature1 = self.first_conv(x)
        feature = self.residual_block(feature1)
        feature2 = self.second_conv(feature)
        feature = feature1 + feature2
        out = self.upsample_layer(feature)
        out = self.last_conv(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape: tuple = (3, 96, 96)):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        in_channel, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)

        def block(in_filters, out_filters, first_block = False):
            layer = []
            layer.append(nn.Conv2d(in_filters, out_filters, 3, 1, 1))
            if not first_block:
                layer.append(nn.BatchNorm2d(out_filters))
            layer.append(nn.LeakyReLU(0.2, inplace = True))
            layer.append(nn.Conv2d(out_filters, out_filters, 3, 2, 1))
            layer.append(nn.BatchNorm2d(out_filters))
            layer.append(nn.LeakyReLU(0.2, inplace = True))
            return layer

        layers = []
        in_filters = in_channel
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(block(in_filters, out_filters, first_block = i == 0))
            in_filters = out_filters
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


def test():
    netG = Generator(num_blocks=1)
    netD = Discriminator((3, 896, 896))
    img = torch.randn(1, 3, 224, 224)
    fake_img = netG(img)
    print(fake_img.shape)
    print(netD(fake_img).shape)
