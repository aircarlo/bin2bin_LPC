import torch
import torch.nn as nn


class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        if down:
            self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1,
                                       groups=in_channels, bias=False, padding_mode="reflect")
            self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.depthwise = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2,
                                                padding=1, groups=in_channels, bias=False)
            self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return self.dropout(x) if self.use_dropout else x


class DSCNN_Generator(nn.Module):
    """
    Depthwise-separable CNN U-NET
    Torchinfo out:
        Total params: 1,152,449
        Trainable params: 1,152,449
        Non-trainable params: 0
        Total mult-adds (G): 1.10
        Input size (MB): 0.26
        Forward/backward pass size (MB): 112.72
        Params size (MB): 4.61
        Estimated Total Size (MB): 117.59
    """
    def __init__(self, in_channels=1, features=64):
        super().__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = DSConvBlock(features, features * 2, down=True, act="leaky")
        self.down2 = DSConvBlock(features * 2, features * 4, down=True, act="leaky")
        self.down3 = DSConvBlock(features * 4, features * 4, down=True, act="leaky")
        self.down4 = DSConvBlock(features * 4, features * 4, down=True, act="leaky")

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 4, features * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.up2 = DSConvBlock(features * 4 * 2, features * 4, down=False, act="relu")
        self.up3 = DSConvBlock(features * 4 * 2, features * 4, down=False, act="relu")
        self.up4 = DSConvBlock(features * 4 * 2, features * 2, down=False, act="relu")
        self.up5 = DSConvBlock(features * 2 * 2, features, down=False, act="relu")

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)

        bottleneck = self.bottleneck(d5)

        up2 = self.up2(torch.cat([bottleneck, d5], dim=1))
        up3 = self.up3(torch.cat([up2, d4], dim=1))
        up4 = self.up4(torch.cat([up3, d3], dim=1))
        up5 = self.up5(torch.cat([up4, d2], dim=1))

        return self.final_up(torch.cat([up5, d1], dim=1))


if __name__ == "__main__":
    from torchinfo import summary
    model = DSCNN_Generator(in_channels=1)
    summary(model, input_size=(1, 1, 256, 256))