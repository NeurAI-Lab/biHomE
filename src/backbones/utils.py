import torch.nn as nn


class ResNet50ConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride):
        super(ResNet50ConvBlock, self).__init__()

        mid_channels = input_channels//stride
        self.upper_branch = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=mid_channels,
                                                    kernel_size=1, padding=0, stride=stride, bias=False),
                                          nn.BatchNorm2d(mid_channels),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                                                    kernel_size=3, padding=1, stride=1, bias=False),
                                          nn.BatchNorm2d(mid_channels),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=mid_channels, out_channels=output_channels,
                                                    kernel_size=1, padding=0, stride=1, bias=False),
                                          nn.BatchNorm2d(output_channels))

        self.lower_branch = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                                    kernel_size=1, padding=0, stride=stride, bias=False),
                                          nn.BatchNorm2d(output_channels))

    def forward(self, x):
        upper = self.upper_branch(x)
        lower = self.lower_branch(x)
        return nn.ReLU()(upper+lower)


class ResNet50IdentityBlock(nn.Module):

    def __init__(self, input_channels):
        super(ResNet50IdentityBlock, self).__init__()

        self.upper_branch = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=input_channels//4,
                                                    kernel_size=1, padding=0, stride=1, bias=False),
                                          nn.BatchNorm2d(input_channels//4),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=input_channels//4, out_channels=input_channels//4,
                                                    kernel_size=3, padding=1, stride=1, bias=False),
                                          nn.BatchNorm2d(input_channels//4),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=input_channels//4, out_channels=input_channels,
                                                    kernel_size=1, padding=0, stride=1, bias=False),
                                          nn.BatchNorm2d(input_channels))

        # self.lower_branch = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
        #                                             kernel_size=1, padding=0, stride=1, bias=False),
        #                                   nn.BatchNorm2d(input_channels))

    def forward(self, x):
        upper = self.upper_branch(x)
        #lower = self.lower_branch(x)
        lower = x
        return nn.ReLU()(upper+lower)


class ResNet50DeconvBlock(nn.Module):

    def __init__(self, input_channels):
        super(ResNet50DeconvBlock, self).__init__()

        self.upper_branch = nn.Sequential(nn.ConvTranspose2d(in_channels=input_channels, out_channels=input_channels,
                                                             kernel_size=2, padding=0, stride=2),
                                          nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
                                                    kernel_size=3, padding=1, stride=1, bias=False),
                                          nn.BatchNorm2d(input_channels),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=input_channels, out_channels=input_channels//2,
                                                    kernel_size=1, padding=0, stride=1, bias=False),
                                          nn.BatchNorm2d(input_channels//2))

        self.lower_branch = nn.Sequential(nn.ConvTranspose2d(in_channels=input_channels, out_channels=input_channels//2,
                                                             kernel_size=2, padding=0, stride=2, bias=False),
                                          nn.BatchNorm2d(input_channels//2))

    def forward(self, x):
        upper = self.upper_branch(x)
        lower = self.lower_branch(x)
        return nn.ReLU()(upper+lower)


class ResNet34ConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride):
        super(ResNet34ConvBlock, self).__init__()

        self.upper_branch = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                                    kernel_size=3, padding=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(output_channels),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=output_channels, out_channels=output_channels,
                                                    kernel_size=3, padding=1, stride=1, bias=False),
                                          nn.BatchNorm2d(output_channels))

        self.lower_is_identity = True
        if input_channels != output_channels:

            self.lower_branch = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                                        kernel_size=1, padding=0, stride=stride, bias=False),
                                              nn.BatchNorm2d(output_channels))
            self.lower_is_identity = False

    def forward(self, x):
        upper = self.upper_branch(x)
        if self.lower_is_identity:
            lower = x
        else:
            lower = self.lower_branch(x)
        return nn.ReLU()(upper+lower)


class ResNet34IdentityBlock(nn.Module):

    def __init__(self, input_channels):
        super(ResNet34IdentityBlock, self).__init__()

        self.upper_branch = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
                                                    kernel_size=3, padding=1, stride=1, bias=False),
                                          nn.BatchNorm2d(input_channels),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
                                                    kernel_size=3, padding=1, stride=1, bias=False),
                                          nn.BatchNorm2d(input_channels))

    def forward(self, x):
        upper = self.upper_branch(x)
        lower = x
        return nn.ReLU()(upper+lower)


class ResNet34DeconvBlock(nn.Module):

    def __init__(self, input_channels):
        super(ResNet34DeconvBlock, self).__init__()

        self.upper_branch = nn.Sequential(nn.ConvTranspose2d(in_channels=input_channels, out_channels=input_channels//2,
                                                             kernel_size=2, padding=0, stride=2),
                                          nn.Conv2d(in_channels=input_channels//2, out_channels=input_channels//2,
                                                    kernel_size=3, padding=1, stride=1, bias=False),
                                          nn.BatchNorm2d(input_channels//2))

        self.lower_branch = nn.Sequential(nn.ConvTranspose2d(in_channels=input_channels, out_channels=input_channels//2,
                                                             kernel_size=2, padding=0, stride=2, bias=False),
                                          nn.BatchNorm2d(input_channels//2))

    def forward(self, x):
        upper = self.upper_branch(x)
        lower = self.lower_branch(x)
        return nn.ReLU()(upper+lower)
