import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):

    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.patch_keys = kwargs['PATCH_KEYS']
        self.target_keys = kwargs['TARGET_KEYS']

        # Get ResNet model without first and last elem
        self.resnet34 = models.resnet34(pretrained=kwargs['PRETRAINED_RESNET'], progress=True)
        # First conv got only 2 channels
        self.resnet34.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Last FC has 8 output units
        self.resnet34.fc = nn.Linear(512, 8, bias=True)

        self.variant = str.lower(kwargs['VARIANT']) if 'VARIANT' in kwargs else 'oneline'
        assert 'oneline' in self.variant or 'doubleline' in self.variant, 'Only OneLine or DoubleLine variant is' \
                                                                          'supported'

    def single_forward(self, x):
        # Forward
        out = self.resnet34(x)
        return out.reshape(-1, 4, 2)

    def forward(self, data):

        (e1, e2) = self.patch_keys

        # Oneline
        o1 = self.target_keys[0]
        p1 = data[e1]
        p2 = data[e2]
        x12 = torch.cat([p1, p2], axis=1)
        data[o1] = self.single_forward(x12)

        # Double
        if self.variant == 'doubleline':
            o2 = self.target_keys[1]
            x21 = torch.cat([p2, p1], axis=1)
            data[o2] = self.single_forward(x21)

        return data

    def predict_homography(self, data):
        return self.forward(data)
