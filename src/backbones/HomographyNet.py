import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.image_size = kwargs['IMAGE_SIZE']
        self.patch_keys = kwargs['PATCH_KEYS']
        self.target_keys = kwargs['TARGET_KEYS']

        self.layer1 = nn.Sequential(nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        self.layer6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))

        if self.image_size == 128:
            self.layer8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        elif self.image_size == 512:
            self.layer8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                        nn.MaxPool2d(2))
            self.layer9 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
            self.layer10 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                         nn.MaxPool2d(2))
            self.layer11 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
            self.layer12 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))

        self.fc1 = nn.Sequential(nn.Linear(128 * 16 * 16, 1024), nn.ReLU())
        self.fc2 = nn.Linear(1024, 8)

    def _forward(self, x):
        # Forward
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        if self.image_size == 512:
            out = self.layer9(out)
            out = self.layer10(out)
            out = self.layer11(out)
            out = self.layer12(out)

        out = out.view(-1, 128 * 16 * 16)
        out = self.fc1(out)
        out = self.fc2(out)
        return out.reshape(-1, 4, 2)

    def forward(self, data):

        (e1, e2) = self.patch_keys
        o = self.target_keys[0]
        p1 = data[e1]
        p2 = data[e2]
        x = torch.cat([p1, p2], axis=1)
        data[o] = self._forward(x)

        return data

    def predict_homography(self, data):
        return self.forward(data)
