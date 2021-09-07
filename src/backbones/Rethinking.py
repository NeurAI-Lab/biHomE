import torch
import torch.nn as nn

from src.backbones.utils import ResNet50ConvBlock
from src.backbones.utils import ResNet50IdentityBlock
from src.backbones.utils import ResNet50DeconvBlock
from src.backbones.utils import ResNet34ConvBlock
from src.backbones.utils import ResNet34IdentityBlock


class Model(nn.Module):

    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.image_size = kwargs['IMAGE_SIZE']
        self.patch_keys = kwargs['PATCH_KEYS']
        self.target_keys = kwargs['TARGET_KEYS']

        self.resnet_block = kwargs['RESNET_BLOCK']
        self.pretrained_resnet = kwargs['PRETRAINED_RESNET']

        self.variant = str.lower(kwargs['VARIANT']) if 'VARIANT' in kwargs else 'oneline'
        assert 'oneline' in self.variant or 'doubleline' in self.variant, 'Only OneLine or DoubleLine variant is' \
                                                                          'supported'

        #######################################################################
        # STAGE 1
        #######################################################################

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, padding=3, stride=2,
                                              bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        #######################################################################
        # STAGE 2
        #######################################################################

        if self.resnet_block == 'ResNet50':
            self.layer2 = nn.Sequential(ResNet50ConvBlock(input_channels=64, output_channels=256, stride=1),
                                        ResNet50IdentityBlock(input_channels=256),
                                        ResNet50IdentityBlock(input_channels=256))
        elif self.resnet_block == 'ResNet34':
            self.layer2 = nn.Sequential(ResNet34ConvBlock(input_channels=64, output_channels=64, stride=1),
                                        ResNet34IdentityBlock(input_channels=64),
                                        ResNet34IdentityBlock(input_channels=64))
        else:
            assert False, 'I know only ResNet50 and ResNet34'

        #######################################################################
        # STAGE 3
        #######################################################################

        if self.resnet_block == 'ResNet50':
            self.layer3 = nn.Sequential(ResNet50ConvBlock(input_channels=256, output_channels=512, stride=2),
                                        ResNet50IdentityBlock(input_channels=512),
                                        ResNet50IdentityBlock(input_channels=512),
                                        ResNet50IdentityBlock(input_channels=512))
        elif self.resnet_block == 'ResNet34':
            self.layer3 = nn.Sequential(ResNet34ConvBlock(input_channels=64, output_channels=128, stride=2),
                                        ResNet34IdentityBlock(input_channels=128),
                                        ResNet34IdentityBlock(input_channels=128),
                                        ResNet34IdentityBlock(input_channels=128))
        else:
            assert False, 'I know only ResNet50 and ResNet34'

        #######################################################################
        # STAGE 4
        #######################################################################

        if self.resnet_block == 'ResNet50':
            self.layer4 = nn.Sequential(ResNet50ConvBlock(input_channels=512, output_channels=1024, stride=2),
                                        ResNet50IdentityBlock(input_channels=1024),
                                        ResNet50IdentityBlock(input_channels=1024),
                                        ResNet50IdentityBlock(input_channels=1024),
                                        ResNet50IdentityBlock(input_channels=1024),
                                        ResNet50IdentityBlock(input_channels=1024),
                                        ResNet50DeconvBlock(input_channels=1024))
        elif self.resnet_block == 'ResNet34':
            self.layer4 = nn.Sequential(ResNet34ConvBlock(input_channels=128, output_channels=256, stride=2),
                                        ResNet34IdentityBlock(input_channels=256),
                                        ResNet34IdentityBlock(input_channels=256),
                                        ResNet34IdentityBlock(input_channels=256),
                                        ResNet34IdentityBlock(input_channels=256),
                                        ResNet34IdentityBlock(input_channels=256),
                                        ResNet50DeconvBlock(input_channels=256))
        else:
            assert False, 'I know only ResNet50 and ResNet34'

        #######################################################################
        # STAGE 5
        #######################################################################

        if self.resnet_block == 'ResNet50':
            self.layer5 = nn.Sequential(ResNet50IdentityBlock(input_channels=512),
                                        ResNet50IdentityBlock(input_channels=512),
                                        ResNet50IdentityBlock(input_channels=512),
                                        ResNet50DeconvBlock(input_channels=512))
        elif self.resnet_block == 'ResNet34':
            self.layer5 = nn.Sequential(ResNet34IdentityBlock(input_channels=128),
                                        ResNet34IdentityBlock(input_channels=128),
                                        ResNet34IdentityBlock(input_channels=128),
                                        ResNet50DeconvBlock(input_channels=128))
        else:
            assert False, 'I know only ResNet50 and ResNet34'

        #######################################################################
        # STAGE 6
        #######################################################################

        if self.resnet_block == 'ResNet50':
            self.layer6 = nn.Sequential(ResNet50IdentityBlock(input_channels=256),
                                        ResNet50IdentityBlock(input_channels=256),
                                        ResNet50DeconvBlock(input_channels=256))
        elif self.resnet_block == 'ResNet34':
            self.layer6 = nn.Sequential(ResNet34IdentityBlock(input_channels=64),
                                        ResNet34IdentityBlock(input_channels=64),
                                        ResNet50DeconvBlock(input_channels=64))
        else:
            assert False, 'I know only ResNet50 and ResNet34'

        #######################################################################
        # STAGE 7
        #######################################################################

        if self.resnet_block == 'ResNet50':
            self.layer7 = nn.Sequential(ResNet50IdentityBlock(input_channels=128),
                                        ResNet50DeconvBlock(input_channels=128))
        elif self.resnet_block == 'ResNet34':
            self.layer7 = nn.Sequential(ResNet34IdentityBlock(input_channels=32),
                                        ResNet50DeconvBlock(input_channels=32))
        else:
            assert False, 'I know only ResNet50 and ResNet34'
        #######################################################################
        # STAGE 8
        #######################################################################

        if self.resnet_block == 'ResNet50':
            self.layer8 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=512, kernel_size=1, padding=0, stride=1),
                                        nn.BatchNorm2d(512), nn.ReLU(),
                                        nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, padding=0, stride=1))
        elif self.resnet_block == 'ResNet34':
            self.layer8 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=128, kernel_size=1, padding=0, stride=1),
                                        nn.BatchNorm2d(128), nn.ReLU(),
                                        nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, padding=0, stride=1))
        else:
            assert False, 'I know only ResNet50 and ResNet34'

        #######################################################################
        # Load pretrained Resnet50 weights
        #######################################################################

        if self.pretrained_resnet:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):

        ###################################################################
        # Help function
        ###################################################################

        def rename(old_dict, old_name, new_name):
            new_dict = {}
            for key, value in zip(old_dict.keys(), old_dict.values()):
                new_key = key if key != old_name else new_name
                new_dict[new_key] = old_dict[key]
            return new_dict

        ###################################################################
        # Download weights
        ###################################################################

        # Get pretrained weights
        from torch.hub import load_state_dict_from_url
        if self.resnet_block == 'ResNet50':
            resnet_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        elif self.resnet_block == 'ResNet34':
            resnet_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        else:
            assert False, 'I know only ResNet50 and ResNet34'
        resnet_model_state = load_state_dict_from_url(resnet_url, progress=True)

        ###################################################################
        # Layer 1
        ###################################################################

        layer1_resnet = dict(filter(lambda elem: 'layer1' in elem[0], resnet_model_state.items()))
        for e in list(layer1_resnet.keys()):
            target_key = e
            if 'layer1' in target_key:
                target_key = target_key.replace('layer1', 'layer2')
            if 'conv1' in target_key:
                target_key = target_key.replace('conv1', 'upper_branch.0')
            if 'bn1' in target_key:
                target_key = target_key.replace('bn1', 'upper_branch.1')
            if 'conv2' in target_key:
                target_key = target_key.replace('conv2', 'upper_branch.3')
            if 'bn2' in target_key:
                target_key = target_key.replace('bn2', 'upper_branch.4')
            if 'conv3' in target_key:
                target_key = target_key.replace('conv3', 'upper_branch.6')
            if 'bn3' in target_key:
                target_key = target_key.replace('bn3', 'upper_branch.7')
            if 'downsample' in target_key:
                target_key = target_key.replace('downsample', 'lower_branch')
            layer1_resnet = rename(layer1_resnet, e, target_key)

        model_dict = self.state_dict()
        model_dict = dict(filter(lambda elem: 'layer2' in elem[0], model_dict.items()))
        for e in layer1_resnet.keys():
            if layer1_resnet[e].shape != model_dict[e].shape:
                print('layer1: SOMETHING WENT WRONG!', layer1_resnet[e].shape, model_dict[e].shape)

        ###################################################################
        # Layer 2
        ###################################################################

        layer2_resnet = dict(filter(lambda elem: 'layer2' in elem[0], resnet_model_state.items()))
        for e in list(layer2_resnet.keys()):
            target_key = e
            if 'layer2' in target_key:
                target_key = target_key.replace('layer2', 'layer3')
            if 'conv1' in target_key:
                target_key = target_key.replace('conv1', 'upper_branch.0')
            if 'bn1' in target_key:
                target_key = target_key.replace('bn1', 'upper_branch.1')
            if 'conv2' in target_key:
                target_key = target_key.replace('conv2', 'upper_branch.3')
            if 'bn2' in target_key:
                target_key = target_key.replace('bn2', 'upper_branch.4')
            if 'conv3' in target_key:
                target_key = target_key.replace('conv3', 'upper_branch.6')
            if 'bn3' in target_key:
                target_key = target_key.replace('bn3', 'upper_branch.7')
            if 'downsample' in target_key:
                target_key = target_key.replace('downsample', 'lower_branch')
            layer2_resnet = rename(layer2_resnet, e, target_key)

        model_dict = self.state_dict()
        model_dict = dict(filter(lambda elem: 'layer3' in elem[0], model_dict.items()))
        for e in layer2_resnet.keys():
            if layer2_resnet[e].shape != model_dict[e].shape:
                print('layer2: SOMETHING WENT WRONG!', layer2_resnet[e].shape, model_dict[e].shape, e)

        ###################################################################
        # Layer 3
        ###################################################################

        layer3_resnet = dict(filter(lambda elem: 'layer3' in elem[0], resnet_model_state.items()))
        for e in list(layer3_resnet.keys()):
            target_key = e
            if 'layer3' in target_key:
                target_key = target_key.replace('layer3', 'layer4')
            if 'conv1' in target_key:
                target_key = target_key.replace('conv1', 'upper_branch.0')
            if 'bn1' in target_key:
                target_key = target_key.replace('bn1', 'upper_branch.1')
            if 'conv2' in target_key:
                target_key = target_key.replace('conv2', 'upper_branch.3')
            if 'bn2' in target_key:
                target_key = target_key.replace('bn2', 'upper_branch.4')
            if 'conv3' in target_key:
                target_key = target_key.replace('conv3', 'upper_branch.6')
            if 'bn3' in target_key:
                target_key = target_key.replace('bn3', 'upper_branch.7')
            if 'downsample' in target_key:
                target_key = target_key.replace('downsample', 'lower_branch')
            layer3_resnet = rename(layer3_resnet, e, target_key)

        model_dict = self.state_dict()
        model_dict = dict(filter(lambda elem: 'layer4' in elem[0], model_dict.items()))
        for e in layer3_resnet.keys():
            if layer3_resnet[e].shape != model_dict[e].shape:
                print('layer3: SOMETHING WENT WRONG!', layer3_resnet[e].shape, model_dict[e].shape, e)

        ###################################################################
        # Load weights
        ###################################################################

        self.load_state_dict({**layer1_resnet, **layer2_resnet, **layer3_resnet}, strict=False)

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
        return out

    def forward(self, data):

        (e1, e2) = self.patch_keys

        # Oneline
        o1 = self.target_keys[0]
        p1 = data[e1]
        p2 = data[e2]
        x12 = torch.cat([p1, p2], axis=1)
        data[o1] = self._forward(x12)

        # Double
        if self.variant == 'doubleline':
            o2 = self.target_keys[1]
            x21 = torch.cat([p2, p1], axis=1)
            data[o2] = self._forward(x21)

        return data

    def predict_homography(self, data):
        return self.forward(data)
