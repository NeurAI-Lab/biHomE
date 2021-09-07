import torch
import torch.nn as nn
import torchvision.models as models


class MaskPredictor(nn.Module):

    def __init__(self, fix_mask=False, normalization_strength=-1):
        super(MaskPredictor, self).__init__()
        self.fix_mask = fix_mask
        self.normalization_strength = normalization_strength
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(4), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(8), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(16), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(32), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(1), nn.Sigmoid())

    @staticmethod
    def __normalize_mask(mask, strength=0.5):
        batch_size, c_m, c_h, c_w = mask.size()
        max_value = mask.reshape(batch_size, -1).max(1)[0]
        max_value = max_value.reshape(batch_size, 1, 1, 1)
        mask = mask / (max_value * strength)
        mask = torch.clamp(mask, 0, 1)
        return mask

    def forward(self, x):
        if self.fix_mask:
            out = torch.ones_like(x)
        else:
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            assert out.shape[-2:] == x.shape[-2:], 'Mask and input image should have the same w/h'

            # Normalize mask
            if self.normalization_strength > 0:
                out = self.__normalize_mask(out, strength=self.normalization_strength)

        return out


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(4), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(8), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(1), nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        assert out.shape[-2:] == x.shape[-2:], 'Feature map and input image should have the same w/h'
        return out

    def retrieve_weights(self):
        weights = {}
        for name, parameter in self.named_parameters():
            weights[name] = parameter.data
        return weights


class Model(nn.Module):

    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.patch_keys = kwargs['PATCH_KEYS']
        self.mask_keys = kwargs['MASK_KEYS']
        self.feature_keys = kwargs['FEATURE_KEYS']
        self.target_keys = kwargs['TARGET_KEYS']

        mask_norm_strength = kwargs['MASK_NORMALIZATION_STRENGTH'] if 'MASK_NORMALIZATION_STRENGTH' in kwargs else -1
        self.mask_predictor = MaskPredictor(fix_mask=kwargs['FIX_MASK'], normalization_strength=mask_norm_strength)
        self.feature_extractor = FeatureExtractor()

        self.variant = str.lower(kwargs['VARIANT'])
        assert 'oneline' in self.variant or 'doubleline' in self.variant, 'Only OneLine or DoubleLine variant is' \
                                                                          'supported'

        # Init weights if we're using pretrained resnet
        pretrained_resnet = kwargs['PRETRAINED_RESNET']
        if pretrained_resnet:
            self.init()

        # Get ResNet model without first and last elem
        self.resnet34 = models.resnet34(pretrained=pretrained_resnet, progress=True)
        # First conv got only 2 channels
        self.resnet34.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Last FC has 8 output units
        self.resnet34.fc = nn.Linear(512, 8, bias=True)

        # Init weights of resnet also
        if not pretrained_resnet:
            self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _forward(self, input_1, input_2):

        # Create G1
        m1 = self.mask_predictor(input_1)
        f1 = self.feature_extractor(input_1)
        g1 = torch.mul(m1, f1)
        assert g1.shape == input_1.shape, 'G feature map should have the same size as input image'

        # Create G2
        m2 = self.mask_predictor(input_2)
        f2 = self.feature_extractor(input_2)
        g2 = torch.mul(m2, f2)
        assert g2.shape == input_2.shape, 'G feature map should have the same size as input image'

        # resnet pass
        g = torch.cat([g1, g2], axis=1)
        assert g.shape[1] == 2, 'G feature map should have 2 channels'
        assert g.shape[-2:] == input_1.shape[-2:], 'G feature map w/h should be the same as input image'
        o = self.resnet34(g).reshape(-1, 4, 2)

        # Return
        return m1, f1, m2, f2, g1, g2, o

    def forward(self, data):

        # image_1, image_2, mask_1, mask_2, features_1, features_2, delta_1to2_hat, delta_2to1_hat
        # for (e1, e2, o) in [('image', 'positive', 'delta_qs_hat'), ('image', 'weak_positive', 'delta_qw_hat'),
        #                     ('positive', 'weak_positive', 'delta_pw_hat')]:
        e1, e2 = self.patch_keys
        m1, m2 = self.mask_keys
        f1, f2 = self.feature_keys
        o1 = self.target_keys[0]

        # Main pass
        data[m1], data[f1], data[m2], data[f2], g1, g2, data[o1] = self._forward(data[e1], data[e2])

        #######################################################################
        # Prepare first patch warped
        #######################################################################

        # Auxiliary pass
        if self.variant == 'doubleline':
            g = torch.cat([g2, g1], axis=1)
            assert g.shape[1] == 2, 'G feature map should have 2 channels'
            assert g.shape[-2:] == data[e1].shape[-2:], 'G feature map w/h should be the same as input image'
            o2 = self.target_keys[1]
            data[o2] = self.resnet34(g).reshape(-1, 4, 2)

        # Return
        return data

    def predict_homography(self, data):

        # Get keys
        e1, e2 = self.patch_keys
        o1 = self.target_keys[0]
        m1, m2 = self.mask_keys

        # Main pass
        data[m1], _, data[m2], _, _, _, data[o1] = self._forward(data[e1], data[e2])

        # Return
        return data

    def retrieve_weights(self):
        weights = {}
        for name, parameter in self.resnet34.named_parameters():
            weights[name] = parameter.data
        return weights