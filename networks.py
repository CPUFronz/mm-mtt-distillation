import torch.nn as nn
import torch.nn.functional as F
import torch
# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

# adapted from
# https://github.com/VICO-UoE/DatasetCondensation

''' MLP '''
class MLP(nn.Module):
    def __init__(self, channel, num_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out



''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers), shape_feat


''' ConvNet '''
class ConvNetGAP(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNetGAP, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        # self.classifier = nn.Linear(num_feat, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(shape_feat[0], num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' LeNet '''
class LeNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel==1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x



''' AlexNet '''
class AlexNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel==1 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



''' VGG '''
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, norm='instancenorm'):
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg_vgg[vgg_name], norm)
        self.classifier = nn.Linear(512 if vgg_name != 'VGGS' else 128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, norm):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel==1 and ic==0 else 1),
                           nn.GroupNorm(x, x, affine=True) if norm=='instancenorm' else nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(channel, num_classes):
    return VGG('VGG11', channel, num_classes)
def VGG11BN(channel, num_classes):
    return VGG('VGG11', channel, num_classes, norm='batchnorm')
def VGG13(channel, num_classes):
    return VGG('VGG13', channel, num_classes)
def VGG16(channel, num_classes):
    return VGG('VGG16', channel, num_classes)
def VGG19(channel, num_classes):
    return VGG('VGG19', channel, num_classes)


''' ResNet_AP '''
# The conv(stride=2) is replaced by conv(stride=1) + avgpool(kernel_size=2, stride=2)

class BasicBlock_AP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2), # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_AP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AP(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet_AP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion * 3 * 3 if channel==1 else 512 * block.expansion * 4 * 4, num_classes)  # modification

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1) # modification
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def ResNet18BN_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes)


''' ResNet '''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class ResNetImageNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNetImageNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def ResNet18BN(channel, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18(channel, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes)

def ResNet34(channel, num_classes):
    return ResNet(BasicBlock, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet50(channel, num_classes):
    return ResNet(Bottleneck, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet101(channel, num_classes):
    return ResNet(Bottleneck, [3,4,23,3], channel=channel, num_classes=num_classes)

def ResNet152(channel, num_classes):
    return ResNet(Bottleneck, [3,8,36,3], channel=channel, num_classes=num_classes)

def ResNet18ImageNet(channel, num_classes):
    return ResNetImageNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes)

def ResNet6ImageNet(channel, num_classes):
    return ResNetImageNet(BasicBlock, [1,1,1,1], channel=channel, num_classes=num_classes)


#####################################################################
# added by Franz
#####################################################################

class CrossAttentionFusion(nn.Module):
    def __init__(self, image_dim, sensor_dim, hidden_dim, num_heads):
        super(CrossAttentionFusion, self).__init__()
        self.image_to_sensor_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.sensor_to_image_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Linear projections to align feature dimensions
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
        self.fusion_proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, image_features, sensor_features):
        """
        image_features: [batch_size, feature_dim]
        sensor_features: [batch_size, feature_dim]
        """
        # Add sequence dimension (required by MultiheadAttention)
        image_features = image_features.unsqueeze(1)    # [batch_size, 1, image_dim]
        sensor_features = sensor_features.unsqueeze(1)  # [batch_size, 1, sensor_dim]

        # Project features to hidden_dim
        image_features = self.image_proj(image_features)     # [batch_size, 1, hidden_dim]
        sensor_features = self.sensor_proj(sensor_features)  # [batch_size, 1, hidden_dim]

        # Transpose for attention (expected shape: [seq_len, batch_size, hidden_dim])
        image_features = image_features.transpose(0, 1)    # [1, batch_size, hidden_dim]
        sensor_features = sensor_features.transpose(0, 1)  # [1, batch_size, hidden_dim]

        # Cross-attention
        attended_image_to_sensor, _ = self.image_to_sensor_attention(sensor_features, image_features, image_features)
        attended_sensor_to_image, _ = self.sensor_to_image_attention(image_features, sensor_features, sensor_features)

        # Combine attended features (concatenate and fuse)
        combined_features = torch.cat([attended_image_to_sensor, attended_sensor_to_image], dim=-1)
        combined_features = self.fusion_proj(combined_features)  # [1, batch_size, hidden_dim]

        # Remove sequence dimension and transpose back: [batch_size, hidden_dim]
        return combined_features.squeeze(0)
    

class MMSConvB(nn.Module):
    def __init__(self, n_layers_img, n_units_img, n_sensors, n_sensor_features, n_layers_sens, n_units_sens, n_heads_fusion, n_units_fusion, n_channels, n_classes, unimodal='', im_size=(32,32), n_groups=8):
        super(MMSConvB, self).__init__()

        self.unimodal = unimodal
        self.im_size = im_size
        self.n_groups = n_groups

        n_in_features_sens = n_sensors * n_sensor_features

        img_layers = [
            nn.Conv2d(n_channels, n_units_img, kernel_size=9, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_layers_img - 2):
            img_layers.extend([
                nn.Conv2d(n_units_img, n_units_img, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(self.n_groups, n_units_img, affine=True),
                nn.ReLU(inplace=True)
            ])
        img_layers.extend([
            nn.Conv2d(n_units_img, n_units_img, kernel_size=13, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        ])
        self.image_stack = nn.Sequential(*img_layers)

        sens_layer = [
            nn.Linear(n_in_features_sens, n_units_sens),
            nn.GroupNorm(self.n_groups, n_units_sens, affine=True),
            nn.ReLU(),
            nn.Dropout(),
        ]
        for _ in range(n_layers_sens - 1):
            sens_layer.extend([
                nn.Linear(n_units_sens, n_units_sens),
                nn.GroupNorm(self.n_groups, n_units_sens, affine=True), 
                nn.ReLU(),
                nn.Dropout(),
            ])
        self.sensor_stack = nn.Sequential(*sens_layer)

        # image_dim, sensor_dim, hidden_dim, num_heads
        self.attn = CrossAttentionFusion(n_units_img, n_units_sens, n_units_fusion, n_heads_fusion)

        self.head = nn.Sequential(
            nn.Linear(n_units_fusion, n_units_fusion),
            nn.GroupNorm(self.n_groups, n_units_fusion, affine=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_units_fusion, n_classes)
        )

    def forward(self, X):        
        if self.unimodal == '':
            img_in, sens_in = X
        elif self.unimodal == 'image':
            img_in = X[0]
            sens_in = torch.zeros((img_in.shape[0], self.sensor_stack[0].in_features)).to(img_in.device)
        elif self.unimodal == 'sensor':
            sens_in = X[1]
            img_in = torch.zeros((sens_in.shape[0], 3, self.im_size[0], self.im_size[1])).to(sens_in.device)
        else:
            raise ValueError("Invalid unimodal option.")
        
        
        device = img_in.device

        feature_map_image = self.image_stack(img_in.to(device))
        feature_map_sensor = self.sensor_stack(sens_in.to(device))
        fused_features = self.attn(feature_map_image, feature_map_sensor)
        logits = self.head(fused_features)
        return logits


import math, torch, torch.nn as nn
from perceiver_pytorch import PerceiverIO              # backbone

# ---------- fourier utilities ----------
def fourier_encode(x, max_freq, num_bands=6):
    dims = x.shape[-1]
    scales = torch.logspace(0., math.log(max_freq / 2, 2),
                            num_bands, base=2., device=x.device)
    enc = (x[..., None] * scales).reshape(*x.shape[:-1], -1)
    return torch.cat((x, torch.sin(enc), torch.cos(enc)), dim=-1)

# ---------- input adapters ----------
class ImageInputAdapter(nn.Module):
    def __init__(self, img_size=224, patch=16, in_ch=3, embed=128, bands=6, max_freq=10.):
        super().__init__()
        gh, gw = img_size[0] // patch, img_size[1] // patch
        self.proj = nn.Linear(in_ch * patch * patch, embed)

        ys, xs = torch.meshgrid(torch.linspace(-1, 1, gh),
                                torch.linspace(-1, 1, gw), indexing="ij")
        coords = torch.stack((xs, ys), dim=-1).view(-1, 2)
        self.register_buffer("pos",
                             fourier_encode(coords, max_freq, bands),
                             persistent=False)
        self.pos_proj = nn.Linear(self.pos.shape[-1], embed)

    def forward(self, x):                     # (B,3,224,224)
        B = x.size(0)
        p = x.unfold(2,16,16).unfold(3,16,16) # (B,3,Hp,Wp,16,16)
        p = p.reshape(B, 3, -1, 16, 16).permute(0,2,1,3,4)
        p = p.reshape(B, -1, 3*16*16)         # (B,N,768)
        t = self.proj(p) + self.pos_proj(self.pos).unsqueeze(0)
        return t                              # (B,N,128)
    

class SensorInputAdapter(nn.Module):
    def __init__(self, embed_dim=128, n_sensors=7, n_features=9, fourier_dim=16, hidden=32):
        super().__init__()

        self.value_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim)
        )

        self.n_sensors = n_sensors
        self.n_features = n_features

        self.sensor_emb = nn.Embedding(n_sensors, embed_dim)
        self.stat_emb   = nn.Embedding(n_features, embed_dim)

        self.freqs = nn.Parameter(
            torch.exp(-math.log(10_000) * torch.arange(fourier_dim) / fourier_dim),
            requires_grad=False
        )
        self.pos_proj = nn.Linear(2 * fourier_dim, embed_dim, bias=False)

        self.type_tok = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.norm = nn.LayerNorm(embed_dim)

    def fourier_encode(self, t):                     # t shape (...)
        ang = t.unsqueeze(-1) * self.freqs          # (…, F)
        return torch.cat([ang.sin(), ang.cos()], -1)

    def forward(self, x, pos_idx=None):        
        B = x.shape[0]
        S = self.n_sensors
        F = self.n_features
        x = x.reshape(B, S, F)

        # --- Values + Semantic -----------------------------------------------
        v_tok = self.value_mlp(x.unsqueeze(-1))      # (B, S, F, E)

        s_idx = torch.arange(S, device=x.device)
        f_idx = torch.arange(F, device=x.device)
        s_tok = self.sensor_emb(s_idx)[None, :, None, :]   # (1,S,1,E)
        f_tok = self.stat_emb(f_idx)[None, None, :, :]     # (1,1,F,E)

        tok = v_tok + s_tok + f_tok                       # (B,S,F,E)

        # --- Time Stamp-------------------------------------------------------
        if pos_idx is not None:                           # pos_idx (B,)
            p = self.pos_proj(self.fourier_encode(pos_idx.float()))  # (B,E)
            tok += p[:, None, None, :]

        tok = self.norm(tok).view(B, S * F, -1)           # flatten to (B,N,E)
        return torch.cat([self.type_tok.expand(B, -1, -1), tok], dim=1)



# ---------- multimodal perceiver ----------
class MultimodalPerceiver(nn.Module):
    def __init__(self, img_size=224, n_sensors=7, n_sensor_features=9, num_classes=10, latent_dim=32, token_dim=128, num_lat=128, depth=3, cross_heads=1, latent_heads=8, seq_dropout_prob=0.2, device='cuda'):
        super().__init__()
        
        self.img_adapt  = ImageInputAdapter(img_size=img_size, embed=token_dim)
        self.sens_adapt = SensorInputAdapter(embed_dim=token_dim, n_sensors=n_sensors, n_features=n_sensor_features)

        self.perceiver = PerceiverIO(
            dim          = token_dim,
            queries_dim  = latent_dim,
            num_latents  = num_lat,
            latent_dim   = latent_dim,
            depth        = depth,
            cross_heads  = cross_heads,
            latent_heads = latent_heads,
            weight_tie_layers=False,
            seq_dropout_prob=seq_dropout_prob
        )
        self.query      = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.out_proj   = nn.Linear(latent_dim, num_classes)
        
        self.device = device
        self.to(device)

    def forward(self, X):
        """
        img    : (B,3,224,224)
        sensor : (B,F)  variable-length
        """
        img, sensor = X
        img = img.to(self.device)
        sensor = sensor.to(self.device)

        tok = torch.cat([self.img_adapt(img),
                         self.sens_adapt(sensor)], dim=1)  # concat streams

        
        decoded = self.perceiver(tok, queries=self.query.repeat(img.size(0),1,1))
        return self.out_proj(decoded.squeeze(1))           # (B, num_classes)


class Widar_CNN3D(nn.Module):
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        self.conv1   = nn.Conv3d(1,   64,  kernel_size=3, padding=1)
        self.conv2   = nn.Conv3d(64,  128, kernel_size=3, padding=1)
        self.conv3   = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool3d(2)
        self.drop    = nn.Dropout(dropout)
        self.fc1     = nn.Linear(256 * 2 * 2 * 2, 512)
        self.fc2     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

    def get_features(self, x):
        """Returns 512-dim penultimate layer features — used for selection."""
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))                  # (B, 512)
