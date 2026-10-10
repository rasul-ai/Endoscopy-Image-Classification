"""Building blocks of CGLE: local CNN+SE, SE block, global SE-ResNet, MLP head."""
import torch.nn as nn
import torchvision.models as models

class CNNwithSE(nn.Module):
    def __init__(self):
        super(CNNwithSE, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.drop3 = nn.Dropout(0.3)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

        self.se = SEBlock(64)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = self.maxpool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        x = self.maxpool(x)

        x = self.se(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel, height, width = x.size()
        y = self.avg_pool(x).view(batch_size, channel)
        y = self.fc(y).view(batch_size, channel, 1, 1)
        return x * y.expand_as(x)


class SEResNet(nn.Module):
    def __init__(self):
        super(SEResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Modify stride to preserve spatial dimensions
        self.resnet.layer4[0].conv2.stride = (1, 1)
        self.resnet.layer4[0].downsample[0].stride = (1, 1)
        self.resnet.layer4[0].conv3.stride = (1, 1)

        self.additional_conv = nn.Conv2d(2048, 64, kernel_size=1)
        self.bn = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)

        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.se = SEBlock(64)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.additional_conv(x)
        x = self.bn(x)
        x = self.dropout(x)

        x = self.avgpool(x)
        x = self.se(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(            
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 15)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x
