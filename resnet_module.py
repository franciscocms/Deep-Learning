import torch
import torch.nn as nn

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_size, downsampling = 1, activation='relu'):
        super().__init__()
        
        self.downsampling = downsampling

        self.in_channels, self.out_channels = in_channels, out_channels

        if in_channels != out_channels:
            self.downsampling = 2
        
        self.bottleneck_size = bottleneck_size
        if self.bottleneck_size == 2:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = self.downsampling, padding = 1, bias = False)
            self.bn1 = nn.BatchNorm2d(num_features = out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.bn2 = nn.BatchNorm2d(num_features = out_channels)
            self.activate = activation_func(activation)
        
        # in_channels and out_channels from the block (first: 64 and 256)
        elif self.bottleneck_size == 3:
            self.conv1 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size = 1, bias = False)
            self.bn1 = nn.BatchNorm2d(num_features = int(out_channels/4))
            self.conv2 = nn.Conv2d(int(out_channels/4), int(out_channels/4), kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.bn2 = nn.BatchNorm2d(num_features = int(out_channels/4))
            self.conv3 = nn.Conv2d(int(out_channels/4), out_channels, kernel_size = 1, bias = False)
            self.bn3 = nn.BatchNorm2d(num_features = out_channels)
        
        
        self.activate = activation_func(activation)
  
    def forward(self, x):

        if self.bottleneck_size == 2:        
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activate(out)

            out = self.conv2(out)
            out = self.bn2(out)

            return out

        elif self.bottleneck_size == 3:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activate(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.activate(out)

            out = self.conv3(out)
            out = self.bn3(out)

            return out

class ResNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, bottleneck_size, downsampling = 1, activation='relu'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_size = bottleneck_size
        self.downsampling = downsampling
        
        if self.bottleneck_size == 2:
            self.downsampling = 1 if self.in_channels == self.out_channels else 2 
        else:
            self.downsampling = 1
      
        self.blocks = nn.ModuleList([
            BasicBlock(self.in_channels, self.out_channels, self.bottleneck_size),
            *[BasicBlock(self.out_channels, self.out_channels, self.bottleneck_size) for i in range(depth - 1)]
        ])

        self.shortcuts = nn.ModuleList([
            *[nn.Sequential(
                nn.Conv2d(self.blocks[i].in_channels, self.blocks[i].out_channels, kernel_size = 1, stride = self.downsampling, bias = False),
                nn.BatchNorm2d(self.blocks[i].out_channels)) if self.blocks[i].in_channels != self.blocks[i].out_channels else nn.Identity() for i in range(depth)]
        ])

        self.activate = activation_func(activation)
    
    def forward(self, x):
          
        for i in range(len(self.blocks)):
            residual = self.shortcuts[i](x)
            x = self.blocks[i](x)
            x += residual
            x = self.activate(x)

        return x

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, depths, bottleneck_size, block_sizes, activation='relu'):
        super().__init__()
        
        self.in_channels = in_channels

        self.block = ResNetResidualBlock
        self.block_sizes = block_sizes
        self.n = len(self.block_sizes)

        
        self.gate = nn.Sequential(
            nn.Conv2d(self.in_channels, self.block_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.block_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        if bottleneck_size == 2:
            self.blocks = nn.Sequential(
                self.block(self.block_sizes[0], self.block_sizes[0], depth = depths[0], bottleneck_size = bottleneck_size),
                *[self.block(self.block_sizes[k], self.block_sizes[k + 1], depth = depths[k+1], bottleneck_size = bottleneck_size) for k in range(len(depths) -1)]       
            )
        elif bottleneck_size == 3:
            self.blocks = nn.Sequential(
                self.block(self.block_sizes[0], self.block_sizes[0]*4, depth = depths[0], bottleneck_size = bottleneck_size),
                *[self.block(self.block_sizes[k]*4, self.block_sizes[k + 1]*4, depth = depths[k+1], bottleneck_size = bottleneck_size) for k in range(len(depths) -1)]       
            )

    def forward(self, x):

        x = self.gate(x)
        x = self.blocks(x)
        
        return x

class ResNetClassifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, depths, block_sizes, bottleneck_size):
        super().__init__()

        self.encoder = ResNetEncoder(in_channels, depths, bottleneck_size, block_sizes)
        self.classifier = ResNetClassifier(self.encoder.blocks[-1].out_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

from torchsummary import summary
import torchvision.models as models

def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, depths = [2,2,2,2], block_sizes = [64, 128, 256, 512], bottleneck_size = 2)

def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, depths = [3,4,6,3], block_sizes = [64, 128, 256, 512], bottleneck_size = 2)

def resnet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, depths = [3,4,6,3], block_sizes = [64, 128, 256, 512], bottleneck_size = 3)

def resnet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, depths = [3,4,23,3], block_sizes = [64, 128, 256, 512], bottleneck_size = 3)

def resnet152(in_channels, n_classes):
    return ResNet(in_channels, n_classes, depths = [3,8,36,3], block_sizes = [64, 128, 256, 512], bottleneck_size = 3)


