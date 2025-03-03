import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Depthwise Separable Convolution
# -------------------------------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    

# -------------------------------
# Squeeze-and-Excitation Block
# -------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# -------------------------------
# Residual Blocks (Basic and Bottleneck)
# -------------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, use_projection=False, use_se=False, dropout_rate=0.0, use_depthwise=False):
        super(BasicBlock, self).__init__()

        conv_layer = DepthwiseSeparableConv if use_depthwise else nn.Conv2d #Changes for depthwise

        self.use_projection = use_projection
        self.use_se = use_se
        self.dropout_rate = dropout_rate
        self.conv1 = conv_layer(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.use_projection:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        if self.use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        if self.use_se:
            out = self.se(out)
        if self.use_projection:
            identity = self.proj(x)
        out += identity
        out = self.relu(out)
        return out

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, use_projection=False, use_se=False, dropout_rate=0.0, use_depthwise=False):
        super(BottleneckBlock, self).__init__()

        conv_layer = DepthwiseSeparableConv if use_depthwise else nn.Conv2d  #Changes for depthwise

        self.use_projection = use_projection
        self.use_se = use_se
        self.dropout_rate = dropout_rate
        width = out_channels
        self.conv1 = conv_layer(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv_layer(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv_layer(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if self.use_projection:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        if self.use_se:
            self.se = SEBlock(out_channels * self.expansion)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)
        if self.use_projection:
            identity = self.proj(x)
        out += identity
        out = self.relu(out)
        return out

# -------------------------------
# EfficientResNet Model controlled by config (including depth via num_blocks)
# -------------------------------
class EfficientResNet(nn.Module):
    def __init__(self, config):
        super(EfficientResNet, self).__init__()
        self.use_bottleneck = config['model'].get('use_bottleneck', False)
        self.use_depthwise = config['model'].get('use_depthwise', False) #Changes for depthwise
        self.use_se = config['model'].get('squeeze_and_excitation', False)
        self.use_dropout = config['model'].get('use_dropout', False)
        self.dropout_rate = config['model'].get('dropout_rate', 0.0) if self.use_dropout else 0.0
        channels = config['model']['channels']           # e.g. [64, 128, 256]
        num_blocks = config['model']['num_blocks']         # e.g. [2, 2, 2] controls the depth
        self.in_channels = channels[0]
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        block = BottleneckBlock if self.use_bottleneck else BasicBlock
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], stride=1, use_depthwise=self.use_depthwise)   #Changes for depthwise
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], stride=2, use_depthwise=self.use_depthwise)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], stride=2, use_depthwise=self.use_depthwise)
        self.layer4 = self._make_layer(block, channels[3], num_blocks[3], stride=2, use_depthwise=self.use_depthwise)
        final_channels = channels[3] * (block.expansion if self.use_bottleneck else 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(final_channels, 10)
    def _make_layer(self, block, out_channels, blocks, stride, use_depthwise=False): #changes for depthwise
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride,
                            use_projection=(self.in_channels != out_channels * block.expansion),
                            use_se=self.use_se, dropout_rate=self.dropout_rate, use_depthwise = use_depthwise)) #changes for depthwise
        
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_se=self.use_se, dropout_rate=self.dropout_rate, use_depthwise = use_depthwise)) #changes for depthwise
        return nn.Sequential(*layers)
    

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out