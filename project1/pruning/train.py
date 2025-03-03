import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torchsummary import summary

# ---------------------------
# Utility functions
# ---------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ---------------------------
# Depthwise Separable Convolution Block
# ---------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution: groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        # Pointwise convolution: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

# ---------------------------
# Basic Residual Block (with optional depthwise conv)
# ---------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_depthwise=False):
        super(BasicBlock, self).__init__()
        conv_layer = DepthwiseSeparableConv if use_depthwise else nn.Conv2d

        self.conv1 = conv_layer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ---------------------------
# ResNet Model (supports multiple variants)
# ---------------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, use_depthwise=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.use_depthwise = use_depthwise
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # CIFAR-10 images are 32x32
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # For CIFAR-10, we typically do not use a maxpool
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, use_depthwise=use_depthwise)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_depthwise=use_depthwise)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_depthwise=use_depthwise)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_depthwise=use_depthwise)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride, use_depthwise):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, use_depthwise))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_depthwise=use_depthwise))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_resnet(model_type, use_depthwise):
    """
    Return a ResNet model given the model type.
    For simplicity, we use the BasicBlock for all variants.
    You can extend this to use Bottleneck blocks for larger networks if needed.
    """
    if model_type == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], use_depthwise=use_depthwise)
    elif model_type == "resnet30":
        # A custom configuration: e.g., [3, 4, 4, 3] blocks (example configuration)
        return ResNet(BasicBlock, [3, 4, 4, 3], use_depthwise=use_depthwise)
    elif model_type == "resnet50":
        # For ResNet50, a bottleneck block is normally used.
        # Here we assume a custom implementation or approximate with more BasicBlocks.
        return ResNet(BasicBlock, [3, 4, 6, 3], use_depthwise=use_depthwise)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# ---------------------------
# Function to apply pruning to convolutional layers
# ---------------------------
def apply_pruning(model, pruning_ratio):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Prune 20% of weights (by default, using L1 unstructured pruning)
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            # Optionally remove the pruning reparameterization:
            prune.remove(module, 'weight')
    return model

# ---------------------------
# Data Augmentation and DataLoader Setup
# ---------------------------
def get_data_loaders(batch_size, augmentation_config):
    transform_train = []
    if augmentation_config.get("random_crop", False):
        transform_train.append(transforms.RandomCrop(32, padding=4))
    if augmentation_config.get("random_flip", False):
        transform_train.append(transforms.RandomHorizontalFlip())
    transform_train.append(transforms.ToTensor())
    transform_train.append(transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                (0.2023, 0.1994, 0.2010)))
    train_transform = transforms.Compose(transform_train)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

# ---------------------------
# Training and Evaluation Functions
# ---------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ---------------------------
# Main Training Loop
# ---------------------------
def main():
    config = load_config()
    set_seed(config["training"].get("seed", 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model based on config
    model = get_resnet(config["model"]["type"], config["model"]["use_depthwise"])
    model = model.to(device)

    summary(model, input_size=(3, 32, 32))

    
    # Optionally apply pruning
    if config["model"].get("pruning", {}).get("enabled", False):
        pruning_ratio = config["model"]["pruning"].get("pruning_ratio", 0.2)
        model = apply_pruning(model, pruning_ratio)
    
    # Count parameters to ensure within limits
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    if total_params > 5e6:
        print("Warning: Model exceeds 5 million parameters!")
    
    # Data loaders
    trainloader, testloader = get_data_loaders(config["training"]["batch_size"], config["training"]["augmentation"])
    
    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer_type = config["training"]["optimizer"]
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                               weight_decay=config["training"]["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["training"]["learning_rate"],
                              momentum=0.9, weight_decay=config["training"]["weight_decay"])
    
    # Setup learning rate scheduler
    scheduler_name = config["training"].get("scheduler", "CosineAnnealingLR")
    if scheduler_name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    elif scheduler_name == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    best_acc = 0.0
    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")
    
    print(f"Best Test Accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()
