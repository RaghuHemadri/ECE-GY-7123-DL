import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import pickle

# Ray Tune imports
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# ---------------------------
# Utility functions
# ---------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
# Basic Residual Block (for shallower models)
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
# Bottleneck Block (for deeper networks)
# ---------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, use_depthwise=False):
        super(Bottleneck, self).__init__()
        conv_layer = DepthwiseSeparableConv if use_depthwise else nn.Conv2d

        # 1x1 conv to reduce dimensions
        self.conv1 = conv_layer(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3 conv
        self.conv2 = conv_layer(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1 conv to restore dimensions
        self.conv3 = conv_layer(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # For CIFAR-10, 32x32 images
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # No maxpool for CIFAR-10 images
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
    Uses BasicBlock for shallower networks and Bottleneck for deeper ones.
    """
    if model_type == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], use_depthwise=use_depthwise)
    elif model_type == "resnet30":
        # Custom configuration (example)
        return ResNet(BasicBlock, [3, 4, 4, 3], use_depthwise=use_depthwise)
    elif model_type == "resnet50":
        return ResNet(Bottleneck, [3, 4, 6, 3], use_depthwise=use_depthwise)
    elif model_type == "resnet101":
        return ResNet(Bottleneck, [3, 4, 23, 3], use_depthwise=use_depthwise)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# ---------------------------
# Function to apply pruning to convolutional layers
# ---------------------------
def apply_pruning(model, pruning_ratio):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
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
# Ray Tune Training Function
# ---------------------------
def train_cifar(config, checkpoint_dir=None):
    # Set seed for reproducibility
    set_seed(config.get("seed", 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model based on hyperparameters
    model = get_resnet(config["model_type"], config["use_depthwise"])
    model = model.to(device)
    
    # Print model summary (for debugging; can be commented out during large-scale tuning)
    summary(model, input_size=(3, 32, 32))
    
    # Optionally apply pruning
    if config.get("pruning_enabled", False):
        pruning_ratio = config.get("pruning_ratio", 0.2)
        model = apply_pruning(model, pruning_ratio)
    
    # Data loaders
    augmentation_config = config.get("augmentation", {})
    trainloader, testloader = get_data_loaders(config["batch_size"], augmentation_config)
    
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer selection
    optimizer_type = config.get("optimizer", "Adam")
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"],
                               weight_decay=config["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"],
                              momentum=0.9, weight_decay=config["weight_decay"])
    
    # Scheduler setup
    scheduler_name = config.get("scheduler", "CosineAnnealingLR")
    if scheduler_name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    elif scheduler_name == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    # Optionally load from checkpoint
    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Training loop
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        if scheduler:
            scheduler.step()
        
        # Save checkpoint every epoch (optional)
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, path)
        
        # Report metrics to Ray Tune
        tune.report(epoch=epoch, test_loss=test_loss, test_acc=test_acc)

# ---------------------------
# Main function for Ray Tune hyperparameter tuning
# ---------------------------
if __name__ == "__main__":
    ray.init()

    # Define the hyperparameter search space
    config = {
        "model_type": tune.choice(["resnet18", "resnet30", "resnet50", "resnet101"]),
        "use_depthwise": tune.choice([True, False]),
        "pruning_enabled": tune.choice([True, False]),
        "pruning_ratio": tune.uniform(0.1, 0.3),
        "batch_size": tune.choice([64, 128]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "optimizer": tune.choice(["Adam", "SGD"]),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "scheduler": tune.choice(["CosineAnnealingLR", "StepLR"]),
        "epochs": 20,  # For tuning, you might run a smaller number of epochs
        "augmentation": {
            "random_crop": True,
            "random_flip": True,
        },
        "seed": 42
    }

    # Use ASHA scheduler for early stopping of poor performing trials
    scheduler = ASHAScheduler(
        metric="test_loss",
        mode="min",
        max_t=20,
        grace_period=5,
        reduction_factor=2
    )

    # Report progress on the CLI
    reporter = CLIReporter(
        metric_columns=["epoch", "test_loss", "test_acc"]
    )

    # Run hyperparameter tuning
    analysis = tune.run(
        train_cifar,
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    print("Best config: ", analysis.get_best_config(metric="test_loss", mode="min"))
