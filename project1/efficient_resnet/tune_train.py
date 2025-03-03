import os
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from efficient_resnet.models import *
from efficient_resnet.dataset import *
from efficient_resnet.optimizations import *

# -------------------------------
# Training and Evaluation Functions (adapted for Ray Tune reporting)
# -------------------------------
def train_epoch(model, device, train_loader, optimizer, criterion, config):
    model.train()
    running_loss = 0.0
    use_mixup = config['training'].get('mixup', False)
    mixup_alpha = config['training'].get('mixup_alpha', 1.0)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if use_mixup:
            data, targets_a, targets_b, lam = mixup_data(data, target, mixup_alpha)
            output = model(data)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            output = model(data)
            loss = criterion(output, target)
        loss.backward()
        clip_val = config['model'].get('gradient_clip', None)
        if clip_val is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, device, test_loader, criterion):
    model.eval()
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return avg_loss, accuracy

# -------------------------------
# Ray Tune Training Function
# -------------------------------
def train_cifar(config, checkpoint_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = config['data']['cifar10_dir']
    train_transform = (config)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616])
    ])
    train_dataset = CIFAR10PickleDataset(data_dir=data_dir, train=True, transform=train_transform)
    test_dataset = CIFAR10PickleDataset(data_dir=data_dir, train=False, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=int(config['training']['batch_size']),
                              shuffle=True, num_workers=int(config['training']['num_workers']))
    test_loader = DataLoader(test_dataset, batch_size=int(config['training']['batch_size']),
                             shuffle=False, num_workers=int(config['training']['num_workers']))
    model = EfficientResNet(config).to(device)
    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pth"))
        model.load_state_dict(checkpoint)
    if config['training']['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), 
                              lr=config['training']['learning_rate'],
                              momentum=config['training']['momentum'],
                              weight_decay=config['training']['weight_decay'])
    elif config['training']['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), 
                               lr=config['training']['learning_rate'],
                               weight_decay=config['training']['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), 
                              lr=config['training']['learning_rate'],
                              momentum=config['training']['momentum'],
                              weight_decay=config['training']['weight_decay'])
    if config['training']['lr_scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config['training']['epochs']))
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, int(config['training']['epochs']) + 1):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion, config)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        # Report metrics to Ray Tune
        tune.report(loss=test_loss, accuracy=test_acc)
        if scheduler is not None:
            scheduler.step()
    # Optionally, save a checkpoint
    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint.pth")
        torch.save(model.state_dict(), path)

# -------------------------------
# Main function for Ray Tune
# -------------------------------
if __name__ == "__main__":
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    # Load default config from YAML
    with open("config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # Define the search space
    config = {
        "model": {
            "name": "EfficientResNet",
            "use_bottleneck": base_config['model'].get("use_bottleneck", False),
            "channels": base_config['model']['channels'],
            # You can also tune num_blocks if desired (here fixed)
            "num_blocks": base_config['model']['num_blocks'],
            "squeeze_and_excitation": base_config['model'].get("squeeze_and_excitation", True),
            "use_dropout": base_config['model'].get("use_dropout", True),
            "dropout_rate": tune.uniform(0.0, 0.5),
            "gradient_clip": base_config['model'].get("gradient_clip", 0.1),
        },
        "training": {
            "optimizer": base_config['training'].get("optimizer", "SGD"),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "momentum": tune.uniform(0.8, 0.99),
            "weight_decay": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([64, 128, 256]),
            # For tuning, we use a smaller number of epochs (e.g. 50)
            "epochs": 50,
            "lr_scheduler": base_config['training'].get("lr_scheduler", "CosineAnnealingLR"),
            "data_augmentation": base_config['training'].get("data_augmentation", True),
            "normalization": base_config['training'].get("normalization", True),
            "mixup": tune.choice([True, False]),
            "mixup_alpha": tune.uniform(0.2, 2.0),
            "cutout": tune.choice([True, False]),
            "autoaugment": tune.choice([False, True]),
            "num_workers": base_config['training'].get("num_workers", 4)
        },
        "data": {
            "cifar10_dir": base_config['data']['cifar10_dir']
        }
    }

    # Use ASHA scheduler for early stopping of bad trials
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=50,
        grace_period=10,
        reduction_factor=2)

    # Run Ray Tune
    analysis = tune.run(
        train_cifar,
        resources_per_trial={"cpu": 4, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        name="tune_cifar"
    )

    print("Best hyperparameters found were: ", analysis.best_config)
