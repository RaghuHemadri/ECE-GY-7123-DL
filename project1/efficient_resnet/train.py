import yaml

import torch
import warnings
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary

from models import EfficientResNet
from dataset import CIFAR10PickleDataset
from optimizations import *
import os
from datetime import datetime
import matplotlib.pyplot as plt

from time import time

# -------------------------------
# Training Functions
# -------------------------------
def train(model, device, train_loader, optimizer, criterion, epoch, config, scaler):
    model.train()
    running_loss = 0.0
    use_mixup = config['training'].get('mixup', False)
    mixup_alpha = config['training'].get('mixup_alpha', 1.0)
    use_mixed_precision = config['training'].get('use_mixed_precision', False)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Apply Mixup if enabled
        if use_mixup:
            data, targets_a, targets_b, lam = mixup_data(data, target, mixup_alpha)
        
        with torch.amp.autocast(device_type='cuda', enabled=use_mixed_precision):
            if use_mixup:
                output = model(data)
                loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping if configured
        clip_val = config['model'].get('gradient_clip', None)
        if clip_val is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss

def evaluate(model, device, test_loader, criterion, use_mixed_precision):
    model.eval()
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.amp.autocast(device_type='cuda', enabled=use_mixed_precision):
                output = model(data)
                loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return avg_loss, accuracy

# -------------------------------
# Main Function
# -------------------------------
def main():
    # Load configuration
    start_time = time()
    warnings.filterwarnings("ignore")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up transforms
    train_transform = get_transforms(config)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616])
    ])

    # Create datasets and loaders
    data_dir = config['data']['cifar10_dir']
    train_dataset = CIFAR10PickleDataset(data_dir=data_dir, train=True, transform=train_transform)
    test_dataset = CIFAR10PickleDataset(data_dir=data_dir, train=False, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=config['training']['num_workers'], pin_memory=config['training']['pin_memory'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                             shuffle=False, num_workers=config['training']['num_workers'], pin_memory=config['training']['pin_memory'])
    
    # Initialize model
    model = EfficientResNet(config).to(device)
    print("Model Summary:")
    summary(model, (3, 32, 32))
    
    # Define optimizer
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
    
    # Learning rate scheduler
    if config['training']['lr_scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    # Create directory for saving models
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("trained_models", run_id)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Saving models to directory: {model_dir}")

    # Save configuration
    with open(os.path.join(model_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    use_mixed_precision = config['training'].get('use_mixed_precision', False)
    scaler = torch.amp.GradScaler(enabled=use_mixed_precision, device='cuda')
    
    train_losses = []
    test_losses = []

    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, config, scaler)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion, use_mixed_precision)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        if scheduler is not None:
            scheduler.step()
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
        # Save model checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(model_dir, f"model_checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
        
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pth"))

    end_time = time()
    print(f"Training took {end_time - start_time:.2f} seconds")

    # Plotting the losses
    plt.figure()
    plt.plot(range(1, config['training']['epochs'] + 1), train_losses, label='Train Loss')
    plt.plot(range(1, config['training']['epochs'] + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Test Losses')
    plt.savefig(os.path.join(model_dir, 'loss_plot.png'))
    plt.close()

    # Write best losses and accuracy to metrics.txt
    with open(os.path.join(model_dir, "metrics.txt"), "w") as f:
        f.write(f"Best Test Accuracy: {best_acc:.2f}%\n")
        f.write(f"Final Train Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final Test Loss: {test_losses[-1]:.4f}\n")
        f.write(f"Training Time: {end_time - start_time:.2f} seconds\n")
        # Write model summary to metrics.txt
        f.write("\nModel Summary:\n")
        summary_str = summary(model, (3, 32, 32), verbose=0)
        f.write(str(summary_str))

    print(f"Best Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
