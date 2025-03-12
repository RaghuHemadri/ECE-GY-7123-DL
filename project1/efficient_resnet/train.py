import yaml

import torch
import warnings
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.backends.cudnn as cudnn  

from models import EfficientResNet
from SEResNet import model
from dataset import CIFAR10PickleDataset
from optimizations import *
import os
from datetime import datetime
import matplotlib.pyplot as plt
import math

from time import time

# -------------------------------
# Training Functions
# -------------------------------
def train(model, device, train_loader, optimizer, criterion, epoch, config):
    model.train()
    running_loss = 0.0
    use_mixup = config['training'].get('mixup', False)
    mixup_alpha = config['training'].get('mixup_alpha', 1.0)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Apply Mixup if enabled
        if use_mixup:
            data, targets_a, targets_b, lam = mixup_data(data, target, mixup_alpha)
            output = model(data)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            output = model(data)
            loss = criterion(output, target)
        loss.backward()
        # Gradient clipping if configured
        clip_val = config['model'].get('gradient_clip', None)
        if clip_val is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)
    return avg_loss

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
                             [0.2023, 0.1994, 0.2010])
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
    # model = EfficientResNet(config).to(device)
    model = model(config).to(device)
    print("Model Summary:")
    summary(model, (3, 32, 32))

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    if ("weights_init_type" in config): 
        def init_weights(m, type='default'): 
            if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)) and hasattr(m, 'weight'): 
                if type == 'xavier_uniform_': torch.nn.init.xavier_uniform_(m.weight.data)
                elif type == 'normal_': torch.nn.init.normal_(m.weight.data, mean=0, std=0.02)
                elif type == 'xavier_normal': torch.nn.init.xavier_normal(m.weight.data, gain=math.sqrt(2))
                elif type == 'kaiming_normal': torch.nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
                elif type == 'orthogonal': torch.nn.init.orthogonal(m.weight.data, gain=math.sqrt(2))
                elif type == 'default': pass 
        model.apply(lambda m: init_weights(m=m, type=config["weights_init_type"])) 

    if config['training']['resume_training']:
        run_id = config['training']['run_id']
        model_dir = os.path.join("trained_models", run_id)
        checkpoint_epoch = config['training']['checkpoint_epoch']
        model.load_state_dict(torch.load(os.path.join(model_dir, f"model_checkpoint_epoch_{checkpoint_epoch}.pth")))
        print(f"Resuming training from epoch {checkpoint_epoch}")

    else:
        # Create directory for saving models
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join("trained_models", run_id)
        os.makedirs(model_dir, exist_ok=True)
        print(f"Saving models to directory: {model_dir}")
        checkpoint_epoch = 0
    
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

    # Save configuration
    with open(os.path.join(model_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    train_losses = []
    test_losses = []

    for epoch in range(checkpoint_epoch + 1, checkpoint_epoch + config['training']['epochs'] + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, config)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
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
