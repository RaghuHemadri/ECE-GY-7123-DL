import os
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import pickle
from models import EfficientResNet
from torchvision import models

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

def main():
    model_time = "20250303_200119"
    model_dir = f"trained_models/{model_time}"
    # Load configuration
    with open(os.path.join(model_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test batch
    # cifar10_batch = load_cifar_batch('/kaggle/input/deep-learning-spring-2025-project-1/cifar_test_nolabel.pkl')
    cifar10_batch = load_cifar_batch('../dataset/cifar_test_nolabel.pkl')
    images = cifar10_batch[b'data']

    # Define the test transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])

    # Initialize model
    model = EfficientResNet(config).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "final_model.pth")))

    # model_name = 'resnet50'
    # if model_name == 'resnet50':
    #     model = models.resnet50(pretrained=True)
    # elif model_name == 'resnet101':
    #     model = models.resnet101(pretrained=True)
    # else:
    #     raise ValueError(f"Unsupported model name: {model_name}")

    # model = model.to(device)

    model.eval()

    # Perform inference
    ids = []
    labels = []
    with torch.no_grad():
        for idx, img in enumerate(images):
            img = test_transform(img).unsqueeze(0).to(device)
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True).item()
            ids.append(idx)
            labels.append(pred)

    # Save predictions to CSV
    df = pd.DataFrame({'ID': ids, 'Labels': labels})
    df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False)

if __name__ == "__main__":
    main()
