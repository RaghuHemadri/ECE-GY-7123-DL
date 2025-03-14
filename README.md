# ECE-GY-7123-DL

---

## **Overview**  
This project implements a **modified ResNet architecture** for **CIFAR-10 classification**, focusing on maximizing accuracy while keeping the total number of trainable parameters **under 5 million**. The model is optimized with **Depthwise Separable Convolutions, Squeeze-and-Excitation (SE) Blocks, and various regularization techniques** to improve efficiency and generalization.  

The training pipeline integrates **AutoAugment, Mixup, Cutout, Cosine Annealing LR Scheduler, and Gradient Clipping**, ensuring robust learning and stable convergence.  

---

## **Dataset**  
The model is trained and evaluated on the **CIFAR-10 dataset**, which consists of **60,000 32×32 color images** divided into **10 classes**:
- Training set: **50,000 images**  
- Test set: **10,000 images**  

Preprocessing includes **normalization**, **random cropping**, **horizontal flipping**, and **other augmentation techniques** to improve generalization.

---

## **Architecture**  
The model follows a **modified ResNet design** with the following enhancements:  
**Depthwise Separable Convolutions** – Reduces computational complexity. This was not used in the final model due to its inability to produce better results.
**Squeeze-and-Excitation (SE) Blocks** – Improves feature recalibration.  
**Residual Connections** – Helps with gradient flow and training deeper networks.  
**Dropout Regularization** – Prevents overfitting.  
**Gradient Clipping** – Stabilizes training.  

The architecture consists of **three main residual stages** with increasing feature map sizes and a **final fully connected classifier**.

---

## **Installation**  
To set up the environment, install the required dependencies:  
```bash
pip install torch torchvision numpy matplotlib pyyaml
```

---

## **Usage**  
### **1. Dataset Preparation**  
Download the **CIFAR-10 dataset** and place it in the `data/` directory. The dataset should contain pickle files (e.g., `data_batch_1`, `data_batch_2`, ..., `test_batch`).  

### **2. Training the Model**  
To train the model, run:  
```bash
python train.py
```
This will automatically:  
- Load the dataset  
- Apply preprocessing and augmentation  
- Train the model with the selected optimizer and scheduler  
- Save model checkpoints and training logs  


## **Configuration**  
The training setup is configured via `config.yaml`. Key parameters include:  
```yaml
training:
  epochs: 100
  batch_size: 128
  optimizer: "SGD"
  learning_rate: 0.1
  lr_scheduler: "CosineAnnealingLR"
  mixup: True
  mixup_alpha: 1.0
  normalization: True
  autoaugment: True
  cutout: True
```
Modify this file to adjust hyperparameters.

---

## **Results & Performance Tracking**  
✔ **Loss and Accuracy Plots** – Training and validation loss curves are automatically saved as `loss_plot.png`.  To generate the loss and accuracy plot using the best model, the `plots.py` file could be used by running the command
```bash
python plots.py
```
✔ To check the results of the best model on any other test data set, or to generate a predictions csv file.
```
python infer.py
```

✔ **Best Model Checkpoint** – The model with the highest accuracy is stored in `trained_models/`.  

---

## **Acknowledgments**  
This project is inspired by **ResNet architectures** and techniques used in deep learning literature. If you use parts of this code, please provide proper citations.  

---
