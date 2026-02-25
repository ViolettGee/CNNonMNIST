# CNNonMNIST
* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install Packages](#22-install-packages)  
  * [2.3 Run CNN](#23-run-cnn)  
* [3 Results](#3-results)  

---

# **1. Introduction**  
This project implements a custom Convolutional Neural Network using PyTorch for the handwritten MNIST dataset. The objective was to design and train a CNN architecture from scratch rather than relying on predifined models. Implementation included: batch normalization, dropout regularization, max pooling and residual (skip-connection) stages inspired by ResNet. Experiments were tracked using Weights & Biases.

# **2. Getting Started**
This project requires Python 3.10+ and runs on macOS, Linux, and Windows.

## **2.1 Preparations**
(1) Clone the repository to your workspace:
```shell
~ $ git clone https://github.com/ViolettGee/CNNonMNIST.git
```

## **2.2 Install Packages**
(2) Install the required dependencies:
```shell
~ $ pip install torch
~ $ pip install torchvision
~ $ pip install wandb
~ $ pip install pandas
```

## **2.3 Run CNN**
The entire model can be run by executing the "main.py" file and the other files "model.py", "training.py" and "testing.py" are called within that file. The file first loads the MNIST dataset into training and testing datasets for further use. GPU accelaeration is recommended but not required.

The model framework is then initalized by calling the "CNN" class in "model.py".
  - "model.py" The file initializes a model in a custom framework as defined below:
      - input: 28 by 28
      - Convolution layer
      - Batch normalization
      - Convolution layer
      - Batch normalization
      - ReLU
      - Dropout
      - Convolution layer
      - Residual block (Skip connection)
      - Flatten
      - Fully connected layer
      - ReLU
      - Dropout
      - Output: 10 classes (digits 0-9)

The model is then trainined using the training data and above model framework in "training.py".
  - "training.py" The file trains the model tracking the trends using Weights & Biases, and writing epoch metrics (loss an accuracy) to the "model_training.csv" file.

The model is then tested using the testing data and trained model in "testing.py".
  - "testing.py" The file evaluates the trained model tracking the trends and writing epoch metrics (accuracy) to the "model_testing.csv" file.

# **3. Results**
The resutling model had achieved a test accuracy of 92.4. Training progres, including epoch-level loss and accuacy metrics is avaliable in "model_training.csv". And evaluation predictions and epoch accuacy metrics are stored in "model_testing.csv".

A detailed technical report can be found in "Assignmet 3.docx".
