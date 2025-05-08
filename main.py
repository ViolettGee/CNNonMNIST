#import necessary libraries
import torch
import torchvision

#import necessary files
import model
import training
import testing

#initialize training dataset
training_data = torchvision.datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = torchvision.transforms.ToTensor()
)

#initialize testing dataset
testing_data = torchvision.datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = torchvision.transforms.ToTensor()
)

#initialzie data loaders
train_loader = torch.utils.data.DataLoader(training_data, batch_size = 100, shuffle = True)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size = 100, shuffle = False)

#initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#initialize model
cnn = model.CNN().to(device)

#train model
training.train_model(cnn, train_loader, 600, device)

#test model
testing.test_model(cnn, test_loader, device)