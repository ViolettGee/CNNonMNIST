#import necessary libraries\n",
import torch

#initialize ResNet18
class ResNet(torch.nn.Module):

    #constructor class
    def __init__(self, channels):
        super(ResNet, self).__init__()

        #matrix input = 8 x 8\n",
    
        #convolution layer\n",
        self.conv1 = torch.nn.Conv2d(
            in_channels = channels, #input channel size
            out_channels = channels, #output channel size
            kernel_size = 3,
            stride = 1,
            padding = 1,
        ) #matrix output = 8 x 8\n",
        
        #batch normalization layer
        self.batchnorm1 = torch.nn.BatchNorm2d(channels)
        
        #relu layer
        self.relu = torch.nn.ReLU()
        
        #convolution layer
        self.conv2 = torch.nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ) #matrix output = 8 x 8
        
        #batch normalization layer
        self.batchnorm2 = torch.nn.BatchNorm2d(channels)
    
    #forward function
    def forward(self, x):
        
        #layer 1
        out = self.relu(self.batchnorm1(self.conv1(x))) 
        #layer 2
        out = self.batchnorm2(self.conv2(out))
        
        #update with input
        return out + x
        
#initialzie CNN class
class CNN(torch.nn.Module):
    
    #constructor class
    def __init__(self):
        super(CNN, self).__init__()

        #input size = 28 x 28

        #convolution layer 1
        self.conv1 = torch.nn.Conv2d(
            in_channels = 1, #chosen due to gray-scale
            out_channels = 8, #number of hyper-parameters
            kernel_size = 3, #kernel size
            stride = 1, #step size
            padding = 2 #padding size
        ) #matrix output = 30 x 30

        #batch normalization layer 1
        self.batchnorm1 = torch.nn.BatchNorm2d(8) #input based on convolution output
                
        #convolution layer 2
        self.conv2 = torch.nn.Conv2d(
            in_channels = 8, #chosen due to gray-scale
            out_channels = 16, #number of hyper-parameters
            kernel_size = 2, #kernel size
            stride = 2, #step size
            padding = 1 #padding size
        ) #matrix output = 16 x 16

        #batch normalization layer 2
        self.batchnorm2 = torch.nn.BatchNorm2d(16) #input based on convolution output

        #RELU layer 1
        self.relu1 = torch.nn.ReLU()

        #dropout layer 1
        self.dropout1 = torch.nn.Dropout(p = 0.25)
        #probability to optimize between preventing overfitting and analysis

        #convolution layer 3
        self.conv3 = torch.nn.Conv2d(
            in_channels = 16, #chosen due to gray-scale
            out_channels = 32, #number of hyper-parameters
            kernel_size = 3, #kernel size
            stride = 1, #step size
            padding = 1 #padding size
        ) #matrix output = 16 x 16

        #maxpool layer
        self.maxpool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2) #matrix output = 8 x 8

        #residual block call
        self.residual = ResNet(32)

        #flatten layer
        self.flatten = torch.nn.Flatten()

        #fully connected layer 1
        self.fullconnect1 = torch.nn.Linear(
            32 * 8 * 8, #32 weights, 8 x 8 matrix
            128) #hidden layer size

        #RELU layer 2
        self.relu2 = torch.nn.ReLU()
        
        #dropout layer 2
        self.dropout2 = torch.nn.Dropout(p = 0.25)
        #probability to optimize between preventing overfitting and analysis

        #fully connected layer 2
        self.fullconnect2 = torch.nn.Linear(
            128, #hidden layer size
            10) #output size

    #forward function
    def forward(self, x):
        
        #convolution layers
        out = self.batchnorm1(self.conv1(x)) #layer 1
        out = self.dropout1(self.relu1(self.batchnorm2(self.conv2(out)))) #layer 2
        out = self.conv3(out) #layer 3
        
        #max pool
        out = self.maxpool(out)

        #residual stages
        out = self.residual(out)
        
        #flatten
        out = self.flatten(out)
        
        #fully connected layers
        out = self.dropout2(self.relu2(self.fullconnect1(out))) #layer 1
        out = self.fullconnect2(out) #layer 2

        return out