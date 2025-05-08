#import necessary libraries 
import torch
import wandb
import csv

#train model
def train_model(model, dataloader, epochs, device):

    #initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), #initialize model parameters
        lr = 0.0001) #learning rate
    
    #initialize loss function
    criterion = torch.nn.CrossEntropyLoss()

    #initialize wandb
    wandb.init()
    #configure wandb learning rate
    wandb.config.update({"learning_rate": 0.0001})

    #set model to train model
    model.train()

    #initialize data container
    data = [['Epoch', 'Training Accuracy', 'Training Loss']] #header for csv file

    #epoch loop
    for epoch in range(epochs):
        #placeholder variables for each epoch
        epoch_loss = 0
        correct = 0
        total = 0
        
        #minibatch iteration
        for iteration, (inputs, labels) in enumerate(dataloader, 1):
        
            #move inputs
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            #reinitialize optimizer
            optimizer.zero_grad()
        
            #feed forward
            outputs = model(inputs)
        
            #compute loss function
            loss = criterion(outputs, labels)
        
            #back propogation
            loss.backward()
        
            #gradient descent step
            optimizer.step()
        
            #track loss
            epoch_loss = loss.item() + epoch_loss
    
            #calculuate accuracy
            _, predicted = torch.max(outputs,1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()

        #calculate accuracy
        accuracy = 100 * correct / total
        #calculate average loss
        avg_loss = epoch_loss / len(dataloader)

        #increment wandb
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        
        #save current iteration accuracy and loss
        data.append([epoch + 1, accuracy, avg_loss])

    #write to csv file
    with open('model_training.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerows(data)