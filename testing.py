#import necessary libraries
import torch
import csv

#model testing
def test_model(model, dataloader, device):

    #set model to test mode
    model.eval()

    #initialize data container
    data = [['Label', 'Prediction']]

    #no gradient tracking

    #iterate through data loader
    with torch.no_grad():
        for x, labels in dataloader:

            #move inputs
            x = x.to(device)
            labels = labels.to(device)
    
            #run model
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
    
            #add data to container
            for i in range(len(labels)):
                data.append([labels[i].item(), predicted[i].item()])
        

    #save data to csv
    with open('model_testing.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)