
'''
Python file for the binary model used to train, validate and test the model to distinguish between 'normal', 'infected with no covid' and 'infected with covid' datasets. 

Parameters
- epochs the number of epochs you want to train the model for 
- gpu whether you would like to utilise gpu 
- lr the learning rate you would like to utilise for training the model 
- batchsize the batchsize you would like to utilise for training the model
- plot, a boolean whether you would like a plot at the end of the training and validation
- upsample, a boolean whether you would like to utilise the upsampled dataset
- transform, a boolean whether you would like to use the tranformations on the dataset
- decay, a boolean whether you would like to apply the weight decay on the model's weight 
- scheduler, a boolean whether you would like to utilise the learning rate scheduler
- sample, loads the sample model that will be tested on a test dataset

'''
import torch
from torch.utils.data import Dataset, DataLoader
# from pytorch_lightning.metrics.classification import ConfusionMatrix
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import time
from lung_data_loader_with_transform import Lung_Dataset
from model import Binary_Classifier_One
from model import Binary_Classifier_Two
from plots import plot_graph
import numpy as np
import argparse
from matplotlib import pyplot as plt
from collections import defaultdict
import datetime
#TODO FOR Tomorrow
# refractor code from main() to __main__
# take model type, plot=true/false, criterion (still need to check other loss), optimizer(with Step LR) as args
#change dataloader acc to model type
# check weight initialization patterns, xavier and kaiming

# n_epochs = args.epochs 

#model = Binary_Classifier_One()
#model = Binary_Classifier_Two()

# ld_train = Lung_Dataset('train', 0)
# ld_test = Lung_Dataset('test', 0)
# ld_val = Lung_Dataset('val', 0)

'''Transformation to be Passed '''
data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(1),
#         transforms.ColorJitter(brightness=0, contrast=0, saturation=0.25, hue=0),
        transforms.RandomAffine(0, translate=None, scale=[0.7, 1.3], shear=None, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],
                             std=[0.229])
    ])


def train(model, device, train_loader, optimizer, epoch, criterion):
    '''
    Trains the model based on the inputs
    
    Parameters: 
    - model should take in a pytorch model 
    - device either cpu or gpu
    - train_loader should take in the train loader, an instance of the data loader
    - optimiser the desired opitmiser such as Adam or RMSprop
    - epoch the current epoch that is being trained
    - critierion the learning criterion 
    
    Returns the training loss
    '''
    model.train()
    running_loss = 0
        
    for batch, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model.forward(data)
        target = target.argmax(dim=1, keepdim=True).float()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
                
        if (batch + 1) % 100 == 0:
            return running_loss/100
        
def validate(model, device, val_loader, criterion):
     '''
    Runs the validation dataset for the model based on the inputs
    
    Parameters: 
    - model should take in a pytorch model 
    - device either cpu or gpu
    - train_loader should take in the train loader, an instance of the data loader
    - optimiser the desired opitmiser such as Adam or RMSprop
    - epoch the current epoch that is being trained
    - critierion the learning criterion 
    
    Returns the validation loss and accuracy
    '''
    model.eval()

    correct = 0
    val_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            target = target.argmax(dim=1, keepdim=True).float()
            
            output = model(data)
            val_loss += criterion(output, target).item()
            
            pred = torch.round(output)
            equal_data = torch.sum(target.data == pred).item()
            correct += equal_data
    
    return (val_loss / len(val_loader)), (100. * correct / len(val_loader.dataset))
    
def test(model, device, test_loader, plot=False):
    '''
    Tests the model based on the inputs
    
    Parameters: 
    - model should take in a pytorch model 
    - device either cpu or gpu
    - test_loader should take in the test loader, an instance of the data loader
    '''
    
    model.eval()
    
    correct = 0
    confusion_matrix = torch.zeros(2, 2)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            target = target.argmax(dim=1, keepdim=True).float()
            
            output = model(data)
#             print(output)
            pred = torch.round(output)

            equal_data = torch.sum(target.data == pred).item()
            correct += equal_data
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print(confusion_matrix)

    print('Test set accuracy: ', 100. * correct / len(test_loader.dataset), '%')
    
def visualize(model, val_loader):
    '''
    Visualises the data in the validation dataset 
    
    Parameters 
    - model should take in a pytorch model 
    - val_loader should take in the valid loader 
    '''
    for data, target in val_loader:
        print(type(data))
        break
    
    
def save_model(model, path, test=True):
    '''
    Saves the model at a desired point in time 
    
    Parameters 
    - model should take in a pytorch model 
    - path the path to save the model 
    
    '''
    if test == False:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, path)
    else:
        torch.save(model.state_dict(), path)

    
# Define function to load model
def load_model(path, test=True):
    '''
    Loads a previously saved model and runs it 
    
    Parameters 
    - path where the saved model is stored
    - test whether the model will be used for testing
    '''
    model = Binary_Classifier_One()
    
    if test == False:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.train()
        
    else:
        model.load_state_dict(torch.load(path))
        model.eval()
    
    return model

    
def run(classifier, device, epochs, learning_rate, batch_size, data_transform,  plot=True, upsample=False, scheduler_bool=False, decay_bool=False):
    '''
    The main function that calls the train, valid and test functions 
    
    Parameters  
    - device either cpu or gpu
    - epochs the number of epochs you would like to run the training of the model for (integer)
    - learning rate the learning rate you would like to set for the model 
    - batch_size the size of each batch size that will be used for training
    - data_transform takes in a list of torchvision transforms to compose 
    - plot a boolean input, whether or not a plot will be displayed upon completion of training
    - upsampled a boolean input, whether you would like to utilise the upsampled data
    - scheduler_bool a boolean input, whether or not you would like to utilise the scheduler 
    - decay_bool a boolean value, whether you would like to utilise weight decay on the parameters 
    '''
    
    weight_decay = 1e-4
    gamma = 0.9 #'Learning rate step gamma (default: 0.7)')

    graph_data = defaultdict(list)    
    
    if classifier == 1:
        ld_train = Lung_Dataset('train', 0, data_transform, upsample)
        ld_test = Lung_Dataset('test', 0, data_transform, upsample)
        model = Binary_Classifier_One().to(device)
        print("Training first classifier(between normal and infected images):\n")

        
    elif classifier == 2:
        ld_train = Lung_Dataset('train', 2, data_transform, upsample)
        ld_test = Lung_Dataset('test', 2, data_transform, upsample)
        model = Binary_Classifier_Two().to(device)
        print("Training first classifier(between covid and non-covid images):\n")


    train_loader = DataLoader(ld_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ld_test, batch_size=batch_size, shuffle=True)
    
    if decay_bool == False: 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    scheduler = StepLR(optimizer, step_size=13, gamma=gamma)

    for epoch in range(1,epochs+1):
        
        train_loss = train(model, device, train_loader, optimizer, epoch, nn.BCELoss())
        val_loss, val_acc = validate(model, device, test_loader, nn.BCELoss())
        if scheduler_bool == False:
            pass
        else: 
            scheduler.step()
#         test(fl_model, device, fl_test_loader)
        print("Epoch: {}/{} @ {} \n".format(epoch, epochs, datetime.datetime.now()),
                      "Training Loss: {:.3f} - ".format(train_loss),
                      "Validation Loss: {:.3f} - ".format(val_loss),
                      "Validation Accuracy: {:.3f}".format(val_acc))
        
        graph_data['train_loss'].append(train_loss)
        graph_data['val_loss'].append(val_loss)
        graph_data['val_acc'].append(val_acc)
        
        if epoch%10 == 0 or epoch == epochs: #saving the model at regular intervals and at the end
            name = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
            save_model(model, "model_" + name + ".pt")

    print("\n\n")
    print("Test Accuracy of model {}:".format(classifier))
    test(model, device, test_loader)
    
    if plot == True:  
        plot_graph(graph_data, epochs, model = '_classifier'+str(classifier))

'''
Details the arguments that will be taken in while running the python file. The inputs for this file have been listed at the beginning of the train_multiclass.py file
'''                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train binary image classifier model")
    parser.add_argument("--epochs", type=int, default=1, help="set epochs")
    parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
    parser.add_argument("--lr", type=float, default=0.001, help="set learning rate")
    parser.add_argument("--batchsize", type=int, default=32, help="set batch size")
    parser.add_argument("--plot", type=bool, default=True, help="plot loss-acc graphs")
    parser.add_argument("--upsample", type=bool, default=False, help="dataset upsampling")
    parser.add_argument("--transform", type=bool, default=False, help="dataset transformation")
    parser.add_argument("--decay", type=bool, default=False, help="dataset transformation")
    parser.add_argument("--scheduler", type=bool, default=False, help="dataset transformation")
    parser.add_argument("--sample", type=bool, default=False, help="use the sample dataset")

    args = parser.parse_args()
    
    scheduler = args.scheduler
    decay = args.decay
    device = args.gpu
    epochs = args.epochs
    plot = args.plot
    learning_rate = args.lr
    batch_size = args.batchsize
    upsample = args.upsample
    transformation = args.transform
    sample = args.sample
    
    if sample == False:
        #Checks if transform has been selected by user before feeding it into run
        if transformation == False: 
            data_transform = None

        run(1, device, epochs, learning_rate, batch_size, data_transform, plot=plot, upsample=upsample, scheduler_bool=scheduler, decay_bool=decay)
        run(2, device, epochs, learning_rate, batch_size, data_transform, plot=plot, upsample=upsample, scheduler_bool=scheduler, decay_bool=decay)
    
    else:
        model = load_model('model_2021_03_21-07:34:47_AM.pt')
        model.to(device)
        ld_test = Lung_Dataset('test', 0, data_transform)
        test_loader = DataLoader(ld_test, batch_size=batch_size, shuffle=True)
        test(model, device, test_loader)
          
          
