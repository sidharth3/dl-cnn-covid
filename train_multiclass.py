
'''
Python file for the multiclass model used to train, validate and test the model to distinguish between 'normal', 'infected with no covid' and 'infected with covid' datasets. 

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

'''

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import time
import argparse
from lung_data_loader_with_transform import Lung_Dataset
from model import Three_Way_Classifier_One
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
import datetime


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


def train(model, device, train_loader, optimizer):
    '''
    Trains the model based on the inputs
    
    Parameters: 
    - model should take in a pytorch model 
    - device either cpu or gpu
    - train_loader should take in the train loader, an instance of the data loader
    - optimiser the desired opitmiser such as Adam or RMSprop
    
    Returns the training loss
    '''
    model.train()
    
    running_loss = 0
    correct = 0
    
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype = torch.int64)
        optimizer.zero_grad()
        output = model.forward(data)
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(torch.max(target,1)[1].view_as(pred)).sum().item()

        loss = F.nll_loss(output, torch.max(target,1)[1])
        loss.backward()
        optimizer.step()
        
        
        running_loss += loss.item()

    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        running_loss/len(train_loader.dataset), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return running_loss/len(train_loader.dataset)
    
            

def validate(model, device, val_loader):
    '''
    Runs the validation dataset for the model based on the inputs
    
    Parameters: 
    - model should take in a pytorch model 
    - device either cpu or gpu
    - val_loader should take in the valid loader, an instance of the data loader
    
    Returns the validation loss and accuracy 
    '''
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, torch.max(target,1)[1], reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(torch.max(target,1)[1].view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return(val_loss, correct / len(val_loader.dataset))


def test(model, device, test_loader, plot=False):
    '''
    Tests the model based on the inputs
    
    Parameters: 
    - model should take in a pytorch model 
    - device either cpu or gpu
    - test_loader should take in the test loader, an instance of the data loader
    '''
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = torch.zeros(3,3)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(torch.max(target,1)[1].view_as(pred)).sum().item()

    print('\nTest Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))    


'''Plotting Function'''
def plot_graph(graph_data, num_epochs, model):
    '''
    Plots the model based on the inputs
    
    Parameters: 
    - graph_data should take in an array of graph data of type defaultdict(list) 
    - epoch the number of epochs the model has been run for
    - model should take in a pytorch model
    
    Returns a plot
    '''
    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.title('Acc vs Epoch [Model {}]'.format(model))
    plt.plot(range(1, num_epochs+1), graph_data['val_acc'], label='val_acc')

    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()

    plt.subplot(122)
    plt.title('Loss vs Epoch [Model {}]'.format(model))
    plt.plot(range(1, num_epochs+1), graph_data['train_loss'], label='train_loss')
    plt.plot(range(1, num_epochs+1), graph_data['val_loss'], label='val_loss')
    plt.xticks((np.asarray(np.arange(1, num_epochs+1, 1.0))))

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    
    fname = 'model_{}_graph.png'.format(model)
    
    plt.savefig(fname, bbox_inches='tight')
    plt.show()    
    
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
    model = Three_Way_Classifier_One()
    
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
    

    
    
'''Run'''   
def run(device, epochs, learning_rate, batch_size, data_transform, plot=True, upsampled = False, scheduler_bool=False, decay_bool=False):
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
    
    ld_train = Lung_Dataset('train', 1, data_transform, upsampled)
    ld_test = Lung_Dataset('test', 1, data_transform, upsampled)
    ld_val = Lung_Dataset('val', 1, data_transform, upsampled)
    model = Three_Way_Classifier_One().to(device)
    print("Training the first model to classify normal, infected and covid images")
    
    train_loader = DataLoader(ld_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(ld_test, batch_size = batch_size, shuffle=True)#changed to val as it will be used to test (24 images only)
    
    if decay_bool == False: 
        optimzier = optim.Adam(model.parameters(), lr=0.01)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    
    scheduler = StepLR(optimizer, step_size=13, gamma=gamma)
    
    for epoch in range(1,epochs+1):
        print('Epoch: {}/{} : '.format(epoch, epochs))
        train_loss = train(model, device, train_loader, optimizer)
        val_loss, val_acc = validate(model, device, test_loader)
        if scheduler_bool == False:
            pass
        else: 
            scheduler.step()
        graph_data['train_loss'].append(train_loss)
        graph_data['val_loss'].append(val_loss)
        graph_data['val_acc'].append(val_acc)
        
        if epoch == epochs: #saving the model at the end
            name = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
            save_model(model, "model_" + name + ".pt")
        
    test(model, device, test_loader)
    if plot == True: 
        plot_graph(graph_data, epochs, model ='multiclass')
    
    

    
'''Main'''
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
    
    args = parser.parse_args()
   
    scheduler = args.scheduler
    decay = args.decay
    device = args.gpu
    epochs = args.epochs
    plot = args.plot
    learning_rate = args.lr
    batch_size = args.batchsize
    upsample = args.upsample
    transform = args.transform
    if transform == False: 
        data_transform = None
    run(device, epochs, learning_rate, batch_size, data_transform, plot=plot, upsampled=upsample, scheduler_bool=scheduler, decay_bool=decay)