import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pillow
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import os

def plot_graph(graph_data, num_epochs, model):
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
    
    fname = 'model{}_graph.png'.format(model)
    
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    print('plot saved as {}\n\n'.format(fname))
    
def plot_bar(dataset_type):
    assert dataset_type == 'normal' or dataset_type == 'upsample'
    if dataset_type == 'normal':
        normal_train = len(os.listdir('./dataset/train/normal/'))
        normal_test = len(os.listdir('./dataset/test/normal/'))
        covid_train = len(os.listdir('./dataset/train/infected/covid'))
        covid_test = len(os.listdir('./dataset/test/infected/covid'))
        noncovid_train = len(os.listdir('./dataset/train/infected/non-covid'))
        noncovid_test = len(os.listdir('./dataset/test/infected/non-covid'))
    else:
        normal_train = len(os.listdir('./dataset_upsampling/train/normal/'))
        normal_test = len(os.listdir('./dataset_upsampling/test/normal/'))
        covid_train = len(os.listdir('./dataset_upsampling/train/infected/covid'))
        covid_test = len(os.listdir('./dataset_upsampling/test/infected/covid'))
        noncovid_train = len(os.listdir('./dataset_upsampling/train/infected/non-covid'))
        noncovid_test = len(os.listdir('./dataset_upsampling/test/infected/non-covid'))
        
    classes = ['normal', 'covid', 'non-covid']
    train = [normal_train, covid_train, noncovid_train]
    test = [normal_test, covid_test, noncovid_test]
    x = np.arange(len(classes))
    barWidth = 0.4

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - barWidth/2, train, barWidth, label='train')
    rects2 = ax.bar(x + barWidth/2, test, barWidth, label='test')

    ax.set_ylabel('count')
    ax.set_title('Class distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()

    fig.tight_layout()
    plt.show()