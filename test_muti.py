# function: test mutimodels from one floder

from threading import Condition
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import pandas as pd
from tqdm import tqdm
import argparse
import visualize

from model.multi_scale_ori import *
from config import configs
from demo import load_data

Object_Path = './modelweights/'
model_number = 90
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# test_acc = np.zeros([model_number+1, 1])
num_test_instances = configs.num_test_instances

if __name__ == '__main__':
    plot_loss = visualize.line('test_muti', port=8097)
    plot_loss.register_line('test_muti', 'model', 'acc')
    # 1 load data 
    test_data_loader = load_data('test')
    # 2 load model
    model =  MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=configs.num_classes)
    model = model.to(device)
    # 3 test
    for index in  range(0,model_number-2):
        model = torch.load(Object_Path+str(index)+'.pth')
        model.eval()
        correct_test = 0
        for i, (samples, labels) in enumerate(test_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.to(device))
                labels = labels.squeeze()
                labelsV = Variable(labels.to(device))
                labelsV = labelsV.view(-1)

            predict_label = model(samplesV)
            prediction = predict_label[0].data.max(1)[1]
            correct_test += prediction.eq(labelsV.data.long()).sum()

            # loss = criterion(predict_label[0], labelsV)
            # loss_x += loss.item()
        # test_acc[i] = float(correct_test)/num_test_instances
        plot_loss.update_line('test_muti', index, float(correct_test)/num_test_instances)
        # accuracy['val_acc'].update_line('val_acc', epoch, float(correct_test) / num_test_instances)
        print("Test accuracy for ", index," is ", float(correct_test)/num_test_instances)
