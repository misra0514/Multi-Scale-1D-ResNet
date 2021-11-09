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
from config import configs
import argparse
import visualize

from model.multi_scale_ori import *
# from multi_scale_nores import *
# from multi_scale_one3x3 import *
# from multi_scale_one5x5 import *
# from multi_scale_one7x7 import *

parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)                       

num_classes = configs.num_classes
batch_size = configs.batch_size
num_epochs = configs.num_epochs
num_train_instances = configs.num_train_instances
num_test_instances = configs.num_test_instances
train_loss = np.zeros([num_epochs, 1])
train_acc = np.zeros([num_epochs, 1])
test_acc = np.zeros([num_epochs, 1])
learning_rate = configs.learningRate
trainingDataFileName = configs.trainingDataFileName
trainingLableFileName = configs.trainingLableFileName
pretrained_path = configs.pretrained_Path
parser.add_argument('--pretrained', dest='pretrained', default=pretrained_path,
                help='path to pre-trained model')

map_lable = configs.map_label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_res(train_loss, train_acc, test_acc):
    plt.plot(train_loss)
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.legend(['train_loss', 'train_acc'])
    plt.save('./res.png')
    # plt.show()

def load_data(datatype='train'):
    if( datatype == 'train'):
        # prepair data load with samples and lables
        # train_data = DataLoaderCsv('testData') # get data from raw file
        # train_label = DataLoaderCsv('')
        # TODO: csv--numpy--tensor感觉有点麻烦，不知道有无优化。
        # TODO: 根据数据格式，这里可能需要做一下优化，包括值的mapping
        # TODO: 这里还是使用了512个数据量的限制。感觉后续可以关掉了。
        train_data = pd.read_csv(trainingDataFileName,header=None).to_numpy()[:,0:512]
        train_data = torch.from_numpy(train_data).type(torch.FloatTensor).view(num_train_instances, 1, -1) # reshape
        train_label = pd.read_csv(trainingLableFileName,header=None).to_numpy()-1
        train_dataset = TensorDataset(train_data, torch.tensor(train_label).view(num_train_instances, 1)) # reshape data&sample
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        return train_data_loader
    elif(datatype=='test'):
        test_data = pd.read_csv(configs.testDataFileName,header=None).to_numpy()[:,0:512]
        test_data = torch.from_numpy(test_data).type(torch.FloatTensor).view(num_test_instances, 1, -1) # reshape
        test_label = pd.read_csv(configs.tesLableFileName,header=None).to_numpy()-1
        test_dataset = TensorDataset(test_data, torch.tensor(test_label).view(num_test_instances, 1)) # reshape data&sample
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=configs.batch_size, shuffle=True)
        return test_data_loader
    else:
        return None

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num=3, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(device)
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


if __name__ == '__main__':
    print('USING DEVICE' +device.type )
    args = parser.parse_args()
    model =  MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=num_classes)
    model = model.to(device)
    # init plot 
    plot_loss = visualize.line('Loss', port=8097)
    plot_loss.register_line('Loss', 'iter', 'loss')
    accuracy = {'{}'.format(x) : visualize.line('{}'.format(x), port=8097) for x in ['train_acc',  'val_acc']}
    for x in ['train_acc', 'val_acc']:
        accuracy['{}'.format(x)].register_line('{}'.format(x), 'epoch', 'accuracy')
    # FIXED: load pretrained model
    if(configs.isPreTrained):
        if(device.type=='CPU'):
            model = torch.load(configs.pretrained_Path, map_location='cpu')
        else:
            model = torch.load(configs.pretrained_Path)
    
    criterion = nn.CrossEntropyLoss(size_average=False).to(device)
    # criterion = WeightedFocalLoss().to(device)
    # criterion = FocalLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 2 load data
    print('LOADING DATA...')
    train_data_loader = load_data('train')
    test_data_loader = load_data('test')

    # train
    print('START TRAINING...')
    for epoch in range(0,num_epochs):
        loss_x = 0
        for (samples, labels) in tqdm(train_data_loader):
            samplesV = Variable(samples.to(device))
            labelsV = Variable(labels.to(device))

            model.train()
            predict_label = model(samplesV) 
            # calc loss & backward
            # FIXED: 这里报错的原因是因为model的输出有问题。调成了输出2维，会返回一个2*2的tensor（其中2 为batch，2为分类结果。是否意味着为1-hot编码？
            # FIXED: 加了一个squeeze之后变成一维向量可以运行了。是原来的数据导入不规则吗？            
            loss = criterion(predict_label[0], labelsV.squeeze())
            loss_x += loss.item()
            # TODO: 关于损失计算，这里还有一点小问题。按理说lable是会映射成0，1..这样子的。但是目前是直接读取了原数据，这样可能会导致最后的结果问题。另外loss需要按照比赛标准更改
            loss.backward()
            optimizer.step() # 更新参数
        # after each epoch; show training deatils
        model.eval()
        torch.save(model, './modelweights/'+ str(epoch) + '.pth')
        # 记录loss
        train_loss[epoch] = loss_x / num_train_instances
        plot_loss.update_line('Loss', epoch, loss_x / num_train_instances)
        # 计算正确数量(使用训练集)
        correct_train = 0
        for i, (samples, labels) in enumerate(train_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.to(device))
                labels = labels.squeeze()
                labelsV = Variable(labels.to(device))
                labelsV = labelsV.view(-1)

                predict_label = model(samplesV)
                prediction = predict_label[0].data.max(1)[1]
                correct_train += prediction.eq(labelsV.data.long()).sum()
                # loss = criterion(predict_label[0], labelsV)
        train_acc[epoch] = float(correct_train)/num_train_instances
        accuracy['train_acc'].update_line('train_acc', epoch, float(correct_train)/num_train_instances)
        # FIXED: check test dataset
        # loss_x = 0
        correct_test = 0
        for i, (samples, labels) in enumerate(test_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.to(device))
                labels = labels.squeeze()
                labelsV = Variable(labels.to(device))
                # labelsV = labelsV.view(-1)

            predict_label = model(samplesV)
            prediction = predict_label[0].data.max(1)[1]
            correct_test += prediction.eq(labelsV.data.long()).sum()

            # loss = criterion(predict_label[0], labelsV)
            # loss_x += loss.item()
        test_acc[epoch] = float(correct_test)/num_test_instances
        accuracy['val_acc'].update_line('val_acc', epoch, float(correct_test) / num_test_instances)
        print("running epoch" + str(epoch) + ", with loss " + str(train_loss[epoch])+' and acc'+str(train_acc[epoch]))
        print("Test accuracy:", test_acc[epoch])
    # plot sth in all
    # FIEXED: 改成screen 
    plot_res(train_loss,train_acc,test_acc)
    # plt.plot(train_loss)
    # plt.plot(train_acc)
    # plt.legend(['train_loss', 'train_acc'])
    # plt.show()


#----------------------------- obsolete------------------------

# creation
# class F1Loss(nn.Module):
#     def __init__(self):
#         super(F1Loss, self).__init__()

#     def forward(self, input, target, eps=1e-10):
#         loss = 0
#         for idx, i in enumerate(torch.eye(3).to(device)):
#             t = i.view(3,1)
#             y_pred_ = input.matmul(t).squeeze()
#             y_true_ = target==idx
#             loss += 0.5 * (y_true_ * y_pred_).sum() / (y_true_ + y_pred_ + eps).sum()
#         return -torch.log(loss+eps)


# def loadData():
# # 1 load data
#     data = sio.loadmat('data/changingSpeed_train.mat')
#     train_data = data['train_data_split']
#     train_label = data['train_label_split']

#     num_train_instances = len(train_data)

#     train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
#     train_label = torch.from_numpy(train_label).type(torch.LongTensor)
#     train_data = train_data.view(num_train_instances, 1, -1)
#     train_label = train_label.view(num_train_instances, 1)

#     train_dataset = TensorDataset(train_data, train_label)
#     train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



#     data = sio.loadmat('data/changingSpeed_test.mat')
#     test_data = data['test_data_split']
#     test_label = data['test_label_split']

#     num_test_instances = len(test_data)

#     test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
#     test_label = torch.from_numpy(test_label).type(torch.LongTensor)
#     test_data = test_data.view(num_test_instances, 1, -1)
#     test_label = test_label.view(num_test_instances, 1)

#     test_dataset = TensorDataset(test_data, test_label)
#     test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# # read from csv
# class DataLoaderCsv(Dataset):
#     def __init__(self, csv_path):
 
#         self.data = pd.read_csv(csv_path).to_numpy()
#         # self.labels = np.asarray(self.data.iloc[:, 0])
#         # self.height = height
#         # self.width = width
#         # self.transforms = transforms
 
#     def __getitem__(self, index): # TODO: awaiting to be amend
#         return self.data[index]
#         # single_image_label = self.labels[index]
#         # # 读取所有像素值，并将 1D array ([784]) reshape 成为 2D array ([28,28])
#         # img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28, 28).astype(float)
#         # # 把 numpy array 格式的图像转换成灰度 PIL image
#         # img_as_img = Image.fromarray(img_as_np)
#         # img_as_img = img_as_img.convert('L')
#         # # 将图像转换成 tensor
#         # if self.transforms is not None:
#         #     img_as_tensor = self.transforms(img_as_img)
#         #     # 返回图像及其 label
#         # return (img_as_tensor, single_image_label)
 
#     def __len__(self):
#         return len(self.data.index)

# # use Focal Loss for uneven data
# class WeightedFocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"    
#     def __init__(self, alpha=.25, gamma=2):
#             super(WeightedFocalLoss, self).__init__()        
#             self.alpha = torch.tensor([alpha, 1-alpha]).to(device)        
#             self.gamma = gamma
            
#     def forward(self, inputs, targets): # forward输入为真实标签
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')        
#             targets = targets.type(torch.long)        
#             at = self.alpha.gather(0, targets.data.view(-1))        
#             pt = torch.exp(-BCE_loss)        
#             F_loss = at*(1-pt)**self.gamma * BCE_loss        
#             return F_loss.mean()

# def train():
#     # 2 modeling
#     msresnet = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=6) # numclass should be 3; while input will also be amend
#     msresnet = msresnet.to(device)

#     criterion = nn.CrossEntropyLoss(size_average=False).to(device)

#     optimizer = torch.optim.Adam(msresnet.parameters(), lr=0.005)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1)
#     train_loss = np.zeros([num_epochs, 1])
#     test_loss = np.zeros([num_epochs, 1])
#     train_acc = np.zeros([num_epochs, 1])
#     test_acc = np.zeros([num_epochs, 1])

#     for epoch in range(num_epochs):
#         print('Epoch:', epoch)
#         msresnet.train()
#         scheduler.step()
#         # for i, (samples, labels) in enumerate(train_data_loader):
#         loss_x = 0
#         for (samples, labels) in tqdm(train_data_loader):
            # samplesV = Variable(samples.to(device))
#             labels = labels.squeeze()
#             labelsV = Variable(labels.to(device))

#             # Forward + Backward + Optimize
#             optimizer.zero_grad()
#             predict_label = msresnet(samplesV)

#             loss = criterion(predict_label[0], labelsV)

#             loss_x += loss.item()

#             loss.backward()
#             optimizer.step()

#         train_loss[epoch] = loss_x / num_train_instances

#         msresnet.eval()
#         # eval之后进入测试模式，否则数据会改变norm层等等的权重
#         # loss_x = 0
#         correct_train = 0
#         for i, (samples, labels) in enumerate(train_data_loader):
#             with torch.no_grad():
#                 samplesV = Variable(samples.to(device))
#                 labels = labels.squeeze()
#                 labelsV = Variable(labels.to(device))
#                 # labelsV = labelsV.view(-1)

#                 predict_label = msresnet(samplesV)
#                 prediction = predict_label[0].data.max(1)[1]
#                 correct_train += prediction.eq(labelsV.data.long()).sum()

#                 loss = criterion(predict_label[0], labelsV)
#                 # loss_x += loss.item()

#         print("Training accuracy:", (100*float(correct_train)/num_train_instances))

#         # train_loss[epoch] = loss_x / num_train_instances
#         train_acc[epoch] = 100*float(correct_train)/num_train_instances

#         trainacc = str(100*float(correct_train)/num_train_instances)[0:6]


#         loss_x = 0
#         correct_test = 0
#         for i, (samples, labels) in enumerate(test_data_loader):
#             with torch.no_grad():
#                 samplesV = Variable(samples.to(device))
#                 labels = labels.squeeze()
#                 labelsV = Variable(labels.to(device))
#                 # labelsV = labelsV.view(-1)

#             predict_label = msresnet(samplesV)
#             prediction = predict_label[0].data.max(1)[1]
#             correct_test += prediction.eq(labelsV.data.long()).sum()

#             loss = criterion(predict_label[0], labelsV)
#             loss_x += loss.item()

#         print("Test accuracy:", (100 * float(correct_test) / num_test_instances))

#         test_loss[epoch] = loss_x / num_test_instances
#         test_acc[epoch] = 100 * float(correct_test) / num_test_instances

#         testacc = str(100 * float(correct_test) / num_test_instances)[0:6]

#         if epoch == 0:
#             temp_test = correct_test
#             temp_train = correct_train
#         elif correct_test>temp_test:
#             torch.save(msresnet, 'weights/changingResnet/ChaningSpeed_Train' + trainacc + 'Test' + testacc + '.pkl')
#             temp_test = correct_test
#             temp_train = correct_train

#     sio.savemat('result/changingResnet/TrainLoss_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'train_loss': train_loss})
#     sio.savemat('result/changingResnet/TestLoss_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'test_loss': test_loss})
#     sio.savemat('result/changingResnet/TrainAccuracy_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'train_acc': train_acc})
#     sio.savemat('result/changingResnet/TestAccuracy_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'test_acc': test_acc})
#     print(str(100*float(temp_test)/num_test_instances)[0:6])

#     plt.plot(train_loss)
#     plt.show()
