import os

class Config:
    num_classes = 3
    batch_size = 32
    num_epochs = 100
    num_train_instances = 189079 # 30 # 189079
    num_test_instances = 40000
    learningRate = 0.0005 # 0.0005
    isPreTrained = True
    pretrained_Path ='pretrained/ptm1.pth'
    trainingDataFileName = '../dataset/train_512_data.csv' # '../dataset/30data.csv' # '../dataset/train_512_data.csv'# '../dataset/testData.csv'
    trainingLableFileName = '../dataset/train_512_lable.csv' # '../dataset/30label.csv' # '../dataset/train_512_lable.csv'# '../dataset/testLable.csv'
    testDataFileName = '../dataset/test_512_data.csv'
    tesLableFileName = '../dataset/test_512_lable.csv'

    def map_label(data): # 1-2-3meaning？
        if data == '1':
            return 0
        elif data == '2':
            return 1
        elif data == '3':
            return 2


configs = Config()