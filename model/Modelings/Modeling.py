import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchbearer
from torchbearer import Trial
import pkbar
import numpy as np
import random
import json
import datetime
import configparser

# image file truncated error prevention
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PyTorchData():
    def __init__(self, _dataType, config):
        '''
        input_dim : image size
        data_path : training data path
        batch_size : batch size
        '''

        if _dataType == "data":
            self.m_DataDim = int(config['input_dim'])
        elif _dataType == "image":
            self.m_ImageDim = int(config['input_dim'])
            self.m_DataPath = config['data_path']
            self.m_BatchSize = int(config['batch_size'])


    def ImageTrain(self):
        transDatagen = transforms.Compose([transforms.Resize((self.m_ImageDim, self.m_ImageDim)),
                                           transforms.ToTensor()])

        '''
        directory 변경 시 modeling_config.ini 파일 및 경로 수정 필요
        '''
        trainPath = self.m_DataPath + '/training'
        #trainPath = self.m_DataPath + '/cropped'
        trainFolder = torchvision.datasets.ImageFolder(root = trainPath,
                                                       transform = transDatagen)

        trainLoader = DataLoader(trainFolder,
                                batch_size = self.m_BatchSize,
                                shuffle = True)

        print("Train Class [", trainLoader.dataset.class_to_idx, "]")
        print("Train Numbers [", len(trainLoader.dataset.imgs), "]")
        print("Train Batch Size [", trainLoader.batch_size, "]")

        return trainLoader


    def ImageValidation(self):
        transDatagen = transforms.Compose([transforms.Resize((self.m_ImageDim, self.m_ImageDim)),
                                           transforms.ToTensor()])

        '''
        directory 변경 시 modeling_config.ini 파일 및 경로 수정 필요
        '''
        validationPath = self.m_DataPath + '/validation'
        #validationPath = self.m_DataPath + '/original'
        validationSet = torchvision.datasets.ImageFolder(root = validationPath,
                                                         transform = transDatagen)

        validationLoader = DataLoader(validationSet,
                                      batch_size = self.m_BatchSize,
                                      shuffle = False)

        print("Validation Class [", validationLoader.dataset.class_to_idx, "]")
        print("Validation Numbers [", len(validationLoader.dataset.imgs),"]")
        print("Validation Batch Size [", validationLoader.batch_size,"]")

        return validationLoader


    def ImageTest(self):
        transDatagen = transforms.Compose([transforms.Resize((self.m_ImageDim, self.m_ImageDim)),
                                           transforms.ToTensor()])
        '''
        directory 변경 시 modeling_config.ini 파일 및 경로 수정 필요
        '''
        testDirectory = self.m_DataPath + '/testing'
        #testDirectory = self.m_DataPath + '/original'
        testSet = torchvision.datasets.ImageFolder(root = testDirectory,
                                                   transform = transDatagen)

        testLoader = DataLoader(testSet,
                                batch_size = self.m_BatchSize,
                                shuffle = False)

        print("Test Class [", testLoader.dataset.class_to_idx, "]")
        print("Test Numbers [", len(testLoader.dataset.imgs), "]")
        print("Test Batch Size [", testLoader.batch_size,"]")

        return testLoader
    




# seed fix
SEED = 3
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.manual_seed(SEED)


class PillModel(nn.Module):
    # bulid cnn model
    def __init__(self, config):
        super(PillModel, self).__init__()
        '''
        ClassNum : class number
        '''

        self.m_ClassNum = int(config['class_num'])

        channel1 = 16
        channel2 = 32
        channel3 = 64
        conv1Size = 3
        conv2Size = 3
        poolSize = 2

        self.m_Conv1 = nn.Conv2d(in_channels = 3, out_channels = channel1, kernel_size = conv1Size, padding = 1)
        self.m_Pool1 = nn.MaxPool2d(poolSize, poolSize)
        self.m_Conv2 = nn.Conv2d(in_channels = channel1, out_channels = channel2, kernel_size = conv2Size, padding = 1)
        self.m_Pool2 = nn.MaxPool2d(poolSize, poolSize)
        self.m_Conv3 = nn.Conv2d(in_channels = channel2, out_channels = channel3, kernel_size = conv2Size, padding = 1)
        self.m_Pool3 = nn.MaxPool2d(poolSize, poolSize)

        self.m_Linear4 = nn.Linear(40000, 256)
        self.m_Drop4 = nn.Dropout2d(0.5)

        self.m_Linear5 = nn.Linear(256, self.m_ClassNum)
        self.m_Relu = nn.ReLU()


    def forward(self, x):
        x = self.m_Relu(self.m_Conv1(x))
        x = self.m_Pool1(x)

        x = self.m_Relu(self.m_Conv2(x))
        x = self.m_Pool2(x)

        x = self.m_Relu(self.m_Conv3(x))
        x = self.m_Pool3(x)

        x = x.view(x.shape[0],-1)
        x = self.m_Relu(self.m_Linear4(x))
        x = self.m_Drop4(x)

        x = self.m_Linear5(x)
        return x

# 모델 hyperparam설정, 저장, 트레이닝, 테스팅
class MakeModel():
    def __init__(self,config):
        '''
        learning_rate : learning rate
        epochs : epoch
        save_path : save path
        model_name : model save name
        '''

        self._epoch = int(config['epochs'])
        self._lr = float(config['learning_rate'])
        self._savePath = config['save_path']
        self._modelName = config['model_name']


    def SaveModel(self, _model, optimizer, _trainData, trainLoss):
        nowdate = datetime.datetime.now().strftime('%y%m%d_%H')
        ret = 0
        try:
            torch.save({'model_state_dict': _model.state_dict(),
                        'epoch': self._epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': trainLoss,
                        'label_name':_trainData.dataset.classes},
                        self._savePath + '/' + self._modelName + '_PyTorchModel.pt')
            print("model saved [", self._savePath + '/' + self._modelName + "_PyTorchModel.pt ]")

        except PermissionError:
                torch.save(_model, './' + nowdate + '_' + self._modelName + '_PyTorchModel.pt')
                print('model saved [ ./' + nowdate + '_' + self._modelName + '_PyTorchModel.pt ]')

        except IOError as e:
            print("IOError except: ", e.errno)
            ret = 1

        return ret


    def Training(self, _device, _model, _trainData, _valData):
        _model.train()

        optimizer = optim.Adam(_model.parameters(), lr = self._lr)
        criterion = torch.nn.CrossEntropyLoss()
        bestValLoss = float('inf')

        for epoch in range(self._epoch):
            trainLoss = 0.0
            trainSize = 0.0
            trainCorrect = 0.0

            print("Epoch {}/{}".format(epoch + 1, self._epoch))
            progress = pkbar.Kbar(target=len(_trainData), width = 25)

            # train
            for batchIdx, data in enumerate(_trainData):
                images, labels = data
                images, labels = images.to(_device), labels.to(_device)

                optimizer.zero_grad()
                outputs = _model(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                trainLoss = loss.item()

                _, predicted = outputs.max(1)
                trainSize += labels.shape[0]
                trainCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                trainAccuracy = 100 * trainCorrect / trainSize

                progress.update(batchIdx, values = [("loss: ", trainLoss), ("acc: ", trainAccuracy)])

                del loss
                del outputs

            # validation
            with torch.no_grad():
                valLoss = 0.0
                valSize = 0.0
                valCorrect = 0.0

                for batchIdx, data in enumerate(_valData):
                    images, labels = data
                    images, labels = images.to(_device), labels.to(_device)

                    outputs = _model(images)
                    valLoss = criterion(outputs, labels).item()

                    _, predicted = outputs.max(1)
                    valSize += labels.shape[0]

                    valCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                    valAccuracy = 100 * valCorrect / valSize

                progress.add(1, values=[("val loss", valLoss), ("val acc", valAccuracy)])

            # if best loss value, save model
            if valLoss < bestValLoss:
                bestValLoss = valLoss
                ret = self.SaveModel(_model, optimizer, _trainData, trainLoss)

        return ret


    # test set model test
    def Testing(self, _device, _model, _testData):
        _model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        testLoss = 0.0
        testSize = 0.0
        testCorrect = 0.0

        progress = pkbar.Kbar(target=len(_testData), width = 25)

        with torch.no_grad():
            for batchIdx, data in enumerate(_testData):
                images, labels = data
                images, labels = images.to(_device), labels.to(_device)
                outputs = _model(images)

                testLoss = criterion(outputs, labels).item()

                _, predicted = outputs.data.max(1)
                testSize += labels.shape[0]
                testCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                accuracy = 100 * testCorrect / testSize

                progress.update(batchIdx, values = [("test loss: ", testLoss), ("test acc: ", accuracy)])

            testLoss /= len(_testData.dataset)
        progress.add(1)


# 해당 파일 실행 시 실행되는 main
class PyTorchMain():
    def __init__(self):
        config = configparser.ConfigParser()
        #config.read('./config_hexagon.ini', encoding='UTF-8')
        config.read('C:\oss_medi\WhatIsMethIs-Model\model\Modelings\modeling_config.ini', encoding='UTF-8')

        self.m_Device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.m_cPytorchModel = PillModel(config['PT_model_info'])

        self.m_cPytorchData = PyTorchData("image", config['PT_model_info'])

        self.m_cMakeModel = MakeModel(config['PT_model_info'])


    def main(self):
        print("\n[ Model ]\n")
        model = self.m_cPytorchModel.to(self.m_Device)
        print(model)

        # load dataset
        print("\n[ Data ]\n")
        trainData = self.m_cPytorchData.ImageTrain()
        valData = self.m_cPytorchData.ImageValidation()
        testData = self.m_cPytorchData.ImageTest()

        # training
        print("\n[ Training ]\n")
        ret = self.m_cMakeModel.Training(_device = self.m_Device,
                                         _model = model,
                                         _trainData = trainData,
                                         _valData = valData)
        if ret == 0 or ret == 1:
            # testing
            print("\n[ Testing ]\n")
            self.m_cMakeModel.Testing(_device = self.m_Device,
                                      _model = model,
                                      _testData = testData)

        '''
        class_to_idx json 파일로 출력.
        '''
        f = open("C:\oss_medi\WhatIsMethIs-Model\model\Modelings\model\trainData_class_to_idx_230828.json", 'w')
        f.write(json.dumps(trainData.dataset.class_to_idx))
        f.close()

# 해당 파일이 모듈로서 말고 직접 실행될 때
# ex) python Modeling.py {circle or ellipse} [config파일명 있어도 되고 없어도됨]
if __name__ == '__main__':
    print("Modeling.py 실행시작")
    mainClass = PyTorchMain()
    mainClass.main()
    print('####### Modeling.py 실행 finish #######')

