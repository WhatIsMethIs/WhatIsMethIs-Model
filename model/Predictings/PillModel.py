#### imports ####
# python file
import PyTorchModel as PyTorchModel
from Predicting import PillName as PillName
from Predicting import ImageProcess as ImageProcess
# python packages
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from operator import attrgetter
import cv2
import json
import os
# pytorch 설치안내-conda 기준
'''
# CUDA 9.0
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=9.0 -c pytorch

# CUDA 10.0
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch

# CPU Only
conda install pytorch-cpu==1.0.1 torchvision-cpu==0.2.2 cpuonly -c pytorch
 '''


# 알약 모델 기능 관련 클래스
class PillModel():
    def __init__(self, config):
        self.pill_code = []
        self.imageProcess = ImageProcess.ImageProcess()
        self.workDirectory = "./Modelings/model/"
        self.top_count = int(config['top_count'])
        self.pill_top = int(config['pill_top'])
        self.ImageDim = int(config['image_dim'])
        self._lr = float(config['learning_rate'])
        self.make_folder_path = config['make_folder_path']
        

    # shape별 모델 파일명 지정
    def pill_shape_conf(self, shape):
        self.model_file = self.workDirectory + shape + "_PyTorchModel.pt"
        
        
    # 학습해서 저장한 model loading
    def pill_model_loading(self, config):
        self.model = PyTorchModel.PillModel(config)
        optimizer = optim.Adam(self.model.parameters(), lr = self._lr)
        
        checkpoint = torch.load(self.model_file, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dataset = checkpoint['label_name']
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()


    # 예측 정확도 순 정렬해서 top5 나열. sorting and top5 
    def pill_sorting(self, output, drug_code_list):
        # accuracy sorting
        indices_objects=[]
        for i in range(len(self.dataset)):
            indices_objects.append(PillName.PillName(self.dataset[i], output[0][i]))
            
        indices_objects = sorted(indices_objects, key=attrgetter('accuracy'), reverse=True)
        self.pill_top = len(self.dataset) if len(self.dataset) < self.top_count else self.top_count

        # resorting with drug list
        if drug_code_list != 'none':
            drug_list = list(set(drug_code_list))
        else:
            drug_list = 'none'

        includ_count = 1

        if drug_list != 'none':
            re_sorting = []
            for drugcode in range(len(indices_objects)):
                if indices_objects[drugcode].index in drug_list:
                    re_sorting.append(indices_objects[drugcode])
            
            # training drug code가 drug code 리스트에 없다면, includ_count는 0이다
            if len(re_sorting) == 0:
                includ_count = 0

            if len(re_sorting) != 5:
                re_len = 5 - len(re_sorting)
                cnt = 0 
                for drugcode in range(len(indices_objects)):
                    if indices_objects[drugcode].index not in drug_list:
                        re_sorting.append(indices_objects[drugcode])
                        cnt += 1
                        if cnt == re_len:
                            break
        else:
            re_sorting = indices_objects

        # top5
        indices_top = []
        i, count = 0, 0
        while (count < self.pill_top):
            indices_top.append(re_sorting[i])
            count += 1
            i += 1
        
        return indices_top, includ_count

    
    # 알약 예측. pill prediction
    def pill_prediction(self, img):
        self.model.eval()
        
        with torch.no_grad():
            for i, (image, label) in enumerate(img):
                image,label = image.to(self.device), label.to(self.device)
                output = self.model(image)
                output_min, _= output.data.min(1)
                plus_output = output - output_min
                per_output = plus_output/plus_output.sum()*100
                
                loss = self.criterion(output, label)
                
            return per_output


    # 클래스명 및 정확도 정보 .json파일 반환. class name and accuracy information
    def pill_information(self, indices_top):
        pill_list=[]
        for i in range(self.pill_top):
            data = {}
            data['rank'] = i + 1
            data['code'] = indices_top[i].index
            data['accuracy'] = float(indices_top[i].accuracy)
    
            pill_list.append(data)
    
        jsonString = json.dumps(pill_list)
        
        return jsonString


    # 하나의 알약이미지 처리: 크롭, 필터링한 뒤 경로만들어 새로 저장
    def pill_image_process(self, img_path):
        image_process = self.imageProcess.CropShape(img_path)
        image_process = self.imageProcess.max_con_CLAHE(image_process)
        image_process = self.imageProcess.max_con_CLAHE(image_process)

        # if img_path is absolute path, use image file, so extraction filename in absolute path
        filename = os.path.basename(img_path)

        # When loading an image in pytorch, it is loaded by folder, so there must be a folder.
        if not os.path.exists(self.make_folder_path):
            os.makedirs(self.make_folder_path)
            os.makedirs(self.make_folder_path+'result')

        cv2.imwrite(self.make_folder_path+'result/'+filename+'_temp.jpg', image_process)
 
            
    # test image set
    def testImage(self, testimgdir):
        transDatagen = transforms.Compose([transforms.Resize((self.ImageDim, self.ImageDim)),
                                           transforms.ToTensor()])
        testimgset = torchvision.datasets.ImageFolder(root = testimgdir,
                                                      transform = transDatagen)
        testimg = DataLoader(testimgset, batch_size=1, shuffle=False)
        
        return testimg


# 모델의 예측 후 클래스 (choice one image)
class ChoiceImage():
    def __init__(self):
        pass
    
    def ChoiceImage(self,shape, image1_result, image2_result, contourcnt1, contourcnt2, image1_path, image2_path, text_option=False):
        # 두 이미지 모두 같은 면인 경우, countourcnt로 판단
        if (image1_result == 'BOTH' and image2_result == 'BOTH') or (image1_result == 'ONESIDE' and image2_result == 'ONESIDE'):
            if contourcnt1 > contourcnt2:
                result = image1_result
                image_path = image1_path
            else:
                result = image2_result
                image_path = image2_path

        # image2 결과가 단면인 경우, image2 결과, 경로는 image1 경로
        elif image1_result == 'BOTH' and image2_result == 'ONESIDE':
            result = 'ONESIDE'
            image_path = image1_path

        # image1 결과가 단면인 경우, image1 결과, 경로는 image2 경로
        elif image1_result == 'ONESIDE' and image2_result == 'BOTH':
            result = 'ONESIDE'
            image_path = image2_path

        # 텍스트가 있다면, both로 아닌 경우 result로 표기
        if text_option == True:
            shape = shape + '_' + 'BOTH'
        else:
            shape = shape + '_' + result

        return shape, image_path

    # 이미지1, 이미지2 중, countourcnt가 큰 이미지 경로를 반환
    def ChoiceImageContour(self, contourcnt1, contourcnt2, image1_path, image2_path):
        if contourcnt1 > contourcnt2:
            image_path = image1_path
        else:
            image_path = image2_path

        return image_path
