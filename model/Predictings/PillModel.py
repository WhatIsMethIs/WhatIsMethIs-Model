#### 알약 모델 기능 관련 클래스 ####
# python file
import PyTorchModel
from PillName import PillName
from ImageProcess import ImageProcess
# python packages
import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from operator import attrgetter
import os
import cv2
import json


class PillModel():
    def __init__(self, config):
        self.pill_code = []
        self.imageProcess = ImageProcess()
        self.workDirectory = "./Modelings/model/" # 모델 폴더의 경로
        self.top_count = int(config['top_count'])
        self.pill_top = int(config['pill_top']) # 데이터 label이 top_count 수 보다 작을 경우를 위해 0부터 시작하는 변수를 만들어 오류 발생 방지
        self.ImageDim = int(config['image_dim'])
        self._lr = float(config['learning_rate'])
        self.make_folder_path = config['make_folder_path'] # 전처리한 사용자입력 이미지 임시저장 폴더 경로
    
    ##### 전처리, 모델에 넣어 예측 #######

    # 모델 파일 경로 =모델 폴더 경로+모델 파일명
    def pill_shape_conf(self, fileName):
        self.model_file = self.workDirectory + fileName
        
    # 저장된 모델 로드
    def pill_model_loading(self, config):
        self.model = PyTorchModel.PillModel(config)

        optimizer = optim.Adam(self.model.parameters(), lr = self._lr) # 옵티마이저는 Adam
        checkpoint = torch.load(self.model_file, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dataset = checkpoint['label_name']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss() # 손실함수는 CrossEntropy

     # 사용자 입력 알약이미지 전처리
    def pill_image_process(self, img_path):
     
        output_path=self.imageProcess.EmptyBG(img_path) # 배경제거하기
        processed_img = self.imageProcess.CropShape(output_path) # 크롭하기

        os.remove(output_path) # "emptyBG.png"파일은 삭제        
        filename = os.path.basename(output_path) # 이미지파일경로(absoloute path)로부터 이미지파일명 추출

        # 임시 폴더 생성: When loading an image in pytorch, it is loaded by folder, so there must be a folder.
        if not os.path.exists(self.make_folder_path):
            os.makedirs(self.make_folder_path)
            os.makedirs(self.make_folder_path+'result')

        # 임시 이미지 저장 (앱실행 전체 종료시 삭제됨)
        cv2.imwrite(self.make_folder_path+'result/'+filename+'_temp.jpg', processed_img)
 
            
    # 전처리된 이미지 resizing&텐서로 변환
    def testImage(self, testimgdir):

        transDatagen = transforms.Compose([transforms.Resize((self.ImageDim, self.ImageDim)),transforms.ToTensor()])
        testimgset = torchvision.datasets.ImageFolder(root = testimgdir, transform = transDatagen)
        testimg = DataLoader(testimgset, batch_size=1, shuffle=False)
        
        return testimg


    # 전처리된 이미지 텐서로 알약 레이블 예측
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



    ##### 여기부터는 예측 결과 정렬 및 반환 #######
    
    # sorting and topN개 반환
    def pill_sorting(self, output, drug_code_list):

        indices_objects=[]

        # 이미지 예측 결과텐서 softmax해서 각 품목에 대한 예측확률(%) 표시
        acc=100*torch.softmax(output[0], dim=0)

        # 기존 label 순서(이름 오름차순)로 예측확률(%)이랑 튜플매칭
        for i in range(len(self.dataset)):
            indices_objects.append(PillName(self.dataset[i], acc[i]))
        
        # 튜플의 리스트를 예측확률(%) 큰 순으로 내림차순 재정렬
        indices_objects = sorted(indices_objects, key=attrgetter('accuracy'), reverse=True)

        # pill_top= min(품목수, 나열할개수)
        self.pill_top = len(self.dataset) if len(self.dataset) < self.top_count else self.top_count

        # 결과를 내놓을 최종 topN
        indices_top = []
        i, count = 0, 0
        while (count < self.pill_top):
            indices_top.append(indices_objects[i])
            count += 1
            i += 1
        
        return indices_top
    
    # 상위 N개 class name and accuracy information를 JSON스트링으로 반환
    def pill_information(self, indices_top):

        pill_list=[]

        for i in range(self.pill_top):
            data = {} # data={'rank':1, 'code': , 'accuracy': }를 1순위부터 주르륵 가지는 리스트
            data['rank'] = i + 1
            data['code'] = indices_top[i].index
            data['accuracy'] = float(indices_top[i].accuracy)
            # print(f"data{i}: {data}")
            
            pill_list.append(data)
    
        jsonString = json.dumps(pill_list)

        return jsonString
    
   