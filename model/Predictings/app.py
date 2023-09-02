#### imports ####
# python files
from PillModel import PillModel
from ImageProcess import ImageContour
# python packages
# from flask import Flask, request
from torchvision import transforms
import configparser
import sys
import shutil
import datetime
from PIL import Image
import cv2

# pytorch 설치안내-conda 기준
'''
# CUDA 9.0
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=9.0 -c pytorch

# CUDA 10.0
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch

# CPU Only
conda install pytorch-cpu==1.0.1 torchvision-cpu==0.2.2 cpuonly -c pytorch
'''


class PillMain():
    # 이 파일의 메인함수
    def main(self, argv):

        # 커맨드 입력 잘못 됐으면 에러
        if len(argv) != 2:
            print("Argument is wrong")
            print("Usage: python app.py [IMAGE FULL PATH]")
            sys.exit()

        # 커맨드 입력 변수 얻기
        image_path = argv[1]
        ori_shape = 'circle'
        shape=ori_shape
        drug_list = 'none'

        ''' 
        data_info = pd.read_csv(text_file_path, delimiter='\t')
        ori_shape = data_info['shape'][0]
        f_text = data_info['f_text'][0]
        b_text = data_info['b_text'][0]
        drug_list_ori = data_info['drug_code'][0].replace('[','').replace(']','').replace(' ','').split(',')

        if drug_list_ori[0] == 'none':
            drug_list = drug_list_ori[0]
        else:
            drug_list = drug_list_ori
        '''

        # 로그 찍기 (파일 객체 f)
        nowdate = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        log_path = './Predictings/predicting_logs/'+nowdate+'.log'
        f=open(log_path,'a')        
        f.write('input_image: '+image_path+'\n') # 이미지 경로 쓰기
        f.write(shape+'\n') # 모양도 쓰고

        # shape이 존재하지 않으면 에러
        shape_list = ['circle']
        if ori_shape not in shape_list:
            f.write('shape is not defined. \n system exit')
            print("SHAPE : circle")
            sys.exit()

        # config file load for circle shape
        config = configparser.ConfigParser()
        config_file_path = 'Predictings\predicting_conifgs\config_circle_ONESIDE.ini' # 경로에 맞게 변경 필요
        config.read(config_file_path, encoding='UTF-8')
        f.write('config file reading completed\n')

        # 클래스 호출
        pillModel = PillModel(config['pill_model_info']) 
        
        # 예측할 이미지 전처리해서 open
        pillModel.pill_image_process(image_path)
        img = pillModel.testImage(config['pill_model_info']['make_folder_path'])
        f.write('preprocessing of input_image completed\n')

        '''
        # image contour관련
        proportion = 4.7
        imagecontourcount = ImageContour()
        contourcnt = imagecontourcount.Process(image_path, proportion) # 이때 image_path는 스트링
        '''

        # 모델 로딩
        pillModel.pill_shape_conf(shape)
        pillModel.pill_model_loading(config['pill_model_info'])
        f.write('model is ready\n')

        # 모델을 통한 예측 및 결과 얻기
        output = pillModel.pill_prediction(img)
        indices_top, includ_count = pillModel.pill_sorting(output, drug_list)

        # 결과 확인차 출력 및 로그파일 닫기
        print("예측결과 출력...\n",pillModel.pill_information(indices_top))
        f.write("Prediction Results are...\n"+pillModel.pill_information(indices_top))
        f.close()

        # 전처리된 입력이미지 임시저장했던 폴더&파일 삭제
        shutil.rmtree(config['pill_model_info']['make_folder_path'])

# 파일 실행 시
if __name__ == "__main__":
    # 터미널에 모델 폴더로 경로 이동 후, 이런 식으로 입력: python Predictings/app.py C:\medi_test\Datasets\pill_41107.png
    print("####### app.py 실행 시작 #######")
    main_class = PillMain()
    main_class.main(sys.argv)
    print('####### app.py 실행 finish #######')
