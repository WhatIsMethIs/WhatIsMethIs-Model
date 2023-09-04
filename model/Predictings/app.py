#### imports ####
# python files
from PillModel import PillModel
# python packages
import configparser
import sys
import shutil
import datetime


class PillMain():
    # 이 파일의 메인함수
    def main(self, argv):

        # 커맨드 입력 잘못 됐으면 에러
        if len(argv) != 2:
            print("Argument is wrong")
            print("Usage: python app.py [IMAGE FULL PATH]") # 나는 python Predictings/app.py Datasets/photo_40720_rembg.png 이렇게 입력.
            sys.exit()

        # 커맨드 입력 변수 얻기
        image_path = argv[1]
        shape='circle'
        drug_list = 'none'

        # 로그 찍기 (파일 객체 f)
        nowdate = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        log_path = './Predictings/predicting_logs/'+nowdate+'.log'
        f=open(log_path,'a')        
        f.write('User input_image: '+image_path+'\n') # 이미지 경로 로그

        # shape이 존재하지 않으면 에러
        shape_list = ['circle']
        if shape not in shape_list:
            f.write('Not defined shape \n system exit') # 에러 로그
            print("available SHAPE : circle")
            sys.exit()

        # config file load
        config = configparser.ConfigParser()
        config_file_path = 'Predictings\predicting_conifg.ini' # ****경로에 맞게 변경 필요
        config.read(config_file_path, encoding='UTF-8')
        f.write('Reading config file is completed\n') # config파일 로드 로그

        # 모델 기능수행 인스턴스생성
        pillModel = PillModel(config['pill_model_info']) 
        
        # 예측할 이미지 전처리해서 텐서로 open
        pillModel.pill_image_process(image_path)
        img = pillModel.testImage(config['pill_model_info']['make_folder_path'])
        f.write('Preprocessing of input_image is completed\n') # 사용자 입력 이미지 전처리 완료 로그


        # 모델 로딩
        fileName="230903_model_02_PyTorchModel.pt" # 모델 변경 시 변경 필요
        pillModel.pill_shape_conf(fileName)
        pillModel.pill_model_loading(config['pill_model_info'])
        f.write('Model is ready\n') # 모델 로딩 완료 로그

        # 모델을 통한 예측 및 결과 얻기
        output = pillModel.pill_prediction(img)
        indices_top = pillModel.pill_sorting(output, drug_list) 

        # 최종 결과 확인차 출력 및 로그파일 닫기
        result=pillModel.pill_information(indices_top)
        print("예측결과 출력...\n",result) # 확인용
        f.write("Prediction Results are "+result) # 예측 결과 로그
        f.close() # 로그파일 닫기

        # 전처리된 입력이미지 임시저장했던 폴더&하위파일 삭제 (이거 안되면 다음 실행시 이전 입력이미지 사용해서 잘못된 결과)
        shutil.rmtree(config['pill_model_info']['make_folder_path'])

# 파일 실행 시
if __name__ == "__main__":
    print("####### app.py 실행시작 #######")
    main_class = PillMain()
    main_class.main(sys.argv)
    print('####### app.py 실행 finish #######')
