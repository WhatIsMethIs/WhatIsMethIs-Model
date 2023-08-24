#### imports ####
# python files
import PillModel as PillModel # PillModel.py로 분리해 둘거면 import할 것
import Image_circle as ImageSide_circle
import Image_ellipse as ImageSide_ellipse
import Image_cnt as ImageContourCount
# python packages
import cv2
import numpy as np
import pandas as pd
import sys
import shutil
import configparser
import datetime


#### 텍스트 유무 기준 양/단면 판별 ####
# Image_circle.py
# Image_ellipse.py
# Image_cnt.py
'''위의 세 files는 import함'''


#### 알약 이름 매칭 ####
# 모델 결과의 label과 accuracy 각각을 label과 그에 해당하는 accuracy를 매칭 시켜서 하나로 묶어주기
class PillName:
    def __init__(self, index, accuracy):
        self.index = index
        self.accuracy = accuracy
    def __repr__(self):
        return repr((self.index, self.accuracy))



#### 이미지 처리 ####
# preprocessing과 유사하게, 좌표찾아서 크롭 및 필터링 적용
class ImageProcess():
    def __init__(self):
        pass


    def ImageArea(self, input_image):
      rgba = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
      rgba = cv2.medianBlur(rgba, 55)

      imgray = cv2.cvtColor(rgba, cv2.COLOR_BGRA2GRAY)
      contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      contours_t = contours[0].transpose()

      right_point_x = np.max(contours_t[0])
      left_point_x = np.min(contours_t[0])
      right_point_y = np.max(contours_t[1])
      left_point_y = np.min(contours_t[1])

      return left_point_x, right_point_x, left_point_y, right_point_y
        

    def CropShape(self, input_image):
        left_x, right_x, left_y, right_y = self.ImageArea(input_image)
        crop_img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
        crop_img = crop_img[left_y:right_y, left_x:right_x]
  
        trans_mask = crop_img[:,:,3]==0
        crop_img[trans_mask] = [255, 255, 255, 255]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGRA2BGR)

        return crop_img
    
    
    def max_con_CLAHE(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl,a,b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return img
    
    

#### 모델 관련 ####
# PillModel.py
'''모델관련 파일은 import함'''


#### main 실행부분. config 파일을 참조해서 터미널과 같이 실행됨 ####
class PillMain():

    def main(self, argv):
        if len(argv) != 4:
            print("Argument is wrong")
            print("Usage: python PillMain.py [IMAGE1 FULL PATH] [IMAGE2 FULL PATH] [TEXT FILE PATH]")
            sys.exit()

        image1_path = argv[1]
        image2_path = argv[2]
        text_file_path = argv[3]

        data_info = pd.read_csv(text_file_path, delimiter='\t')
        ori_shape = data_info['shape'][0]
        f_text = data_info['f_text'][0]
        b_text = data_info['b_text'][0]
        drug_list_ori = data_info['drug_code'][0].replace('[','').replace(']','').replace(' ','').split(',')

        if drug_list_ori[0] == 'none':
            drug_list = drug_list_ori[0]
        else:
            drug_list = drug_list_ori

        nowdate = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        log_path = '/data/pred_log/'+nowdate+'.log'
        f=open(log_path,'a')

        shape_list = ['circle', 'ellipse', 'triangle', 'diamond', 'pentagon', 'hexagon', 'octagon', 'square', 'etc']
        if ori_shape not in shape_list:
            print("SHAPE : circle, ellipse, triangle, diamond, pentagon, hexagon, octagon, square, and etc")
            sys.exit()

        # if circle and ellipse shape, judgment both side or one side
        if ori_shape == 'circle':
            imageside = ImageSide_circle.ImageContour()
            proportion = 4.7
            _, image1_result, contourcnt1 = imageside.Process(image1_path, proportion)
            _, image2_result, contourcnt2 = imageside.Process(image2_path, proportion)

        elif ori_shape == 'ellipse':
            imageside = ImageSide_ellipse.ImageContour()
            proportion = 5.5
            _, image1_result, contourcnt1 = imageside.Process(image1_path, proportion)
            _, image2_result, contourcnt2 = imageside.Process(image2_path, proportion)

        else:
            imagecontourcount = ImageContourCount.ImageContour()
            proportion = 4.7
            contourcnt1 = imagecontourcount.Process(image1_path, proportion)
            contourcnt2 = imagecontourcount.Process(image2_path, proportion)

        # choice one input image
        choiceimage = PillModel.ChoiceImage()
        if ori_shape in ('circle', 'ellipse'):
            # if input text count is two, shape is 'BOTH'
            if f_text != 'none' and b_text != 'none':
                shape, image_path = choiceimage.ChoiceImage(ori_shape, image1_result, image2_result, contourcnt1, contourcnt2, image1_path, image2_path, True)
            else:
                shape, image_path = choiceimage.ChoiceImage(ori_shape, image1_result, image2_result, contourcnt1, contourcnt2, image1_path, image2_path)
        else:
            image_path = choiceimage.ChoiceImageContour(contourcnt1, contourcnt2, image1_path, image2_path)
            shape = ori_shape

        f.write(shape+'\n')
        f.write(image_path+'\n')

        # config file load for each shape
        config = configparser.ConfigParser()
        shape_path = '/data/config_shape/'

        if shape == 'circle_BOTH':
            config_file = shape_path + 'config_circle_BOTH.ini'
        elif shape == 'circle_ONESIDE':
            config_file = shape_path + 'config_circle_ONESIDE.ini'
        elif shape == 'ellipse_BOTH':
            config_file = shape_path + 'config_ellipse_BOTH.ini'
        elif shape == 'ellipse_ONESIDE':
            config_file = shape_path + 'config_ellipse_ONESIDE.ini'
        elif shape == 'triangle':
            config_file = shape_path + 'config_triangle.ini'
        elif shape == 'diamond':
            config_file = shape_path + 'config_diamond.ini'
        elif shape == 'pentagon':
            config_file = shape_path + 'config_pentagon.ini'
        elif shape == 'hexagon':
            config_file = shape_path + 'config_hexagon.ini'
        elif shape == 'octagon':
            config_file = shape_path + 'config_octagon.ini'
        elif shape == 'square':
            config_file = shape_path + 'config_square.ini'
        elif shape == 'etc':
            config_file = shape_path + 'config_etc.ini'
        
        config.read(config_file, encoding='UTF-8')
        pillModel = PillModel.PillModel(config['pill_model_info']) 
        
        # image processing
        pillModel.pill_image_process(image_path)
        
        # image open
        img = pillModel.testImage(config['pill_model_info']['make_folder_path'])
        
        # model loading
        pillModel.pill_shape_conf(shape)
        pillModel.pill_model_loading(config['pill_model_info'])

        # prediction
        output = pillModel.pill_prediction(img)
        indices_top, includ_count = pillModel.pill_sorting(output, drug_list)

        # if shape_oneside( or shape_both) model training drug code is not in drug list, try shape_both (or shape_oneside) model
        if (includ_count == 0) and (ori_shape in ('circle','ellipse')):
            if shape == ori_shape+'_ONESIDE':
                shape = ori_shape+'_BOTH'
                config_file = shape_path + 'config_' + shape + '.ini'

            else:
                shape = ori_shape+'_ONESIDE'
                config_file = shape_path + 'config_' + shape+'.ini'

            f.write(shape+'\n')

            config.read(config_file, encoding='UTF-8')
            pillModel = PillModel.PillModel(config['pill_model_info']) 
            pillModel.pill_shape_conf(shape)
            pillModel.pill_model_loading(config['pill_model_info'])

            output = pillModel.pill_prediction(img)
            indices_top, includ_count = pillModel.pill_sorting(output, drug_list)

        print(pillModel.pill_information(indices_top))
        f.write(pillModel.pill_information(indices_top))
        f.close()

        # remove filter image folder
        shutil.rmtree(config['pill_model_info']['make_folder_path'])


if __name__ == '__main__':
    main_class = PillMain()
    main_class.main(sys.argv)
    
