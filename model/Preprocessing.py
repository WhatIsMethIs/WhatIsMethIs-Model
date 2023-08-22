# Preprocessing.py 파이썬파일 with 주석
# config 파일에서 경로설정하면 됨
import os
import sys
import configparser
import errno
import shutil
import numpy as np
import cv2 # 설치필요 pip install opencv-python
from PIL import Image # 설치필요 pip install pillow

# Image Crop & Rotation
class ImageCrop():
    def __init__(self, config):
        '''
        open_path : original image open path
        save_path : crop image save path
        rotate_path : rotate image save path
        rotation_angle_circle : circle rotation angle
        rotation_angle_ellipse : ellipse rotation angle
        '''

        self.open_path = config['open_path']
        self.save_path = config['save_path']
        self.rotate_path = config['rotate_path']
        self.rotate_angle_circle = int(config['rotation_angle_circle']) 
        self.rotate_angle_ellipse = int(config['rotation_angle_ellipse']) 

        # Permission Error retry another path
        self.error_path_crop = './crop'
        self.error_path_rotation = './rotation'

        # dir index setting
        self.start_dir = config['start_dir_idx']
        self.end_dir = config['end_dir_idx']


    # square four point return
    def ImageArea(self, input_image):
        rgba = cv2.medianBlur(input_image, 55)

        imgray = cv2.cvtColor(rgba, cv2.COLOR_BGRA2GRAY)
        contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        contours_t = contours[0].transpose()

        right_point_x = np.max(contours_t[0])
        left_point_x = np.min(contours_t[0])
        right_point_y = np.max(contours_t[1])
        left_point_y = np.min(contours_t[1])

        return left_point_x, right_point_x, left_point_y, right_point_y


    # image crop
    def CropShape(self, input_image):
        left_x, right_x, left_y, right_y = self.ImageArea(input_image)
        crop_img = input_image[left_y:right_y, left_x:right_x]

        return crop_img


    # circle image rotation - 원 회전
    def rotate_image_circle(self, save_rotate_img, input_image):
        i = 0
        height, width, channel = input_image.shape

        while i < 360:
            f_path = save_rotate_img + '_' + str(i) + '.png'
            if not os.path.isfile(f_path):
                matrix = cv2.getRotationMatrix2D((width/2, height/2), i, 1)
                dst = cv2.warpAffine(input_image, matrix, (width, height))
                dst = self.CropShape(dst)

                cv2.imwrite(f_path, dst)
            else:
                print('rotate file exits : ', f_path)

            i = i + self.rotate_angle_circle


    # ellipse image rotation - 타원 회전
    def rotate_image_ellipse(self, save_rotate_img, input_image):
        i = 0
        height, width, channel = input_image.shape

        while i < 360:
            if (i < 30) or (150 < i and i < 210) or (330 < i):
                f_path = save_rotate_img + '_' + str(i) + '.png'
                if not os.path.isfile(f_path):
                    matrix = cv2.getRotationMatrix2D((width/2, height/2), i, 1)
                    dst = cv2.warpAffine(input_image, matrix, (width, height))
                    dst = self.CropShape(dst)

                    cv2.imwrite(f_path, dst)
                else:
                    print('rotate file exits : ', f_path)

            i = i + self.rotate_angle_ellipse


    # image crop and rotation process - 전체 과정
    def ImageProcess(self, shape):
        or_dirnames = os.listdir(self.open_path) # 원본디렉토리 내 품목디렉토리들

        if( int(self.start_dir) == -1 ):
            dirnames = or_dirnames
        else:
            dirnames = or_dirnames[int(self.start_dir):int(self.end_dir)]

        for dir in dirnames: # 각 품목디렉토리에 대하여
            try_num = 0
            # try
            try:
                # not exists folder in path, make folder
                if not os.path.exists(self.save_path + dir):
                    os.makedirs(self.save_path + '/' + dir)

                if not os.path.exists(self.rotate_path + dir):
                    os.makedirs(self.rotate_path + '/' + dir)
                try_num = 1

            except PermissionError:
                if not os.path.exists(self.error_path_crop + dir):
                    os.makedirs(self.error_path_crop + '/' + dir)
                    print("PermissionError except, image save path: ", self.error_path_crop)

                if not os.path.exists(self.error_path_rotation + dir):
                    os.makedirs(self.error_path_rotation + '/' + dir)
                    print("PermissionError except, image save path: ", self.error_path_rotation)
                try_num = 2

            except IOError as e:
                print("IOError except: ", e.errno)

            open_folder_path = os.path.join(self.open_path, dir) # 품목 디렉토리 경로 = 원본디렉토리 경로 + 품목 디렉토리명
            if try_num == 1:
                save_folder_path = os.path.join(self.save_path, dir)
                rotate_folder_path = os.path.join(self.rotate_path, dir)
            elif try_num == 2:
                save_folder_path = os.path.join(self.error_path_crop, dir)
                rotate_folder_path = os.path.join(self.error_path_rotation, dir)

            for path, folder, files in os.walk(open_folder_path):

                for file in files:
                    input_image = open_folder_path + '/' + file

                    print(input_image)

                    save_image = save_folder_path + '/' + file[0:len(file)-3] + 'png'
                    input_image = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)

                    '''image crop'''
                    if not os.path.isfile(save_image):
                        crop_img = self.CropShape(input_image)
                        cv2.imwrite(save_image, crop_img)
                    else:
                        print( 'crop image file exits : ', save_image)

                    '''rotation''' # shape에 따라 다른 회전 필요
                    save_rotate_img = rotate_folder_path + '/' + file[0:len(file)-4]

                    if shape == 'circle':
                        self.rotate_image_circle(save_rotate_img, input_image)
                    elif shape == 'ellipse':
                        self.rotate_image_ellipse(save_rotate_img, input_image)

# Image Filtering
class ImageFiltering():
    def __init__(self, config):
        '''
        rotate_path : rotate image save path
        filter_path : filter image save path
        '''

        self.origin_folder_path = config['rotate_path']
        self.filter_folder_path = config['filter_path']

        # Permission Error retry another path
        self.error_path_filter = './filter'


    # make white background - image 배경을 white로 바꿔서 저장
    def white_Background(self, img):
        trans_mask = img[:,:,3]==0
        img[trans_mask] = [255, 255, 255, 255]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img


    # filter (컬러모델의 l채널 변경하여 적용)
    def max_con_CLAHE(self, img):
        # Converting image to LAB Color model
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        # Merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))

        # Converting image from LAB Color model to RGB model
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return img

    # if image open fail, write log of fail image full path
    def tracelog(self, text):
        if text != "":
            text += "\n"
            f = open("tracelog.log","a")
            f.write(text)
            f.close()


    # filtering processing (실제 실행 과정)
    def imgFiltering(self):
        dirnames = os.listdir(self.origin_folder_path)

        for dir in dirnames: # 디렉토리당 각각의 품목디렉토리에 대하여
            try_num = 0
            try:
                if not os.path.exists(self.filter_folder_path + dir):
                    os.makedirs(self.filter_folder_path + dir)
                try_num = 1

            except PermissionError:
                if not os.path.exists(self.error_path_filter + dir):
                    os.makedirs(self.error_path_filter + dir)
                    print("PermissionError except, image save path: ", self.error_path_filter)
                try_num = 2

            except IOError as e:
                print("IOError except: ", e.errno)

            origin_file_path = os.path.join(self.origin_folder_path, dir) # rotated의 각 품목 디렉토리 경로
            if try_num == 1:
                filter_file_path = os.path.join(self.filter_folder_path, dir) # filtered의 각 품목 디렉토리 경로
            elif try_num == 2:
                filter_file_path = os.path.join(self.error_path_filter, dir)

            for path, folder, files in os.walk(origin_file_path):

                for file in files: # 디렉토리 내 파일들 중 각 파일에 대하여
                    print(filter_file_path+'/' + file)
                    print("origin path : ", origin_file_path +'/'+ file)

                    # file check
                    save_path = filter_file_path + '/' + file[0:len(file)-4] + '.jpg' # 저장경로는 filtered의 각 품목 디렉토리 경로+파일명
                    if os.path.isfile(save_path):
                        print("filter file exists : {}".format(save_path))
                        continue

                    img = cv2.imread(origin_file_path +'/'+ file, cv2.IMREAD_UNCHANGED)

                    # image check
                    if img is None:
                        self.tracelog(origin_file_path +'/'+ file)
                        print('typecheck ', type(img))
                        continue

                    ''' white background '''
                    img = self.white_Background(img)

                    ''' default filter '''
                    img = self.max_con_CLAHE(img)
                    img = self.max_con_CLAHE(img)

                    ''' filtering image save '''
                    cv2.imwrite(save_path , img)

# 데이터셋 분리
class Separate():
    def __init__(self, config): 
        '''
        filter_folder_path : filter image save path
        folder_save_path : traing/validation/traing separate image folder save path
        '''
        self.open_path = config['rotate_path']
        self.save_path = config['separate_path']


    def FolderList(self): # open_path 내 하위폴더명 반환
        type_list = [] # 필터사진 폴더 내 품목폴더 경로 리스트
        folder_list = [] # 필터사진 폴더 내 품목폴더 이름 리스트

        for (path, folder, files) in os.walk(self.open_path):
            type_list.append(path)
            folder_list.append(folder)

        folders = ','.join(folder_list[0]) 
        folder_list = folders.split(',')
        type_list.pop(0)

        return folder_list


    def makeSubfolder(self, new_dir_path, folder_list): # new_dir_path 하위에 foler_list의 각 원소명으로 폴더들 만들기
        for num in range(len(folder_list)):
            os.makedirs(new_dir_path + folder_list[num])


    def ml_directory(self, folder_list): # folder_list 내 쓰임에 따라 seperated+쓰임 디렉토리 만들기
        if not os.path.exists(self.save_path + 'training'):
            os.makedirs(self.save_path + 'training') # separated 폴더경로에 training 폴더 만들기
            self.makeSubfolder(self.save_path + 'training/', folder_list) # training 폴더 하위에 folder_list 각 원소명으로 폴더들 만들기

        if not os.path.exists(self.save_path + 'testing'):
            os.makedirs(self.save_path + 'testing')
            self.makeSubfolder(self.save_path + 'testing/', folder_list)

        if not os.path.exists(self.save_path + 'validation'):
            os.makedirs(self.save_path + 'validation')
            self.makeSubfolder(self.save_path + 'validation/', folder_list)


    def separate(self, dir_path, x):
        dirname = self.open_path + x # 특정품목필터디렉토리명= 필터디렉토리경로+ 특정품목번호
        filenames = os.listdir(dirname) # 특정품목번호의 모든 필터링된 이미지명들
        i = 0
        for filename in filenames: # 특정품목번호의 필터링된 이미지 각각에 대하여
            full_filename = os.path.join(dirname, filename) # 풀이미지명=특정품목필터디렉토리명+이미지명

            with Image.open(full_filename) as image:
                if i % 10 < 7: # 70%는 트레이닝용
                    training_directory = os.path.join(dir_path + 'training/', x) # seprated 폴더 내 training 폴더 경로 + 품목번호 를 경로라고 하자.
                    shutil.copyfile(full_filename, os.path.join(training_directory, filename)) # 풀이미지명 파일을 위 경로+파일명으로 카피하자.

                elif i % 10 >= 7 and i % 10 < 8: # 10%는 테스트용
                    validation_directory = os.path.join(dir_path + 'testing/', x)
                    print(f"{full_filename}파일을 테스트용으로 {os.path.join(validation_directory, filename)}에 카피할 것임")
                    shutil.copyfile(full_filename, os.path.join(validation_directory, filename))

                else: # 20%는 검증용
                    testing_directory = os.path.join(dir_path + 'validation/', x)
                    shutil.copyfile(full_filename, os.path.join(testing_directory, filename))
            i = i + 1


    def separateProcess(self):
        folder_list = self.FolderList() # 품목명 리스트
        self.ml_directory(folder_list) # 저장 경로 내부에 training/validation/testing 폴더 만들고 각 폴더 하위에 품목번호리스트로 폴더 만들기
        # 결과 예) seperated -> training -> 40122 -> 이미지 구조

        for x in folder_list: # 각 품목번호에 대하여
            self.separate(self.save_path, x) 

# 메인
class PreprocessingMain():
    def __init__(self): # config파일명 default
        self.config_file = 'preprocessing_config.ini'

    def main(self, argv):
        # ex) python3 PreProcessing.py circle config.ini
        if len(argv) == 3: # config파일명 따로 있으면 이거 따름
            self.config_file = argv[2]

        
        shape = argv[1]

        config = configparser.ConfigParser()
        config.read(self.config_file, encoding='UTF-8')

        self.ImgCrop = ImageCrop(config['img_processing'])
        self.ImgFilter = ImageFiltering(config['img_processing'])
        self.Separate = Separate(config['img_processing'])


        """ Image Crop """
        self.ImgCrop.ImageProcess(shape)

        """ Image Filtering """
        self.ImgFilter.imgFiltering()

        """ Image Separte"""
        self.Separate.separateProcess()

        print('#### main 실행 finish ######')

# 해당 파일이 모듈로서 말고 직접 실행될 때
# ex) python PreprocessingMain.py {circle or ellipse} [config파일명 있어도 되고 없어도됨]
if __name__ == '__main__':
    print("Preprocessing.py 실행시작")
    main_class = PreprocessingMain()
    main_class.main(sys.argv)
    print('####### Preproecessing.py 실행 finish #######')
