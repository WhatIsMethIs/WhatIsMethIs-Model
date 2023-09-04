#### 이미지 전처리 : 크롭 ####
import cv2
import numpy as np
from rembg import remove # pip install rembg
from PIL import Image

class ImageProcess():
    def __init__(self):
        pass

    # 배경 제거
    def EmptyBG(self, input_image):
        img=Image.open(input_image)
        out=remove(img)
        output_path="emptyBG_temp.png"
        out.save(output_path)

        return output_path
    
    # 네 꼭짓점 좌표 찾기
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
        
    # 꼭짓점 좌표에 맞춰 크롭하기
    def CropShape(self, input_image):
        left_x, right_x, left_y, right_y = self.ImageArea(input_image) 
        crop_img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
        crop_img = crop_img[left_y:right_y, left_x:right_x]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGRA2BGR)
        
        '''
        # 배경색을 바꿔주는 작업
        # trans_mask = crop_img[:,:,3]==0 # 에러유의! png 쉐입은 (,,4)이지만 jpg/jpeg는 쉐입이 (,,3)이기 때문에 [:,:,3]이 인덱스초과
        # crop_img[trans_mask] = [0, 0, 0, 0] # 이건 검은색 배경
        # crop_img[trans_mask] = [255, 255, 255, 255] # 이건 흰색 배경
        '''
        return crop_img
    
    # 색필터링 --> 제거