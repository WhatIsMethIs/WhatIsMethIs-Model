#### 이미지 처리 ####
# preprocessing과 유사하게, 좌표찾아서 크롭 및 필터링 적용
import cv2
import numpy as np
from PIL import Image

# 이미지 전처리 (색필터링 및 크롭)
class ImageProcess():
    def __init__(self):
        pass
    
    # 색 필터링 (이건 엣지 뽑을 때 사용)
    def max_con_CLAHE(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl,a,b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return img
    
    # 네 꼭짓점 좌표 찾기 (크롭에 사용)
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
        
    # 꼭짓점 좌표에 맞춰 크롭하기 *중요한 전처리임
    def CropShape(self, input_image):
        left_x, right_x, left_y, right_y = self.ImageArea(input_image) # ImageArea 호출.
        crop_img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
        crop_img = crop_img[left_y:right_y, left_x:right_x]
        
        print(f"크롭이미지 쉐입 {crop_img.shape}")
        '''
        #  .jpg파일 형변환
        cv2.imwrite('test.png',crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"크롭이미지 쉐입은 {crop_img.shape}")
        '''
        # 배경 검은색으로 바꿔주는 작업
        trans_mask = crop_img[:,:,3]==0 # png 쉐입은 (,,4)이지만 jpg/jpeg는 쉐입이 (,,3)이기 때문에 [:,:,3]이 인덱스초과
        crop_img[trans_mask] = [0, 0, 0, 0]
        # crop_img[trans_mask] = [255, 255, 255, 255] # 이건 흰색

        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGRA2BGR)

        # 이미지 결과 출력
        print(f"크롭이미지 Type: {type(crop_img)}")
        cv2.imshow('crop_img',crop_img)

        return crop_img
    

# 해당 원형 이미지내 text 유무 판별 ( ImageProcess클래스를 활용 )
class ImageContour():
    # 초기화 : 이미지 전처리 클래스를 상속.
    def __init__(self):
        self.m_cImageProcess = ImageProcess()

    # 이미지 Canny: 이미지 전처리 & 전처리한 이미지에서 물체의 edge를 찾음
    def ImageCanny(self, _inputImage, _option = 1):
        inputImage = self.m_cImageProcess.CropShape(_inputImage) # 전처리(크롭)한 이미지 반환
        inputImage = cv2.resize(inputImage, (200, 200), interpolation=cv2.INTER_LINEAR) # 이미지 200x200 사이즈로 변환

        if _option == 1: 
            filterImage = cv2.bilateralFilter(inputImage, 9, 75, 75) # 엣지 보존 블러링 - 기본
            filterImage = cv2.medianBlur(filterImage, 7) # salt-pepper 노이즈 제거 블러링
            filterImage = cv2.cvtColor(filterImage, cv2.COLOR_BGRA2BGR) # 색선명

            clahe = self.m_cImageProcess.max_con_CLAHE(filterImage) # 색 필터링
            canny = cv2.Canny(clahe, 50, 200) # 이미지 엣지 추출 - 엣지를 크게 판단

        elif _option == 2:
            filterImage = cv2.bilateralFilter(inputImage, 9, 39, 39) # 엣지 보존 블러링 - 좀더세게블러링
            filterImage = cv2.medianBlur(filterImage, 7) # salt-pepper 노이즈 제거 블러링
            filterImage = cv2.cvtColor(filterImage, cv2.COLOR_BGRA2BGR) # 색선명

            clahe = self.m_cImageProcess.max_con_CLAHE(filterImage) # 색 필터링
            canny = cv2.Canny(clahe, 50, 100) # 이미지 엣지 추출 - 섬세한 것도 엣지로 판단

        # 이미지 전처리 결과 및 엣지 반환
        return inputImage, canny 

    #  컨투어 체크 이미지의 contour의 길이를 확인하는 함수
    def ContourCheck(self, _canny, _proportion):
        contours, _ = cv2.findContours(_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # 컨투어와 컨투어계층구조를 얻음
        canny = cv2.cvtColor(_canny, cv2.COLOR_GRAY2BGR) # 흑백 --> BGR(원본색)으로

        newContours = list()
        overContours = list()
        newContours.append(contours[0])
        maxLine = 0
        total_contour = []

        """ remove noise options 노이즈 제거"""
        for line in range(len(contours)):
            total_contour.append(len(contours[line]))
            if len(contours[maxLine]) < len(contours[line]):  # 가장 긴 contour 업데이트 (# 가장 긴 컨투어는 shape이라고 판단하여 text contour에서 제외)
                maxLine = line

            if len(contours[line]) > 50 and len(contours[line]) < 500: # 일정 기준보다 짧은 contour는 noise로 판단하여 text contour에서 제외
                newContours.append(contours[line])

            if len(contours[line]) > 500:
                overContours.append(contours[line])  # 500보다 긴 건 overContour
                
        newContours[0] = contours[maxLine]
        
        """ exchange a wrong contour (캡슐 접합부 같은 거) """
        for line in range(len(overContours)):
            if np.array_equal(newContours[0], overContours[line]):
                del overContours[line] 
                break

        if self.CompareContours(newContours[0], canny, _proportion): # if True
            for line in range(len(overContours)):
                if self.CompareContours(overContours[line], canny, _proportion) == False:
                    newContours.insert(0, overContours[line])
                    break

        result, total_coordinate_cnt = self.PillSideCheck(newContours, canny, _proportion)

        """ draw contours 컨투어 그려주기 """
        imageContours = cv2.drawContours(canny, newContours[1:], -1, (0, 255, 0), 2)
        height, width, _ = imageContours.shape
        yellowColor = (0, 255, 255)
        imageContours = cv2.rectangle(imageContours, (int(width//_proportion), int(height//_proportion)), (int(width-width//_proportion), int(height-height//_proportion)), yellowColor, 2)

        imageContours = cv2.drawContours(imageContours, newContours[0], -1, (0, 0, 255), 2)

        return imageContours, result, total_coordinate_cnt

    # 컨투어 비교: contour 길이를 비교해서 가장 긴 contour를 찾는 함수
    def CompareContours(self, _contour, _inputImage, _proportion):
        height, width, _ = _inputImage.shape
        rectRightX, rectLeftX = int(width - width//_proportion), int(width//_proportion)
        rectRightY, rectLeftY = int(height - height//_proportion), int(height//_proportion)

        coordinateCnt = 0
        contour = (_contour.transpose()).reshape(2, -1)

        for coordinates in range(len(_contour)):
            if contour[0][coordinates] < rectRightX and contour[0][coordinates] > rectLeftX and contour[1][coordinates] < rectRightY and contour[1][coordinates] > rectLeftY:
                coordinateCnt += 1

        if coordinateCnt/len(_contour) > 0.5:
            result = True

        else:
            result = False

        return result

    # 이미지의 중앙에 가상의 사각형을 그린 다음 사각형 안에 들어가는 contour의 비율 확인
    # 특정 기준 이상이면 해당 contour가 text contour라고 판별
    def PillSideCheck(self, _contours, _inputImage, _proportion):
        oneside = "ONESIDE"

        contours = _contours[1:]
        total_coordinate_cnt = 0

        if len(contours) == 0:
            result = oneside

        else:
            result = oneside
            
            height, width, _ = _inputImage.shape
            rectRightX, rectLeftX = int(width - width//_proportion), int(width//_proportion)
            rectRightY, rectLeftY = int(height - height//_proportion), int(height//_proportion)

            coordinateCnt_list = []
            coordinateCnt_total = []
            for line in range(len(contours)):
                contour = (contours[line].transpose()).reshape(2, -1)
                coordinateCnt_total.append(len(contour[0]))
                coordinateCnt = 0

                for coordinates in range(len(contours[line])):
                    if contour[0][coordinates] > rectLeftX and contour[0][coordinates] < rectRightX and contour[1][coordinates] > rectLeftY and contour[1][coordinates] < rectRightY:
                        coordinateCnt += 1

                coordinateCnt_list.append(coordinateCnt)
                total_coordinate_cnt += coordinateCnt

            for i in range(len(coordinateCnt_list)):
                if coordinateCnt_list[i]/coordinateCnt_total[i] > 0.5:
                    result = oneside
                    break

        return result, total_coordinate_cnt

    # 이걸 실행해줄거임
    def Process(self, _openPath, _proportion):
        # open_image = cv2.imread(_openPath, cv2.IMREAD_UNCHANGED) # 이미지 열고 (str타입 --> np.array타입)
        inputImages, canny = self.ImageCanny(_openPath) # 이미지 전처리 & 찾은 엣지 반환
        imageContours, result, total_coordinate_cnt = self.ContourCheck(canny, _proportion) # 엣지 컨투어 조정
        
        return imageContours, result, total_coordinate_cnt # 컨투어 체크 결과 반환
    
    
