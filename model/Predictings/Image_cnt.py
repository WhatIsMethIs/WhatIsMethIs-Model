#### '그 외 shaped' 알약이미지 양/단면 판별 ####
import cv2
import numpy as np
#import ImageSideProcess_circle as ImageProcess

# 사용자 촬영 원형이미지 양/단면 판단 진행 이전 전처리 (이건 circle에서의 ImageProcess클래스를 가져옴)
class ImageProcess():
    def __init__(self):
        pass

    def max_con_CLAHE(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        limg = cv2.merge((cl,a,b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return img


    def ImageArea(self, input_image):
        rgba = cv2.medianBlur(input_image, 55)

        imgray = cv2.cvtColor(rgba, cv2.COLOR_BGRA2GRAY)
        contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_t = contours[0].transpose()

        right_point_x = np.max(contours_t[0]) + 5
        left_point_x = np.min(contours_t[0]) - 5
        right_point_y = np.max(contours_t[1]) + 5
        left_point_y = np.min(contours_t[1]) - 5

        return left_point_x, right_point_x, left_point_y, right_point_y


    def CropShape(self, input_image):
        left_x, right_x, left_y, right_y = self.ImageArea(input_image)
        crop_img = input_image[left_y:right_y, left_x:right_x]

        trans_mask = crop_img[:,:,3]==0
        crop_img[trans_mask] = [0, 0, 0, 0]

        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGRA2BGR)

        return crop_img

# ?해당 이미지 내 contour 개수 return하는 프로세스?
class ImageContour():
    def __init__(self):
        self.m_cImageProcess = ImageProcess.ImageProcess()


    def ImageCanny(self, _inputImage, _option = 1):
        inputImage = self.m_cImageProcess.CropShape(_inputImage)
        
        inputImage = cv2.resize(inputImage, (200, 200), interpolation=cv2.INTER_LINEAR)

        if _option == 1: 
            filterImage = cv2.bilateralFilter(inputImage, 9, 75, 75)
            filterImage = cv2.medianBlur(filterImage, 7)
            filterImage = cv2.cvtColor(filterImage, cv2.COLOR_BGRA2BGR)
            clahe = self.m_cImageProcess.max_con_CLAHE(filterImage)
            canny = cv2.Canny(clahe, 50, 200)

        elif _option == 2:
            filterImage = cv2.bilateralFilter(inputImage, 9, 39, 39)
            filterImage = cv2.medianBlur(filterImage, 7)
            filterImage = cv2.cvtColor(filterImage, cv2.COLOR_BGRA2BGR)
            clahe = self.m_cImageProcess.max_con_CLAHE(filterImage)
            canny = cv2.Canny(clahe, 50, 100)

        return inputImage, canny


    def ContourCheck(self, _canny, _proportion):
        contours, _ = cv2.findContours(_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        canny = cv2.cvtColor(_canny, cv2.COLOR_GRAY2BGR)

        newContours = list()
        overContours = list()
        newContours.append(contours[0])
        maxLine = 0
        total_contour = []

        """ remove noise options """
        for line in range(len(contours)):
            total_contour.append(len(contours[line]))
            if len(contours[maxLine]) < len(contours[line]):
                maxLine = line

            if len(contours[line]) > 50 and len(contours[line]) < 500:
                newContours.append(contours[line])

            if len(contours[line]) > 500:
                overContours.append(contours[line])

        newContours[0] = contours[maxLine]
        
        """ exchange a wrong contour """
        for line in range(len(overContours)):
            if np.array_equal(newContours[0], overContours[line]):
                del overContours[line]
                break

        if self.CompareContours(newContours[0], canny, _proportion): # if True
            for line in range(len(overContours)):
                if self.CompareContours(overContours[line], canny, _proportion) == False:
                    newContours.insert(0, overContours[line])
                    break

        total_coordinate_cnt = self.PillSideCheck(newContours, canny, _proportion)

        return total_coordinate_cnt


    def CompareContours(self, _contour, _inputImage, _proportion):
        height, width, _ = _inputImage.shape
        rectRightX, rectLeftX = int(width - width//_proportion), int(width//_proportion)
        rectRightY, rectLeftY = int(height - height//_proportion), int(height//_proportion)

        coordinateCnt = 0
        contour = (_contour.transpose()).reshape(2, -1) # coordinates[0]: the array of X, coordinates[1]: the array of Y

        for coordinates in range(len(_contour)):
            if contour[0][coordinates] < rectRightX and contour[0][coordinates] > rectLeftX and contour[1][coordinates] < rectRightY and contour[1][coordinates] > rectLeftY:
                coordinateCnt += 1

        if coordinateCnt/len(_contour) > 0.5:
            result = True

        else:
            result = False

        return result


    def PillSideCheck(self, _contours, _inputImage, _proportion):
        contours = _contours[1:]
        total_coordinate_cnt = 0

        height, width, _ = _inputImage.shape
        rectRightX, rectLeftX = int(width - width//_proportion), int(width//_proportion)
        rectRightY, rectLeftY = int(height - height//_proportion), int(height//_proportion)

        for line in range(len(contours)):
            contour = (contours[line].transpose()).reshape(2, -1)
            coordinateCnt = 0

            for coordinates in range(len(contours[line])):
                if contour[0][coordinates] > rectLeftX and contour[0][coordinates] < rectRightX and contour[1][coordinates] > rectLeftY and contour[1][coordinates] < rectRightY:
                    coordinateCnt += 1

            total_coordinate_cnt += coordinateCnt

        return total_coordinate_cnt


    def Process(self, _openPath, _proportion):
        open_image = cv2.imread(_openPath, cv2.IMREAD_UNCHANGED)
        inputImages, canny = self.ImageCanny(open_image)
        total_coordinate_cnt = self.ContourCheck(canny, _proportion)
        
        return total_coordinate_cnt