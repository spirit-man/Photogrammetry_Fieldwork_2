from sys import flags
import cv2 as cv
import numpy as np
from cus_forstner import Cus_Forstner


def read_img(*img_path):
    imgs = []
    for img in img_path:
        imgs.append(np.float32(cv.imread(img)))
    return imgs


def show_img(**imgs):
    for key, value in imgs.items():
        cv.imshow(key, value/255)


def cus_pipeline(**func):
    # call mutiple feature extraction algorithms at one time
    imgs = []
    for key, value in func.items():
        temp = np.copy(value)
        gray = cv.cvtColor(value, cv.COLOR_BGR2GRAY)
        img8bit = cv.normalize(
            gray, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        # blur image and set thresholds for edge operators
        kernel = np.ones((5, 5), np.float32)/25
        imgblurred = cv.filter2D(img8bit, -1, kernel)
        t_lower = 50
        t_upper = 150
        if key == "cv.cornerHarris":
            corners = eval(key+"(gray, 2, 7, 0.04)")
            imgHarris = cv.dilate(corners, np.ones((1, 1)))
            temp[imgHarris > 0.15*imgHarris.max()] = [255, 0, 0]
        elif key == "cv.SIFT":
            sift = cv.SIFT_create()
            kp = sift.detect(img8bit, None)
            temp = cv.drawKeypoints(img8bit, kp, temp)
        elif key == "cv.SURF":
            # SURF is still patented, so this codes cannot be excecuted in opencv, you have to build it from src (with opencv contrib modules)
            surf = cv.xfeatures2d.SURF_create(400)
            kp, des = surf.detectAndCompute(img8bit, None)
            temp = cv.drawKeypoints(img8bit, kp, temp)
        elif key == "cv.FAST":
            fast = cv.FastFeatureDetector_create()
            kp = fast.detect(temp, None)
            temp = cv.drawKeypoints(img8bit, kp, temp)
        elif key == "cv.ORB":
            orb = cv.ORB_create(nfeatures=100000, scoreType=cv.ORB_FAST_SCORE)
            kp = orb.detect(temp, None)
            temp = cv.drawKeypoints(
                img8bit, kp, None, color=(255, 255, 255), flags=cv.DrawMatchesFlags_DEFAULT)
        elif key == "cv.Canny":
            temp = cv.Canny(imgblurred, t_lower, t_upper)
        elif key == "cv.HoughLines":
            edges = cv.Canny(imgblurred, t_lower, t_upper)
            lines = cv.HoughLinesP(
                edges,
                1,
                np.pi/180,
                threshold=100,
                minLineLength=5,
                maxLineGap=10
            )
            for points in lines:
                x1, y1, x2, y2 = points[0]
                cv.line(temp, (x1, y1), (x2, y2), (255, 0, 0), 1)
        elif key == "Cus_Forstner":
            forstner = Cus_Forstner()
            corners = forstner(gray)
            imgForstner = cv.dilate(corners, np.ones((1, 1)))
            for item in imgForstner:
                item = [int(x) for x in item]
                temp[item[0], item[1], :] = [255, 0, 0]
        else:
            pass
        imgs.append(np.float32(temp))

    return imgs
