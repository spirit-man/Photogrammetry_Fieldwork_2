import cv2 as cv
import numpy as np
from cus_forstner import Cus_Forstner


class Cus_Corr_Coefficient(object):
    def __init__(self, threshold=0.9, m=5, n=5) -> None:
        self.threshold = threshold
        self.m = m
        self.n = n
        pass

    def cornerHarris(self, img):
        corners = cv.cornerHarris(img, 2, 7, 0.04)
        harris = np.empty(shape=[0, 2])
        for i in range(np.shape(corners)[0]):
            for j in range(np.shape(corners)[1]):
                if corners[i, j] > 0.15*corners.max():
                    harris = np.append(harris, np.expand_dims(
                        np.asarray((i, j)), axis=0), axis=0)
        return harris

    def cornerForstner(self, img):
        corners = Cus_Forstner()
        forstner = corners(img)
        return forstner

    def corr_coefficient(self, img1, img2):
        gray1, gray2 = np.array(cv.cvtColor(img1, cv.COLOR_BGR2GRAY)), np.array(cv.cvtColor(
            img2, cv.COLOR_BGR2GRAY))

        # use Harris operator to detect corner
        # corner1, corner2 = self.cornerHarris(gray1), self.cornerHarris(gray2)
        # use Forstner operator to detect corner
        corner1, corner2 = self.cornerForstner(
            gray1), self.cornerForstner(gray2)

        N = self.m*self.n
        m1, n1 = int(self.m/2), int(self.n/2)

        # fill in the blanks
        gray1 = np.concatenate(
            (gray1, np.zeros((np.shape(gray2)[0]-np.shape(gray1)[0], np.shape(gray1)[1]))), axis=0)
        gray1 = np.concatenate(
            (gray1, np.zeros((np.shape(gray1)[0], np.shape(gray2)[1]-np.shape(gray1)[1]))), axis=1)
        corr_img = np.concatenate((gray1, gray2), axis=1)

        # calculate correlation coefficient parameters for each point pair
        for point1 in corner1[:]:
            p1x, p1y = int(point1[0]), int(point1[1])
            if p1x < m1 or p1x > np.shape(gray1)[0]-m1-1 or p1y < n1 or p1y > np.shape(gray1)[1]-n1-1:
                continue
            start_x, start_y = p1x-m1, p1y-n1
            S_g1g1 = np.sum(np.multiply(
                gray1[start_x:self.m+start_x, start_y:self.n+start_y], gray1[start_x:self.m+start_x, start_y:self.n+start_y]))
            S_g1 = np.sum(
                gray1[start_x:self.m+start_x, start_y:self.n+start_y])
            for point2 in corner2[:]:
                p2x, p2y = int(point2[0]), int(point2[1])
                if p2x < m1 or p2x > np.shape(gray2)[0]-m1-1 or p2y < n1 or p2y > np.shape(gray2)[1]-n1-1:
                    continue
                r, c = p2x-p1x, p2y-p1y
                S_g1g2 = np.sum(np.multiply(
                    gray1[start_x:self.m+start_x, start_y:self.n+start_y], gray2[r+start_x:r+start_x+self.m, c+start_y:c+start_y+self.n]))
                S_g2g2 = np.sum(np.multiply(
                    gray2[r+start_x:r+start_x+self.m, c+start_y:c+start_y+self.n], gray2[r+start_x:r+start_x+self.m, c+start_y:c+start_y+self.n]))
                S_g2 = np.sum(gray2[r+start_x:r+start_x +
                              self.m, c+start_y:c+start_y+self.n])
                p = (S_g1g2-S_g1*S_g2/N) / \
                    np.sqrt(abs((S_g1g1-S_g1**2/N)*(S_g2g2-S_g2**2/N)))
                if p > self.threshold:
                    cv.line(corr_img, (p1x, p1y),
                            (p2x+np.shape(gray1)[0], p2y), (255, 0, 0), 1)
        return corr_img

    def __call__(self, img1, img2):
        return self.corr_coefficient(img1, img2)
