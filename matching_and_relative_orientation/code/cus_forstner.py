import numpy as np
import statistics as sta
from scipy import linalg
# from numba import jit


class Cus_Forstner(object):
    def __init__(self, kernel_size=7,
                 threshold_for_q=0.5,
                 maximum_kernel_size=9,
                 threshold_for_pre_select=20):
        self._threshold_q = threshold_for_q
        self._kernel_size = kernel_size
        self._maximum_kernel_size = maximum_kernel_size
        self._t_for_pre_s = threshold_for_pre_select

    def cus_forstner_corner(self, image):
        k = int(self._kernel_size/2)

        # extract pre_corners
        pre_corner, forstner_corner = {}, np.empty(shape=[0, 2])
        for i in range(k, np.shape(image)[0]-k):
            for j in range(k, np.shape(image)[1]-k):
                d_g1 = abs(image[i, j]-image[i+1, j])
                d_g2 = abs(image[i, j]-image[i, j+1])
                d_g3 = abs(image[i, j]-image[i-1, j])
                d_g4 = abs(image[i, j]-image[i, j-1])
                M = sta.median([d_g1, d_g2, d_g3, d_g4])
                if M > self._t_for_pre_s:
                    # calculate N and q
                    gu2, gv2, gugv = 0, 0, 0
                    for i_win in range(i-k, i+k):
                        for j_win in range(j-k, j+k):
                            u = image[i_win+1, j_win+1]-image[i_win, j_win]
                            v = image[i_win, j_win+1]-image[i_win+1, j_win]
                            gu2 += u**2
                            gv2 += v**2
                            gugv += u*v
                    N = np.array([[gu2, gugv],
                                  [gugv, gv2]], dtype=np.float32)
                    if linalg.det(N) == 0:
                        continue
                    Q = linalg.inv(N)
                    q = 4*linalg.det(N)/((np.trace(N))**2)
                    if q > self._threshold_q:
                        w = 1/np.trace(Q)
                        pre_corner[(i, j)] = w

                else:
                    pass

        # extract maximum w in window
        k2 = int(self._maximum_kernel_size/2)
        for key, value in pre_corner.items():
            maxw = value
            maxkey = key
            for c in range(key[0]-k2, key[0]+k2):
                for r in range(key[1]-k2, key[1]+k2):
                    if (c, r) in pre_corner:
                        if pre_corner[(c, r)] > maxw:
                            maxkey = (c, r)
            if maxkey == key:
                forstner_corner = np.append(forstner_corner, np.expand_dims(
                    np.asarray(maxkey), axis=0), axis=0)

        return forstner_corner

    def __call__(self, image):
        return self.cus_forstner_corner(image)
