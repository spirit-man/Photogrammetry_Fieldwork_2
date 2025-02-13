from concurrent.futures import ThreadPoolExecutor
import cv2 as cv
from utils import(
    read_img,
    show_img,
    cus_pipeline,
)
from cus_correlation_coefficient import Cus_Corr_Coefficient


# load data here
IMG1_PATH = "../LOR49.bmp"
IMG2_PATH = "../LOR50.bmp"


def main():
    # read images
    imgs = read_img(IMG1_PATH, IMG2_PATH)

    def cal_feature(i):
        img = imgs[i]

        # call feature extraction algorithms
        funcs = {
            # "cv.Canny": img,
            "cv.cornerHarris": img,
            "Cus_Forstner": img,
            "cv.FAST": img,
            # "cv.HoughLines": img,
            "cv.ORB": img,
            "cv.SIFT": img
            # "cv.SURF": img,
        }
        dst = cus_pipeline(**funcs)
        # key in dsts should be arranged from A to Z
        dsts = {
            "img"+str(i): img,
            # "dst"+str(i)+"_cv.Canny": dst[0],
            "dst"+str(i)+"_cv.cornerHarris": dst[0],
            "dst"+str(i)+"_Cus_Forstner": dst[1],
            "dst"+str(i)+"_cv.FAST": dst[2],
            # "dst"+str(i)+"_cv.HoughLines": dst[4],
            "dst"+str(i)+"_cv.ORB": dst[3],
            "dst"+str(i)+"_cv.SIFT": dst[4],
            # "dst"+str(i)+"_cv.SURF": dst[7],
        }
        return dsts

    # with ThreadPoolExecutor(len(imgs)) as executor:
    #     for result in executor.map(cal_feature, range(len(imgs))):
    #         show_img(**result)
    #     cv.waitKey(100)

    # calculate correlation coefficient
    corrs = Cus_Corr_Coefficient()
    corr_img = corrs(imgs[0], imgs[1])
    pair = {
        "img_pair": corr_img
    }
    show_img(**pair)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
