import cv2 as cv
import numpy as np


def read_images(image_path1, image_path2):
    img1 = cv.imread(image_path1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(image_path2, cv.IMREAD_GRAYSCALE)
    return img1, img2


def extract_features(img):
    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def match_features(desc1, desc2):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def filter_matches(matches, keypoints1, keypoints2, threshold=2.5):
    good_matches = []
    for m in matches:
        if m.distance < threshold * matches[0].distance:
            good_matches.append(m)
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    return good_matches, points1, points2


def compute_fundamental_matrix(points1, points2, use_ransac=True, confidence=0.99, ransac_reproj_threshold=3.0):
    if use_ransac:
        method = cv.FM_RANSAC
    else:
        method = cv.FM_8POINT

    F, mask = cv.findFundamentalMat(points1, points2, method, ransac_reproj_threshold, confidence)
    return F, mask


def compute_relative_orientation(E):
    e1, e2, e3 = E[0, 0], E[0, 1], E[0, 2]
    e4, e5, e6 = E[1, 0], E[1, 1], E[1, 2]
    e7, e8, e9 = E[2, 0], E[2, 1], E[2, 2]

    BX_square = e4**2 + e5**2 + e6**2 + e7**2 + e8**2 + e9**2 - e1**2 - e2**2 - e3**2
    BX = np.sqrt(abs(BX_square) / 2)
    BY = -(e1 * e4 + e2 * e5 + e3 * e6) / BX
    BZ = -(e1 * e7 + e2 * e8 + e3 * e9) / BX

    a1 = (e5 * e9 - e6 * e8 + BZ * e4 - BY * e7) / (BX**2 + BY**2 + BZ**2)
    b1 = (e7 + BY * a1) / BX
    c1 = (BZ * a1 - e4) / BX

    a2 = (e6 * e7 - e4 * e9 + BZ * e5 - BY * e8) / (BX**2 + BY**2 + BZ**2)
    b2 = (e8 + BY * a1) / BX
    c2 = (BZ * a1 - e5) / BX

    a3 = (e4 * e8 - e5 * e7 + BZ * e6 - BY * e9) / (BX**2 + BY**2 + BZ**2)
    b3 = (e9 + BY * a1) / BX
    c3 = (BZ * a1 - e6) / BX

    phi = -np.arctan2(a3, c3)
    omega = -np.arcsin(b3 - int(b3))
    kappa = np.arctan2(b1, b2)

    return phi, omega, kappa, BY, BZ


def main():
    img1, img2 = read_images("../LOR50.bmp", "../LOR49.bmp")
    keypoints1, descriptors1 = extract_features(img1)
    keypoints2, descriptors2 = extract_features(img2)
    # K = np.array([[1150, 0, 225.0],
    #               [0, 1150, 225.0],
    #               [0, 0, 1]])

    K = np.array([[1, 0,-225.0],
                  [0, -1, 225.0],
                  [0, 0, -1150]])

    matches = match_features(descriptors1, descriptors2)
    filtered_matches, points1, points2 = filter_matches(matches, keypoints1, keypoints2)

    F, mask = compute_fundamental_matrix(points1, points2, use_ransac=True)
    points1 = points1[mask.ravel() == 1]
    points2 = points2[mask.ravel() == 1]
    
    E = K.T @ F @ K
    phi, omega, kappa, by, bz = compute_relative_orientation(E)

    print("by:", by, "bz:", bz)
    print("phi (俯仰角):", phi, "omega (翻滚角):", omega, "kappa (偏航角):", kappa)

    img_ori_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('Original_Matches', img_ori_matches)
    img_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, filtered_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('Matches', img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
