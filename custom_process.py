#! /usr/bin/python3
import cv2
import numpy as np


def calc_disparity(left_image, right_image):

    sad_window_size = 9

    stereo = cv2.StereoSGBM_create(


        minDisparity=0,
        numDisparities=112,
        blockSize=3,
        P1 = sad_window_size*sad_window_size*4,
        P2 = sad_window_size*sad_window_size*32,
        disp12MaxDiff = 64,
        uniquenessRatio = 5,
        speckleWindowSize=0,
        speckleRange=32,
        preFilterCap=1




    )



    return stereo.compute(left_image , right_image)

def calculateDisparityForZedCamera(left_image, right_image):
    window_size = 2

    stereo = cv2.StereoSGBM_create(
        minDisparity=1,
        numDisparities=80,
        blockSize=5,
        P1=64,
        P2=1024,
        disp12MaxDiff=1,
        uniquenessRatio=1,
        speckleWindowSize = 0,
        speckleRange=3,
        preFilterCap=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY

    )
    return stereo.compute(left_image , right_image)



def to_gray(im):
    if im.ndim == 3:
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        return im.copy()
def calc_point_cloud(image, disp, q):
    points = cv2.reprojectImageTo3D(disp, q).reshape(-1, 3)
    colors = image.reshape(-1, 3)
    return remove_invalid(disp.reshape(-1), points, colors)



def project_points(points, colors, r, t, k, dist_coeff, width, height):
    projected, _ = cv2.projectPoints(points, r, t, k, dist_coeff)
    xy = projected.reshape(-1, 2).astype(np.int)
    mask = (
        (0 <= xy[:, 0]) & (xy[:, 0] < width) &
        (0 <= xy[:, 1]) & (xy[:, 1] < height)
    )
    return xy[mask], colors[mask]


def calc_projected_image(points, colors, r, t, k, dist_coeff, width, height):
    xy, cm = project_points(points, colors, r, t, k, dist_coeff, width, height)
    image = np.zeros((height, width, 3), dtype=colors.dtype)
    image[xy[:, 1], xy[:, 0]] = cm
    return image


#custom reproject image to 3D
def reprojectImageTo3D(disp, Q):
    points = np.empty((disp.shape[0], disp.shape[1]), dtype=float)


    height, width = disp.shape
    _3Dimage = np.empty((height, width, 3), dtype = float)
    maxz = 10000.0
    for x in range(height):
        for y in range(width):

            d = disp[x][y]
            vector = np.array([x, y, d, 1 ])
            vector = vector.reshape(4,1)

            vector =np.dot(Q, vector)
            if vector[3] == 0:
                _3Dimage[x][y][0] = 50
                _3Dimage[x][y][1] = 50
                _3Dimage[x][y][2] = 50

                continue
            vector = vector/vector[3]


            _3Dimage[x][y][0] = vector[0]
            _3Dimage[x][y][1] = vector[1]
            _3Dimage[x][y][2] = vector[2]



    return _3Dimage.reshape(-1,3)



def replaceDisparity(disparity, replacement):
        height, width = disparity.shape
        for x in range(height):
            for y in range(width):
                if(y == 0 or x == 0 or x == (height-1) or y == (width-1)):
                    disparity[x][y]=0
                if disparity[x][y] < 5:

                    disparity[x][y] = replacement

        return disparity

