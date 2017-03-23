import numpy as np
import os
import cv2


class Helpers:

    def __init__(self):
        pass

    @staticmethod
    def crop_image(img):

        img = 255 - img

        crop = [0, 0, img.shape[1], img.shape[0]]

        for i in range(0, img.shape[0]):
            if np.sum(img[i:i + 1, :]) != 0:
                crop[0] = i
                break

        for i in range(img.shape[0], 0, -1):
            if np.sum(img[i - 1:i, :]) != 0:
                crop[3] = i
                break

        for i in range(0, img.shape[1]):
            if np.sum(img[:, i:i + 1]) != 0:
                crop[1] = i
                break

        for i in range(img.shape[1], 0, -1):
            if np.sum(img[:, i - 1:i]) != 0:
                crop[2] = i
                break

        return (255 - img)[crop[0]:crop[3], crop[1]:crop[2]]

    @staticmethod
    def img2data(img, offset=0):

        pattern = np.full((1, 35), 0, dtype=np.uint8)
        pattern[0][offset] = 1

        return [img.ravel(), pattern[0]]

    @staticmethod
    def learning2data():

        learning = []
        outs = []
        i = 0

        for file in os.listdir("./learning_data"):
            img = cv2.imread("./learning_data/" + file, cv2.CV_8U)

            img = Helpers.crop_image(img)
            img = cv2.resize(img, (34, 50), interpolation=cv2.INTER_AREA)

            ret = Helpers.img2data(img, i)
            learning.append(ret[0])
            outs.append(ret[1])

            i += 1

        return [np.asarray(learning), np.asarray(outs)]
