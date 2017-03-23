import cv2
import numpy as np
import skimage.morphology as sm


class PlateExtraction:

    def __init__(self):
        pass

    @staticmethod
    def opening_by_reconstruction(img, kernel):

        return sm.reconstruction(cv2.erode(img.copy(), kernel), img.copy(), 'dilation')

    @staticmethod
    def closing_by_reconstruction(img, kernel):

        return sm.reconstruction(cv2.dilate(img.copy(), kernel), img.copy(), 'erosion')

    @staticmethod
    def double_threshold(img, a, b, c, d):

        ret, mask = cv2.threshold(img.copy(), a, d, cv2.THRESH_BINARY)
        ret, binarisation = cv2.threshold(img.copy(), b, c, cv2.THRESH_BINARY)

        return sm.reconstruction(binarisation, mask, 'dilation')

    @staticmethod
    def find_plate(img):

        global plate

        img = cv2.resize(img, (648, 486))

        h = 20

        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (h / 2, h / 2))
        kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (4 * h, 1))
        kernel2_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h / 2))
        kernel2_h = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * h, 1))
        kernel23 = cv2.getStructuringElement(cv2.MORPH_RECT, (h, h / 2))

        # First step - contrast enhancement

        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel2)
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel2)

        contrast = cv2.subtract(cv2.add(img, tophat), blackhat)

        # Second step - background cleaning

        tophat = np.subtract(contrast, PlateExtraction.opening_by_reconstruction(contrast, kernel2))
        blackhat = np.subtract(PlateExtraction.closing_by_reconstruction(contrast, kernel2), contrast)

        supremum = np.maximum(tophat, blackhat)

        # Third step - plate area detection

        closing = cv2.morphologyEx(supremum, cv2.MORPH_CLOSE, kernel4)

        opening_v = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2_v)
        opening_h = np.array(cv2.morphologyEx(opening_v, cv2.MORPH_OPEN, kernel2_h), 'uint8')

        m = int(np.amax(opening_h))

        thresh = PlateExtraction.double_threshold(opening_h, m / 2, m - 1, 255, 255)

        dilation = np.array(cv2.dilate(thresh, kernel23), 'uint8')

        im2, cnt, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)

            plate = img.copy()[y:y + h, x:x + w]

        return plate

    @staticmethod
    def comparator(a, b):
        try:
            xa, ya, wa, ha = cv2.boundingRect(a)
            xb, yb, wb, hb = cv2.boundingRect(b)

            if xa > xb:
                return 1
            elif xa == xb:
                return 0
            else:
                return -1
        except ZeroDivisionError:
            return 0

    @staticmethod
    def segment_plate(img):
        h = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h / 2, h / 2))

        tophat = np.subtract(img, PlateExtraction.opening_by_reconstruction(img, kernel))
        blackhat = np.subtract(PlateExtraction.closing_by_reconstruction(img, kernel), img)

        better = tophat if tophat.sum() > blackhat.sum() else blackhat
        better = np.array(better, 'uint8')

        tophat = cv2.morphologyEx(better, cv2.MORPH_TOPHAT, better)
        blackhat = cv2.morphologyEx(better, cv2.MORPH_BLACKHAT, kernel)

        contrast = cv2.subtract(cv2.add(better, tophat), blackhat)

        ret, binarisation = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        im2, cnt, hierarchy = cv2.findContours(binarisation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnt.sort(PlateExtraction.comparator)

        segments = []

        for i in range(len(cnt)):
            x, y, w, h = cv2.boundingRect(cnt[i])

            if h > float(img.shape[0]) * 0.4 \
                    and h > w \
                    and float(w) / h > 0.20:
                segment = (255 - contrast)[y:y + h, x:x + w]

                ret, binary_segment = cv2.threshold(segment, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                segments.append(binary_segment)

        return segments
