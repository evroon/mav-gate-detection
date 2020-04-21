import numpy as np
import cv2
import imageio
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10


class detection:
    def apply_filter(self, img):
        ''' Create the filter mask and apply it to the image '''
        center_blue = np.uint8([[[255, 115, 0]]])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        center_blue = cv2.cvtColor(center_blue, cv2.COLOR_BGR2HSV)

        upper_blue = center_blue + np.array([10, 100, 100])
        lower_blue = center_blue - np.array([10, 100, 100])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        img = cv2.bitwise_and(img, img, mask=mask)

        img = cv2.GaussianBlur(img, (25, 25), 0)

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        img = cv2.bitwise_and(img, img, mask=mask)
        return img

    def __init__(self, train_index=265):
        self.gate_image = cv2.imread('gate.png')
        self.train_index = train_index

    def detect_gate(self):
        img = cv2.imread('WashingtonOBRace/img_{}.png'.format(self.train_index))

        if not isinstance(img, np.ndarray):
            return False

        img = self.apply_filter(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, threshold = cv2.threshold(img, 10, 100, 0)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.imread('WashingtonOBRace/img_{}.png'.format(self.train_index))

        pillars = []
        color = (0, 0, 255)

        for contour in contours:
            width = np.max(contour[:, 0, 0]) - np.min(contour[:, 0, 0])
            height = np.max(contour[:, 0, 1]) - np.min(contour[:, 0, 1])

            if height > 2 * width:
                heighest = np.argmax(contour[:, 0, 1])
                lowest = np.argmin(contour[:, 0, 1])

                top_outer = np.array([contour[heighest, 0, 0], contour[heighest, 0, 1]], dtype=np.float)
                bottom_outer = np.array([contour[lowest, 0, 0], contour[lowest, 0, 1]], dtype=np.float)

                center_outer = 0.5 * (top_outer + bottom_outer)
                center_x = 0.5 * (np.min(contour[:, 0, 0]) + np.max(contour[:, 0, 0]))

                parallel = top_outer - bottom_outer
                bottom_extrapolated = center_outer - parallel
                top_extrapolated = center_outer + parallel

                left_pillar = True

                # Right pillar has outer edge right from center
                if top_outer[0] > center_x:
                    left_pillar = False

                top_extrapolated = (int(top_extrapolated[0]), int(top_extrapolated[1]))
                bottom_extrapolated = (int(bottom_extrapolated[0]), int(bottom_extrapolated[1]))


                pillars.append([left_pillar, top_extrapolated, bottom_extrapolated, height])

        for i in pillars:
            if i[0]:
                leftpillar = i
                for j in pillars:
                    if not j[0] and j[3] < 1.25 * i[3] and j[3] > 0.75 * i[3]:
                        rightpillar = j

                        topleft = leftpillar[1]
                        bottomleft = leftpillar[2]
                        topright = rightpillar[1]
                        bottomright = rightpillar[2]

                        rectangle = np.array([[
                            [bottomleft],
                            [topleft],
                            [topright],
                            [bottomright]
                        ]], dtype=np.int32)
                        cv2.drawContours(img, rectangle, 0, color, 1)

        return img


for image in range(438):
    det = detection(image)
    img = det.detect_gate()

    if not isinstance(img, bool):
        print('saving {}'.format(image))
        cv2.imwrite('results/result_{}.png'.format(str(image)), img)
