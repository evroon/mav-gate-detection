import numpy as np
import cv2
import imageio
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10


class detection:
    def __init__(self):
        train_index = 429

        self.gate_image = cv2.imread('gate.png')
        self.train_image = cv2.imread('WashingtonOBRace/img_{}.png'.format(train_index))

    def sift(self):
        # Initiate SIFT detector
        detector = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        self.kp1, des1 = detector.detectAndCompute(self.gate_image, None)
        self.kp2, des2 = detector.detectAndCompute(self.train_image, None)
        des1 = np.float32(des1)
        des2 = np.float32(des2)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                            trees=5,
                            # algorithm = FLANN_INDEX_LSH,
                            table_number = 6, # 12
                            key_size = 12,     # 20
                            multi_probe_level = 1)
        search_params = dict(checks=50)

        print(len(des1), len(des2))

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        self.matches = good
        
        print('{}/{}'.format(len(self.matches), len(matches)))

    def transform(self):
        if len(self.matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([self.kp1[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            self.matchesMask = mask.ravel().tolist()

            h, w = self.gate_image.shape[:2]
            pts = [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
            pts = np.float32(pts).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            self.train_image = cv2.polylines(self.train_image, [np.int32(dst)], True, 255, 1, cv2.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(self.matches), MIN_MATCH_COUNT))
            self.matchesMask = None

    def draw(self):
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=self.matchesMask,  # draw only inliers
                           flags=2)

        result = cv2.drawMatches(self.gate_image, self.kp1,
                                 self.train_image, self.kp2,
                                 self.matches, None, **draw_params)

        if not cv2.imwrite('result.png', result):
            raise Exception('failed saving result.png')


detection = detection()
detection.sift()
detection.transform()
detection.draw()
