import numpy as np
import cv2
import imageio
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


MIN_MATCH_COUNT = 10


class detection:
    def apply_filter(self, img):
        ''' Create the filter mask and apply it to the image '''
        center_blue = np.uint8([[[255, 115, 0]]])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        center_blue = cv2.cvtColor(center_blue, cv2.COLOR_BGR2HSV)

        upper_blue = center_blue + np.array([10, 100, 100])
        lower_blue = center_blue - np.array([10, 110, 110])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        img = cv2.bitwise_and(img, img, mask=mask)

        # img = cv2.GaussianBlur(img, (25, 25), 0)

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        img = cv2.bitwise_and(img, img, mask=mask)
        return img

    def detect_gate(self, train_index=265):
        self.train_index = train_index

        img = cv2.imread('WashingtonOBRace/img_{}.png'.format(self.train_index))
        mask = cv2.imread('WashingtonOBRace/mask_{}.png'.format(self.train_index))

        if not isinstance(img, np.ndarray):
            return False, np.nan, np.nan

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        img = self.apply_filter(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, threshold = cv2.threshold(img, 10, 100, 0)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.imread('WashingtonOBRace/img_{}.png'.format(self.train_index))

        pillars = []
        true_positives = 0
        false_positives = 0

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
                left_pillar = top_outer[0] < center_x

                top_extrapolated = (int(top_extrapolated[0]), int(top_extrapolated[1]))
                bottom_extrapolated = (int(bottom_extrapolated[0]), int(bottom_extrapolated[1]))


                pillars.append([left_pillar, top_extrapolated, bottom_extrapolated, height])

        used_right_pillars = []
        color = (0, 0, 255)

        output_img = np.zeros((img.shape[0], img.shape[1]))

        for i in range(len(pillars)):
            leftpillar = pillars[i]

            if leftpillar[0]:
                for j in range(0, len(pillars)):
                    rightpillar = pillars[j]

                    # pillar must not be already used, be a right pillar,
                    if j not in used_right_pillars and \
                        not rightpillar[0] and \
                        rightpillar[3] < 1.25 * leftpillar[3] and \
                        rightpillar[3] > 0.75 * leftpillar[3]:

                        topleft = np.array(leftpillar[1])
                        bottomleft = np.array(leftpillar[2])
                        topright = np.array(rightpillar[1])
                        bottomright = np.array(rightpillar[2])

                        topleft_inner = topleft + (bottomright - topleft) * 0.15
                        bottomleft_inner = bottomleft + (topright - bottomleft) * 0.15
                        topright_inner = topright + (bottomleft - topright) * 0.15
                        bottomright_inner = bottomright + (topleft - bottomright) * 0.15

                        width = bottomright[0] - bottomleft[0]
                        height = topleft[1] - bottomleft[1]

                        # Calculate lengths of outer edges
                        edge_length = np.zeros(4)
                        edge_length[0] = np.sqrt(np.sum((topright - topleft) ** 2))
                        edge_length[1] = np.sqrt(np.sum((topleft - bottomleft) ** 2))
                        edge_length[2] = np.sqrt(np.sum((bottomleft - bottomright) ** 2))
                        edge_length[3] = np.sqrt(np.sum((bottomright - topright) ** 2))

                        if (topleft[0] < 2 and bottomleft[0] < 2) or \
                            width < 30 or \
                            width > height * 20 or \
                            width < height // 20 or \
                            np.absolute(topleft[1] - topright[1]) > 35 or \
                            np.std(edge_length) > 20:
                            continue

                        used_right_pillars.append(j)

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

                        polygon_outer = [*bottomleft, *topleft, *topright, *bottomright]
                        polygon_inner = [*bottomleft_inner, *topleft_inner, *topright_inner, *bottomright_inner]

                        draw_img = Image.new('L', img.shape[1::-1], 0)
                        draw = ImageDraw.Draw(draw_img)
                        draw.polygon(polygon_outer, outline=1, fill=1)
                        draw.polygon(polygon_inner, outline=1, fill=0)
                        draw_img = np.array(draw_img)

                        output_img = output_img + draw_img

                        img[np.array(draw_img) == 1, :] = [0, 0, 255]

        output_img[output_img > 1] = 1
        true_positives += np.sum(output_img * mask)
        false_positives += np.sum(output_img * (255 - mask))

        return img, true_positives / np.sum(mask), false_positives / np.sum(255 - mask)

    def process(self):
        plot_data = []

        for image in range(500):
            img, TPR, FPR = det.detect_gate(image)
            plot_data.append((FPR, TPR))

            if not isinstance(img, bool):
                cv2.imwrite('results/result_{}.png'.format(str(image)), img)

        np.save('plot_data', plot_data)


    def plot(self):
        plot_data = np.load('plot_data.npy')

        x = [item[0] for item in plot_data]
        y = [item[1] for item in plot_data]

        plt.plot(x, y, ls='', marker='o')
        plt.title('ROC (TPR versus FPR)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid()
        # plt.hist(np.array(ious), bins=np.linspace(0, 100, 50))
        # plt.grid()
        plt.savefig('roc', bbox_inches='tight')




det = detection()
det.process()
det.plot()