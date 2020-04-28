import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import imageio
import warnings
import timeit
import os


class detection:
    def __init__(self, save_output=True):
        self.save_output = save_output


    def apply_filter(self, img):
        '''
        Filter the sides of the gates and apply it to the image.

        Args:
            img(Image): the image to filter

        Returns:
            img(Image): the image where the side of the gates are filtered
        '''

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([96, 145, 145])
        upper_blue = np.array([116, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        img = cv2.bitwise_and(img, img, mask=mask)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def detect_gate(self, image_index=265, threshold=0.15):
        '''
        Detect the gates in the image given by image_index.

        Args:
            image_index(int): which image to load
            threshold(float): the maximum penalty for gates to be accepted

        Returns:
            img: the original image with gates segmented in red
            TPR: the true positive ratio
            FPR: the false positive ratio
        '''

        img = cv2.imread('WashingtonOBRace/img_{}.png'.format(image_index))

        # Stop if image could not be read.
        if not isinstance(img, np.ndarray):
            return False, np.nan, np.nan

        if self.save_output:
            mask = cv2.imread('WashingtonOBRace/mask_{}.png'.format(image_index))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        filtered_img = self.apply_filter(img)

        # Calculate the contours.
        _, threshold_img = cv2.threshold(filtered_img, 50, 100, 0)
        contours, _ = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        pillars = []

        # Process all contours.
        for contour in contours:
            width = np.max(contour[:, 0, 0]) - np.min(contour[:, 0, 0])
            height = np.max(contour[:, 0, 1]) - np.min(contour[:, 0, 1])

            if height < 1.7 * width:
                continue

            # Calculate the extrema of the contour.
            heighest = np.argmax(contour[:, 0, 1])
            lowest = np.argmin(contour[:, 0, 1])

            top_outer = np.array([contour[heighest, 0, 0], contour[heighest, 0, 1]], dtype=np.float)
            bottom_outer = np.array([contour[lowest, 0, 0], contour[lowest, 0, 1]], dtype=np.float)

            center_outer = 0.5 * (top_outer + bottom_outer)
            center_x = 0.5 * (np.min(contour[:, 0, 0]) + np.max(contour[:, 0, 0]))

            # Extrapolate the calculated vertices to get the vertices of the entire pillar.
            parallel = top_outer - bottom_outer
            bottom_extrapolated = center_outer - parallel
            top_extrapolated = center_outer + parallel

            # Right pillar has an outer edge right from center
            left_pillar = top_outer[0] < center_x

            top_extrapolated = (int(top_extrapolated[0]), int(top_extrapolated[1]))
            bottom_extrapolated = (int(bottom_extrapolated[0]), int(bottom_extrapolated[1]))

            pillars.append([left_pillar, top_extrapolated, bottom_extrapolated, height])

        output_img = np.zeros((img.shape[0], img.shape[1]))
        used_right_pillars = []

        # Process all detected pillars.
        for leftpillar in pillars:
            penalties = []
            outer_polygons = []
            inner_polygons = []

            topleft = np.array(leftpillar[1])
            bottomleft = np.array(leftpillar[2])
            height = topleft[1] - bottomleft[1]

            if leftpillar[0]:
                for j, rightpillar in enumerate(pillars):
                    # Right pillar must not be already used, be a right pillar and
                    # have approximately the same dimensions as the left pillar.
                    if j not in used_right_pillars and \
                        not rightpillar[0] and \
                        rightpillar[3] < 1.25 * leftpillar[3] and \
                        rightpillar[3] > 0.75 * leftpillar[3]:

                        # Calculate the vertices of the gate in image coordinates.
                        topright = np.array(rightpillar[1])
                        bottomright = np.array(rightpillar[2])

                        topleft_inner = topleft + (bottomright - topleft) * 0.15
                        bottomleft_inner = bottomleft + (topright - bottomleft) * 0.15
                        topright_inner = topright + (bottomleft - topright) * 0.15
                        bottomright_inner = bottomright + (topleft - bottomright) * 0.15

                        width = bottomright[0] - bottomleft[0]

                        # Ignore "gates" that are too small.
                        if width < 30:
                            continue

                        # Calculate lengths of outer edges.
                        edge_length = np.zeros(4)
                        edge_length[0] = np.sqrt(np.sum((topright - topleft) ** 2))
                        edge_length[1] = np.sqrt(np.sum((topleft - bottomleft) ** 2))
                        edge_length[2] = np.sqrt(np.sum((bottomleft - bottomright) ** 2))
                        edge_length[3] = np.sqrt(np.sum((bottomright - topright) ** 2))

                        # Calculate the penalty that is to be minimized.
                        penalty = np.std(edge_length) / np.average(edge_length)

                        used_right_pillars.append(j)
                        penalties.append(penalty)
                        outer_polygons.append([*bottomleft, *topleft, *topright, *bottomright])
                        inner_polygons.append([*bottomleft_inner, *topleft_inner, *topright_inner, *bottomright_inner])

                penalties = np.array(penalties)
                if len(penalties) < 1 or np.min(penalties) > threshold:
                    continue

                # Determine which pillar matches best to the left pillar.
                best_match = np.argmin(penalties)
                polygon_outer = outer_polygons[best_match]
                polygon_inner = inner_polygons[best_match]

                if self.save_output:
                    # Render the polygon of the detected gate onto a new image.
                    draw_img = Image.new('L', img.shape[1::-1], 0)
                    draw = ImageDraw.Draw(draw_img)
                    draw.polygon(polygon_outer, outline=1, fill=1)
                    draw.polygon(polygon_inner, outline=1, fill=0)
                    draw_img = np.array(draw_img)

                    # Save output and overlay polygon in original image in red.
                    output_img = output_img + draw_img
                    img[np.array(draw_img) == 1, :] = [0, 0, 255]


        if self.save_output:
            # Calculate the TPR and FPR values.
            output_img[output_img > 1] = 1
            true_positives = np.sum(output_img * mask)
            false_positives = np.sum(output_img * (255 - mask))
            return img, true_positives / np.sum(mask), false_positives / np.sum(255 - mask)

        return img, np.nan, np.nan


    def process(self, img_list=None, thresholds=[0.15]):
        '''
        Process a list of images and output the resulting segmented images.

        Args:
            img_list (list): The images to process. If it is None, all images will be processed
            thresholds (list): The thresholds for which to calculate the ROC curve

        The TPR and FPR data are stored in the 'results' directory.
        '''

        self.thresholds = thresholds

        for threshold in self.thresholds:
            plot_data = []

            if img_list == None:
                img_list = range(439)

            start = timeit.default_timer()

            for image in img_list:
                img, TPR, FPR = det.detect_gate(image, threshold)
                plot_data.append((FPR, TPR))

                if self.save_output and not isinstance(img, bool):
                    cv2.imwrite('results/result_{}.png'.format(str(image)), img)

            stop = timeit.default_timer()
            print('Computation time per image: ', (stop - start) / 308 * 1e3, 'ms')

            np.save('results/data_{:.2f}'.format(threshold), plot_data)


    def plot(self):
        ''' Plots the ROC curve for different thresholds. '''

        if not self.save_output:
            return

        for threshold in self.thresholds:
            # Ignore warnings about nan values.
            warnings.filterwarnings('ignore')

            # Load data
            data_file = 'results/data_{:.2f}.npy'.format(threshold)
            plot_data = np.load(data_file)

            x = np.array([item[0] for item in plot_data])
            y = np.array([item[1] for item in plot_data])

            # Cluster/window the data into bins to make plot more readble.
            bins = np.zeros(22)
            bins[:4] = np.linspace(0, 0.01, 4)
            bins[4:] = np.linspace(0.02, np.max(x[x <= 1.0]), 18)
            avg_std = np.zeros((len(bins), 4))

            for i in range(1, len(bins)):
                data = np.where(np.logical_and(x > bins[i - 1], x <= bins[i]))
                avg_std[i, :] = [
                    np.average(x[data]), np.average(y[data]),
                    np.std(y[data]), np.std(x[data])
                ]

            no_gates_count = np.sum(x == 0)
            label = 'threshold: {:2.2f}, {:3} images without detections'.format(threshold, no_gates_count)

            # Remove nan values.
            avg_std = avg_std[avg_std[:, 0] <= 1.0, :]

            # Plot errorbars only for the optimal threshold.
            if threshold == 0.15:
                plt.errorbar(avg_std[:, 0], avg_std[:, 1], avg_std[:, 2], avg_std[:, 3],
                    marker='o', markersize=6, capsize=3, barsabove=False, label=label, zorder=1, color='indigo')
            else:
                plt.plot(avg_std[:, 0], avg_std[:, 1], label=label, marker='o', zorder=0, markersize=4, ls='--')

            print('Number of images with no gates detected: {} for threshold: {}'.format(no_gates_count, threshold))

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid()
        plt.legend()
        plt.savefig('roc', bbox_inches='tight')


# Put the optimal threshold (0.15) at the end such that
# the optimal images are not overwritten by other results.
thresholds = [0.06, 0.1, 0.25, 0.15]

det = detection()
det.process(None, thresholds)
det.plot()