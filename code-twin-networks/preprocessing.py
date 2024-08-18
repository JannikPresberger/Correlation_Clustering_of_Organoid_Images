import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_histograms(
        input_directory: Path,
        output_directory: Path
):
    for label in os.listdir(input_directory):

        label_subdir = os.path.join(input_directory, label)

        out_subdir = os.path.join(output_directory, label)

        if not os.path.exists(out_subdir):
            os.makedirs(out_subdir)

        for file in os.listdir(label_subdir):
            print(f"Processing file {file}")
            file_path = os.path.join(label_subdir, file)

            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            bgr_planes = cv2.split(image)
            histSize = [2 ** 16]
            histRange = [0, 2 ** 16]

            histograms = np.empty(shape=(3, 2**16))

            for i in range(3):
                histogram = cv2.calcHist(bgr_planes, [i], None, histSize, histRange, accumulate=False)
                cv2.normalize(histogram, histogram, 1, 0, cv2.NORM_L1)
                histograms[i, :] = histogram[:, 0]
                histograms[i, :] = np.cumsum(histograms[i, :])


            out_filename = os.path.splitext(file)[0] + ".npy"
            out_path = os.path.join(out_subdir, out_filename)


            with open(out_path, "wb") as f:
                np.save(f, histograms)

            #
            # cv2.normalize(image, image, 0, 255, norm_type=cv2.NORM_MINMAX)
            #
            # figure, axes = plt.subplots(1, 2)
            # axes[0].imshow(image)
            # axes[1].plot(histograms[0], color='blue')
            # axes[1].plot(histograms[1], color='green')
            # axes[1].plot(histograms[2], color='red')
            # plt.show()



if __name__ == "__main__":
    preprocess_histograms(
        input_directory="/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids New/unlabeled",
        output_directory="/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids New/histograms/unlabeled"
    )