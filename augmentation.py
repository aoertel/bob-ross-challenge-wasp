import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import os
import imageio

ia.seed(42)

for filename in os.listdir("curated"):
    print("Current file: " + filename)
    img = imageio.imread("curated/" + filename)
    images = np.array(
        [img for _ in range(32)], dtype=np.uint8)

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.AdditiveGaussianNoise(loc=0, scale=(
                0.0, 0.05*255), per_channel=0.5),
            iaa.LinearContrast((0.75, 1.5)),
            # iaa.Rotate((-5, 5))
            iaa.GammaContrast((0.75, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 1.0))
        ],
        random_order=True
    )

    images_aug = seq.augment_images(images)

    for i in range(8):
        imageio.imwrite(
            "augmented/" + filename[:-4] + "_" + str(i) + ".png", images_aug[i])
