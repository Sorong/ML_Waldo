import sys

import skimage

from Mask_RCNN.mrcnn.model import MaskRCNN
from config import FinderConfig
import numpy as np
from PIL import Image


class WaldoFinder:
    def __init__(self, weights, configuration=FinderConfig()):
        self.config = configuration
        self.weights_path = weights
        self.config.display()

    def find(self, imgpath, outputname):
        model = MaskRCNN(mode="inference", config=self.config,
                         model_dir=self.config.MODEL_PATH)
        print("weights_path: ", self.weights_path)
        model.load_weights(self.weights_path, by_name=True)

        image = skimage.io.imread(imgpath)
        masks = model.detect([image], verbose=1)[0]["masks"]

        print("Masks:", masks)

        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        mask_filter = (np.sum(masks, -1, keepdims=True) >= 1)

        if mask_filter.shape[0] > 0:
            waldo = np.where(mask_filter, image, gray).astype(np.uint8)
            img = Image.fromarray(waldo, 'RGB')
            img.save(outputname + ".jpg")
        else:
            print("Can't find Waldo. Hmm..")
