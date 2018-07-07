import os
from Mask_RCNN.mrcnn.config import Config


class TrainerConfig(Config):
    NAME = "waldo"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.5
    BACKBONE = "resnet50"

    WEIGHT_PATH = os.path.join("models", "mask_rcnn_coco.h5")
    # WEIGHT_PATH = os.path.join("models", "mask_rcnn_waldo_0028.h5")
    MODEL_PATH = os.path.join("models", "logs")

    # Input image resizing
    # Generally, use the "square" resizing mode for training and inferencing
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "pad64"
    IMAGE_MIN_DIM = None
    # IMAGE_MIN_DIM = 512
    # IMAGE_MAX_DIM = 512
    # IMAGE_MIN_SCALE = 2.0
    # RESULTS_DIR = os.path.join("results", "waldo")
    DATA_DIR = "ImageSet"
    DETECTION_MAX_INSTANCES = 2


class FinderConfig(TrainerConfig):
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
