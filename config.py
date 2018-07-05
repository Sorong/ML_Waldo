import os
from Mask_RCNN.mrcnn.config import Config


class TrainerConfig(Config):
    NAME = "waldo"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

    WEIGHT_PATH = os.path.join("models", "mask_rcnn_coco.h5")
    MODEL_PATH = os.path.join("models", "logs")
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0
    # RESULTS_DIR = os.path.join("results", "waldo")
    DATA_DIR = "ImageSet"


class FinderConfig(TrainerConfig):
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
