import os

from Mask_RCNN.mrcnn.model import MaskRCNN
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils


class TrainerConfig(Config):
    NAME = "waldo"
    IMAGES_PER_GPU = 4
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
    WEIGHT_PATH = os.path.join("models", "mask_rcnn_coco.h5")
    MODEL_PATH = os.path.join("models", "logs")
    DATA_DIR = "ImageSet"


class Dataset(utils.Dataset):
    def __init__(self, data_dir):
        self.root_dir = os.path.join(os.getcwd(), data_dir)
        super(Dataset, self).__init__()

    def load(self, path):
        self.add_class("waldo", 1, "waldo")
        if path == "waldo":
            image_path = os.path.join(self.root_dir, "128", path)
        else:
            image_path = os.path.join(self.root_dir, path)
        image_ids = next(os.walk(image_path))[2]

        for image_id in image_ids:
            self.add_image(
                "waldo",
                image_id=image_id,
                path=os.path.join(image_path, image_id))


class Trainer:
    def __init__(self, config=TrainerConfig()):
        self.config = config
        self.config.display()
        self.model = MaskRCNN(mode="training", config=config, model_dir=config.MODEL_PATH)
        self.weights = config.WEIGHT_PATH
        if not os.path.exists(self.weights):
            utils.download_trained_weights(self.weights)

        super(self.__class__, self).__init__()

    def train(self):
        dataset_train = Dataset(self.config.DATA_DIR)
        dataset_train.load("waldo")
        dataset_train.prepare()

        dataset_val = Dataset(self.config.DATA_DIR)
        dataset_val.load("original-images")
        dataset_val.prepare()

        self.model.train(dataset_train, dataset_val,
                         learning_rate=self.config.LEARNING_RATE,
                         epochs=10,
                         layers='heads')
