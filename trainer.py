import os

from Mask_RCNN.mrcnn import utils
from Mask_RCNN.mrcnn.model import MaskRCNN
from config import TrainerConfig
from data import Dataset


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
        dataset_train.load("train")
        dataset_train.prepare()

        dataset_val = Dataset(self.config.DATA_DIR)
        dataset_val.load("val")
        dataset_val.prepare()

        self.model.train(dataset_train, dataset_val,
                         learning_rate=self.config.LEARNING_RATE,
                         epochs=30,
                         layers='heads')
