import os
from imgaug import augmenters as iaa
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
        augmentation = iaa.SomeOf((1, 4), [
            iaa.CropAndPad(percent=(0, 20)),
            iaa.Fliplr(p=0.2),
            iaa.Grayscale((0.1, 1.0)),
            iaa.CoarseDropout(size_percent=(0.02, 0.1)),
            iaa.Dropout(p=0.10),
            iaa.CropAndPad(5),
            iaa.Affine(rotate=(-45, 45)),
            iaa.Affine(scale=(0.5, 1.5)),
            iaa.Affine(translate_percent=(0.05, 0.15)),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0)),
        ])
        dataset_train = Dataset(self.config.DATA_DIR)
        dataset_train.load("train")
        dataset_train.prepare()

        dataset_val = Dataset(self.config.DATA_DIR)
        dataset_val.load("val")
        dataset_val.prepare()

        self.model.train(dataset_train, dataset_val,
                         learning_rate=self.config.LEARNING_RATE,
                         augmentation=augmentation,
                         epochs=30,
                         layers='heads')
