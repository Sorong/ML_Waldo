import os
import json

from Mask_RCNN.mrcnn.model import MaskRCNN
from Mask_RCNN.mrcnn import utils, visualize
import skimage.draw
import numpy as np
from imgaug import augmenters as iaa

from config import TrainerConfig


class Dataset(utils.Dataset):
    VAL_IMAGE_IDS_128 = [
        "1_2_3.jpg",
        "2_3_1.jpg",
        "4_1_5.jpg",
        "5_1_2.jpg",
        "6_7_0.jpg",
        "9_0_5.jpg",
        "10_7_2.jpg",
        "19_0_3.jpg"
    ]
    # VAL_IMAGE = [
    #    "3.jpg", "7.jpg", "11.jpg", "13.jpg", "17.jpg"
    # ]
    # VAL_IMAGE = [
    #    "1.jpg", "17.jpg", "18.jpg", "19.jpg"
    # ]
    #VAL_IMAGE = ["4.jpg", "10.jpg"]
    VAL_IMAGE = ["3.jpg", "4.jpg", "18.jpg", "19.jpg"]
    ##VAL_EVAL = ["17.jpg", "18.jpg", "19.jpg"]
    # VAL_EVAL = ["6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg",
    #            "11.jpg", "12.jpg", "13.jpg", "14.jpg", "15.jpg", "16.jpg"]
    #VAL_EVAL = ["1.jpg", "2.jpg", "5.jpg", "8.jpg", "9.jpg", "12.jpg", "14.jpg", "15.jpg", "16.jpg"]
    VAL_EVAL = ["9.jpg", "14.jpg", "15.jpg"]

    def __init__(self, data_dir):
        self.root_dir = os.path.join(os.getcwd(), data_dir)
        super(Dataset, self).__init__()

    def load(self, subset):
        self.add_class("waldo", 1, "waldo")
        dataset_dir = os.path.join(self.root_dir, "128", "waldo")
        dataset_org = os.path.join(self.root_dir, "original-images")
        if subset == "train":
            image_ids = next(os.walk(dataset_dir))[2]
            image_ids = list(set(image_ids) - set(self.VAL_IMAGE_IDS_128))
            image_ids_org = next(os.walk(dataset_org))[2]
            image_ids_org = list(set(image_ids_org) - set(self.VAL_IMAGE) - set(self.VAL_EVAL))
        else:
            image_ids = self.VAL_IMAGE_IDS_128
            image_ids_org = self.VAL_IMAGE
        annotations = list(json.load(open(os.path.join(os.getcwd(), "annotations", "128waldo.json"))).values())
        annotations_org = list(
            json.load(open(os.path.join(os.getcwd(), "annotations", "original_images.json"))).values())
        # self._add_images(annotations, image_ids, dataset_dir)
        self._add_images(annotations_org, image_ids_org, dataset_org)

    def _add_images(self, annotations, image_ids, dataset_dir):
        augmentation = iaa.SomeOf((1, 4), [
            iaa.CropAndPad(percent=(0, 20)),
            iaa.Fliplr(p=0.2),
            iaa.Grayscale((0.1, 1.0)),
            iaa.CoarseDropout(size_percent=(0.02, 0.1)),
            iaa.Dropout(p=0.10),
            iaa.CropAndPad(5),
            iaa.Affine(rotate=(-45, 45)),
            iaa.Scale((0.5, 1.5)),
            iaa.Affine(translate_percent=(0.05, 0.15)),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0)),
        ])

        for item in annotations:
            if not item['filename'] in image_ids:
                continue
            polygons = [region['shape_attributes'] for region in item['regions']]

            image_path = os.path.join(dataset_dir, item['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                "waldo",
                image_id=item['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "waldo":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        visualize.display_top_masks(self.load_image(image_id), mask, np.ones([mask.shape[-1]], dtype=np.int32),
                                    self.class_names, limit=1)
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "waldo":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


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
