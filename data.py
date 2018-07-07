import os
import json
from imgaug import augmenters as iaa
import skimage.draw
import numpy as np

from Mask_RCNN.mrcnn import utils, visualize


class Dataset(utils.Dataset):
    NO_WALDO_64 = ["4_1_6.jpg", "9_0_5.jpg", "12_1_6.jpg", "16_5_2.jpg"]
    NO_WALDO_128 = ["4_2_12.jpg", "9_0_10.jpg", "9_0_11.jpg", "12_3_12.jpg", "13_5_12.jpg", "13_5_13.jpg",
                    "18_2_15.jpg", "18_4_12.jpg", "18_14_7.jpg", "19_0_6.jpg", "19_0_7.jpg"]
    NO_WALDO_256 = ["4_0_3.jpg", "9_0_2.jpg", "12_0_3.jpg", "13_0_3.jpg", "19_0_1.jpg"]
    VAL_IMAGE_64 = ["1_4_7.jpg", "3_15_1.jpg", "4_2_11.jpg", "9_5_5.jpg", "9_9_5.jpg", "12_2_1.jpg", "18_2_14.jpg",
                    "16_10_4.jpg"]
    VAL_IMAGE_128 = ["3_7_0.jpg", "7_6_2.jpg", "10_7_2.jpg", "13_1_5.jpg", "16_5_2.jpg", "18_7_3.jpg"]
    VAL_IMAGE_256 = ["2_1_0.jpg", "7_3_1.jpg", "9_0_3.jpg", "11_1_2.jpg", "13_1_0.jpg", "19_2_3.jpg"]

    # VAL_IMAGE = [
    #    "3.jpg", "7.jpg", "11.jpg", "13.jpg", "17.jpg"
    # ]
    # VAL_IMAGE = [
    #    "1.jpg", "17.jpg", "18.jpg", "19.jpg"
    # ]
    # VAL_IMAGE = ["4.jpg", "10.jpg"]
    VAL_IMAGE_ORG = ["3.jpg", "4.jpg", "18.jpg", "19.jpg"]
    ##VAL_EVAL = ["17.jpg", "18.jpg", "19.jpg"]
    # VAL_EVAL = ["6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg",
    #            "11.jpg", "12.jpg", "13.jpg", "14.jpg", "15.jpg", "16.jpg"]
    # VAL_EVAL = ["1.jpg", "2.jpg", "5.jpg", "8.jpg", "9.jpg", "12.jpg", "14.jpg", "15.jpg", "16.jpg"]
    EVAL_IMAGE_ORG = ["9.jpg", "14.jpg", "15.jpg"]

    def __init__(self, data_dir):
        self.root_dir = os.path.join(os.getcwd(), data_dir)
        super(Dataset, self).__init__()

    def load(self, subset):
        self.add_class("waldo", 1, "waldo")
        _64 = self._load64_images(subset)
        _128 = self._load128_images(subset)
        _256 = self._load256_images(subset)
        # _org = self._loadorg_images(subset)
        self._add_images(_64[0], _64[1], _64[2])
        self._add_images(_128[0], _128[1], _128[2])
        self._add_images(_256[0], _256[1], _256[2])
        # self._add_images(_org[0], _org[1], _org[2])

    def _load64_images(self, subset):
        dataset_dir = os.path.join(self.root_dir, "64", "waldo")
        if subset == "train":
            image_ids = next(os.walk(dataset_dir))[2]
            image_ids = list(set(image_ids) - set(self.VAL_IMAGE_64) - set(self.NO_WALDO_64))
        else:
            image_ids = self.VAL_IMAGE_64
        annotations = list(json.load(open(os.path.join(os.getcwd(), "annotations", "64waldo.json"))).values())
        return annotations, image_ids, dataset_dir

    def _load128_images(self, subset):
        dataset_dir = os.path.join(self.root_dir, "128", "waldo")
        if subset == "train":
            image_ids = next(os.walk(dataset_dir))[2]
            image_ids = list(set(image_ids) - set(self.VAL_IMAGE_128) - set(self.NO_WALDO_128))
        else:
            image_ids = self.VAL_IMAGE_128
        annotations = list(json.load(open(os.path.join(os.getcwd(), "annotations", "128waldo.json"))).values())
        return annotations, image_ids, dataset_dir

    def _load256_images(self, subset):
        dataset_dir = os.path.join(self.root_dir, "256", "waldo")
        if subset == "train":
            image_ids = next(os.walk(dataset_dir))[2]
            image_ids = list(set(image_ids) - set(self.VAL_IMAGE_256) - set(self.NO_WALDO_256))
        else:
            image_ids = self.VAL_IMAGE_256
        annotations = list(json.load(open(os.path.join(os.getcwd(), "annotations", "256waldo.json"))).values())
        return annotations, image_ids, dataset_dir

    def _loadorg_images(self, subset):
        dataset_dir = os.path.join(self.root_dir, "original-images")
        if subset == "train":
            image_ids = next(os.walk(dataset_dir))[2]
            image_ids = list(set(image_ids) - set(self.VAL_IMAGE_ORG) - set(self.EVAL_IMAGE_ORG))
        else:
            image_ids = self.VAL_IMAGE_ORG
        annotations = list(
            json.load(open(os.path.join(os.getcwd(), "annotations", "original_images.json"))).values())
        return annotations, image_ids, dataset_dir

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
