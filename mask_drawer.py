import os
import json
import skimage.draw
import numpy as np
from data import Dataset
from matplotlib import pyplot as plt


def image_data(image_id, annotations):
    for a in annotations:
        if a['filename'] == image_id:
            return a
    return None


if __name__ == '__main__':
    data = Dataset("ImageSet")
    masks_path = os.path.join(os.getcwd(), "masks")
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)
    images = []
    annotations, image_ids, dataset_dir = data._loadorg_images("all")
    for img in image_ids:
        polygon = image_data(img, annotations)
        if not polygon:
            continue
        i = skimage.io.imread(os.path.join(dataset_dir, img))
        i.fill(0)
        for p in polygon['regions']:
            y = p["shape_attributes"]["all_points_y"]
            x = p["shape_attributes"]["all_points_x"]
            rr, cc = skimage.draw.polygon(y, x)
            i[rr, cc] = (255, 255, 255)
        images.append(i)
        skimage.io.imsave(os.path.join(masks_path, str(len(images)) + ".jpg"), i)

    skimage.io.imshow(images[0])
    plt.show()
