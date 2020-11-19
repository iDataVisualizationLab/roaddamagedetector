import copy
import random

import albumentations as A
import cv2
# import some common libraries
import numpy as np
import torch
from detectron2.data import DatasetCatalog
from detectron2.data import build_detection_train_loader, \
    detection_utils as utils, transforms as T
from detectron2.data.transforms.augmentation import apply_augmentations
from detectron2.engine import DefaultTrainer

from color_transfer import color_transfer


def get_all_locs(train_dicts):
    all_locs = []
    for item in train_dicts:
        all_locs = all_locs + [[anno['bbox'][0], anno['bbox'][1]] for anno in item['annotations'] if
                               len(item['annotations']) > 0]
    return all_locs


def sample_a_damage_of_type(dataset_dicts, damage_category_id):
    dataset_dicts = copy.deepcopy(dataset_dicts)
    while True:
        dataset_dict = random.sample(dataset_dicts, 1)[0]
        if "annotations" in dataset_dict and len(dataset_dict['annotations']) > 0:
            for obj in dataset_dict['annotations']:
                category_id = obj['category_id']
                if category_id == damage_category_id:
                    bbox = obj['bbox']
                    image = utils.read_image(dataset_dict["file_name"], format="BGR")
                    damage = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    gray = cv2.cvtColor(damage, cv2.COLOR_BGR2GRAY)
                    img_binary = (cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY)[1])
                    damage_masked = np.zeros_like(damage)
                    mask = img_binary == 0
                    damage_masked[mask] = damage[mask]
                    return {'damage': damage, 'annotation': obj, 'damage_masked': damage_masked}


def rotate_image(image, angle):
    """
    image: the image
    angle: in degrees
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# def rotate_image(mat, angle):
#     """
#     Rotates an image (angle in degrees) and expands image to avoid cropping
#     """

#     height, width = mat.shape[:2] # image shape has 3 dimensions
#     image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

#     rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

#     # rotation calculates the cos and sin, taking absolutes of those.
#     abs_cos = abs(rotation_mat[0,0])
#     abs_sin = abs(rotation_mat[0,1])

#     # find the new width and height bounds
#     bound_w = int(height * abs_sin + width * abs_cos)
#     bound_h = int(height * abs_cos + width * abs_sin)

#     # subtract old image center (bringing image back to origo) and adding the new image center coordinates
#     rotation_mat[0, 2] += bound_w/2 - image_center[0]
#     rotation_mat[1, 2] += bound_h/2 - image_center[1]

#     # rotate image with the new bounds and translated rotation matrix
#     rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
#     return rotated_mat


# Default augmentation
def build_augmentation(cfg):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    min_size = cfg.INPUT.MIN_SIZE_TRAIN
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style),
                    T.RandomFlip(prob=0.5, horizontal=True, vertical=False)]

    return augmentation


def check_conflict_boxes(the_box, conflict_boxes):
    if len(conflict_boxes) == 0:
        return False
    the_box = np.array(the_box)
    conflict_boxes = np.array(conflict_boxes)
    xx1 = np.maximum(the_box[0], conflict_boxes[:, 0])
    yy1 = np.maximum(the_box[1], conflict_boxes[:, 1])
    xx2 = np.minimum(the_box[2], conflict_boxes[:, 2])
    yy2 = np.minimum(the_box[3], conflict_boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    return any(i > 0 for i in inter)


def train_transforms():
    return A.Compose([
        A.RandomSizedCrop(min_max_height=(512, 540), height=600, width=600, p=0.1),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.3,
                                 val_shift_limit=0.3, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=0.5),
        ], p=0.9),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
        # A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
        # A.Cutout(num_holes=4, max_h_size=100, max_w_size=2, fill_value=0, p=0.5)
    ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels']
        )
    )


class MyDatasetMapper:
    def __init__(self, augs, all_locs, dataset_dicts_to_sample, for_vis=True, sample_probs={}):
        self.augmentations = augs
        self.all_locs = all_locs
        self.dataset_dicts_to_sample = dataset_dicts_to_sample
        self.for_vis = for_vis
        self.sample_probs = sample_probs

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        imgh, imgw = image.shape[:2]
        utils.check_image_size(dataset_dict, image)
        transform = train_transforms()
        country = dataset_dict['file_name'].split('/')[-1].split('_')[0]
        #Augment the image
        for category_id, sample_prob in enumerate(self.sample_probs[country]):
            if np.random.random() <= sample_prob:
                damage_obj = sample_a_damage_of_type(self.dataset_dicts_to_sample, category_id)
                if "annotations" not in dataset_dict:
                    dataset_dict["annotations"] = []
                # Duplicate the damage at the index
                image = copy.deepcopy(image)
                image.setflags(write=1)

                # damage = damage_obj['damage_masked']
                damage = damage_obj['damage']
                # the place to put
                posx, posy = random.sample(self.all_locs, 1)[0]
                dh, dw = damage.shape[:2]
                bboxes = np.array([obj['bbox'] for obj in dataset_dict['annotations']])
                counter = 0
                while len(bboxes) > 0 and check_conflict_boxes([posx, posy, posx + dw, posy + dh], bboxes):
                    posx, posy = random.sample(self.all_locs, 1)[0]
                    counter += 1
                    # only try for 1000 times maximum
                    if counter > 1000:
                        break
                # make sure that we don't place it out of the picture
                posy = min(posy, imgh - dh)
                posx = min(posx, imgw - dw)
                # make sure the damage is not out of bounds
                posy = 0 if posy < 0 else posy
                posx = 0 if posx < 0 else posx
                dh = imgh - posy if posy + dh > imgh else dh
                dw = imgw - posx if posx + dw > imgw else dw
                damage = damage[0:imgh, 0:imgw]

                # scale its color to its underlying range
                area_tobe_replaced = image[posy:posy + dh, posx:posx + dw]
                # Also transfer the color from the original picture to this
                damage = color_transfer(area_tobe_replaced, damage)

                # rotate it
                if category_id == 0 or category_id == 1:
                    damage = rotate_image(damage, random.randint(-5, 5))
                if category_id == 2 or category_id == 3:
                    damage = rotate_image(damage, random.randint(-30, 30))
                dh, dw = damage.shape[:2]

                # Build the mask to avoid the black due to rotation
                mask = np.full((imgh, imgw), False)  # default to not set all

                mask1 = damage.max(axis=2) > 0
                mask[posy:posy + dh, posx:posx + dw] = mask1
                image[mask] = damage[mask1]
                image.setflags(write=0)
                # change the box location of the annotation
                damage_obj['annotation']['bbox'] = [posx, posy, posx + dw, posy + dh]
                # Add the annotation to the set
                dataset_dict["annotations"].append(damage_obj['annotation'])

        # TODO: Augmentation comes here
        if "annotations" in dataset_dict and len(dataset_dict['annotations']) > 0:
            bboxes = np.array([obj['bbox'] for obj in dataset_dict['annotations']])
            # Make sure the bounding boxes are not out  of ranges
            bw = bboxes[:, 2] - bboxes[:, 0]
            bh = bboxes[:, 3] - bboxes[:, 1]
            bw[bw <= 0] = 1
            bh[bh <= 0] = 1

            bboxes[:, 0] = np.maximum(bboxes[:, 0], 0)
            bboxes[:, 0] = np.minimum(bboxes[:, 0], imgw - 1)
            bboxes[:, 1] = np.maximum(bboxes[:, 1], 0)
            bboxes[:, 1] = np.minimum(bboxes[:, 1], imgh - 1)
            bboxes[:, 2] = bboxes[:, 0] + bw
            bboxes[:, 3] = bboxes[:, 1] + bh

            class_labels = np.array([obj['category_id'] for obj in dataset_dict['annotations']])

            if transform:
                for i in range(10):
                    sample = {
                        'image': image,
                        'bboxes': bboxes,
                        'class_labels': class_labels
                    }
                    sample = transform(**sample)

                    if len(sample['bboxes']) > 0:
                        image = sample['image']
                        bboxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0).numpy()
                        class_labels = sample['class_labels']
                        break
                # Update the annotations
                annotations = []
                bbox_mode = dataset_dict.pop("annotations")[0]['bbox_mode']
                for i in range(len(bboxes)):
                    annotations.append({'bbox': bboxes[i], 'bbox_mode': bbox_mode, 'category_id': class_labels[i]})
                dataset_dict["annotations"] = annotations

        if "annotations" in dataset_dict and len(dataset_dict["annotations"]) > 0:
            bboxes = np.array([obj['bbox'] for obj in dataset_dict['annotations']])
            aug_input = T.StandardAugInput(image, boxes=bboxes)

            apply_augmentations(self.augmentations, aug_input)

            image = aug_input.image
            image_shape = image.shape[:2]  # height, width

            # USER: Implement additional transformations if you have other types of data
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            for i, obj in enumerate(dataset_dict["annotations"]):
                if obj.get("iscrowd", 0) == 0:
                    obj['bbox'] = aug_input.boxes[i]

            annos = [obj for obj in dataset_dict["annotations"]]  # keep for visualization purposes

            if not self.for_vis:
                dataset_dict.pop('annotations')  # remove annotations if we don't need it for visualization

            instances = utils.annotations_to_instances(
                annos, image_shape
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.

            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


class MyTrainerWithAugmentation(DefaultTrainer):
    sample_probs = {'Czech':[0.2, 0.2, 0.0, 0.4], 'India':[0.3, 0.5, 0.2, 0.0], 'Japan': [0.0, 0.6, 0.3, 0.5]}

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        pass

    @classmethod
    def build_train_loader(cls, cfg):
        augs = build_augmentation(cfg)
        train_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        all_locs = get_all_locs(train_dicts)
        mapper = MyDatasetMapper(augs, all_locs, train_dicts, sample_probs=MyTrainerWithAugmentation.sample_probs)
        return build_detection_train_loader(cfg, mapper=mapper)
