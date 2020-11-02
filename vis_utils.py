import collections
import os
import random

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython import display
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

from processing_utils import output_str_to_img_id_boxes_classes


def plot_damage_distributions(dataset_dicts):
    category_ids = []
    for dataset_dict in dataset_dicts:
        category_ids += [anno['category_id'] for anno in dataset_dict['annotations']]

    count_dict = collections.Counter(category_ids)
    cls_count = []
    damage_types = ["D00", "D10", "D20", "D40"]
    for damage_type in range(len(damage_types)):
        print(str(damage_type) + ' : ' + str(count_dict[damage_type]))
        cls_count.append(count_dict[damage_type])

    sns.set_palette("winter", 4)
    sns.barplot(damage_types, cls_count)
    return cls_count


def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
      Args:
        a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
          (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
          image.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display.display(PIL.Image.fromarray(a))


def vis_predicted_images_without_eval(predictor, dataset_dicts, number_of_images=50, road_damage_metadata=None):
    for d in random.sample(dataset_dicts, number_of_images):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        if road_damage_metadata is None:
            road_damage_metadata = MetadataCatalog.get("road_damage_train")
        v = Visualizer(im, metadata=road_damage_metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
        print(outputs['instances'].pred_classes)
        print(outputs["instances"].pred_boxes)

        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2_imshow(v.get_image())


def vis_image(img, ax=None, figsize=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

    ax.imshow(img.astype(np.uint8))
    return ax


def vis_output_str(image_path, output_str, title=None):
    damage_label_mappings = {1: 'D00', 2: 'D10', 3: 'D20', 4: 'D40'}
    img_id, boxes, category_ids = output_str_to_img_id_boxes_classes(output_str)

    img = cv2.imread(os.path.join(image_path, img_id))

    # boxes, category_ids = group_overlapping_boxes(boxes, category_ids)
    labels = [damage_label_mappings[category_id] for category_id in category_ids]
    return vis_bbox(img, boxes, category_ids, labels, title=title)


def vis_output_str_from_file(file_name, output_str, title=None):
    damage_label_mappings = {1: 'D00', 2: 'D10', 3: 'D20', 4: 'D40'}
    img_id, boxes, category_ids = output_str_to_img_id_boxes_classes(output_str)

    img = cv2.imread(file_name)

    # boxes, category_ids = group_overlapping_boxes(boxes, category_ids)
    labels = [damage_label_mappings[category_id] for category_id in category_ids]
    return vis_bbox(img, boxes, category_ids, labels, title=title)


def vis_bbox(img, bbox, category_ids, label_names=None, score=None, ax=None, figsize=(10, 10), color=None, title=None):
    if color is None:
        color = 'red'

    if category_ids is not None and not len(bbox) == len(category_ids):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    if ax is None:
        ax = vis_image(img, ax=ax, figsize=figsize)

    # Set title
    if title is not None:
        ax.set_title(title)
    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[0], bb[1])
        width = bb[2] - bb[0]
        height = bb[3] - bb[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor=color, linewidth=2))

        caption = list()

        if label_names is not None:
            caption.append(label_names[i])

        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[0], bb[1],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})
    return ax


def vis_predicted_images_with_eval(predictor, dataset_dicts, number_of_images=50):
    damage_label_mappings = {1: 'D00', 2: 'D10', 3: 'D20', 4: 'D40'}

    for d in random.sample(dataset_dicts, number_of_images):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        gt_bboxes = None
        gt_label_ids = None
        gt_labels = None
        if len(d['annotations']) > 0:
            gt_bboxes = [a['bbox'] for a in d['annotations']]
            gt_label_ids = [a['category_id'] for a in d['annotations']]
            gt_labels = [damage_label_mappings[lid + 1] for lid in gt_label_ids]
        if gt_bboxes is None:
            gt_bboxes = []

        ax = vis_bbox(im, gt_bboxes, gt_label_ids, gt_labels, figsize=(10, 10), color='red')

        pred_bboxes = None
        pred_scores = None
        pred_category_ids = None
        pred_labels = None
        if len(list(outputs['instances'].pred_boxes)) > 0:
            pred_bboxes = outputs['instances'].pred_boxes.tensor.cpu().data.numpy()
            pred_scores = outputs['instances'].scores.cpu().data.numpy()
            pred_category_ids = outputs['instances'].pred_classes.cpu().data.numpy()
            pred_labels = [damage_label_mappings[lb + 1] for lb in pred_category_ids]
        if pred_bboxes is None:
            pred_bboxes = []

        pred_bboxes = np.array(pred_bboxes)
        pred_category_ids = np.array(pred_category_ids)
        pred_scores = np.array(pred_scores)

        #         pred_bboxes, pred_category_ids, pred_scores = group_overlapping_boxes(pred_bboxes, pred_category_ids, pred_scores)
        #         if pred_bboxes.size > 0:
        #             pred_labels = np.array([damage_label_mappings[lb+1] for lb in pred_category_ids])

        vis_bbox(im, pred_bboxes, pred_category_ids, pred_labels, pred_scores, ax=ax, figsize=(10, 10), color='blue',
                 title=d['image_id'])


def visualize_sample_outputs(dataset_dicts, output_lines, num_images=10):
    img_idxs = random.sample(range(len(output_lines)), num_images)
    for img_idx in img_idxs:
        output_str = output_lines[img_idx].replace('\n', '')
        image_path = dataset_dicts[img_idx]['file_name']
        title = dataset_dicts[img_idx]['image_id']
        vis_output_str_from_file(image_path, output_str, title=title)
