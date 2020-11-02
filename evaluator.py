import copy
import datetime
import os
import time

import cv2
import detectron2.data.detection_utils as utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.transforms import ResizeTransform, NoOpTransform, ResizeShortestEdge, RandomFlip, \
    apply_augmentations
from detectron2.engine import DefaultPredictor
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA


# TODO: which one is wrong, wrong because of which class, wrong because of classification or IoU?
def find_ious(eval_boxes, output_boxes):
    """
    Both are np.array of boxes in form of x1, y1, x2, y2
    """
    eval_areas = (eval_boxes[:, 2] - eval_boxes[:, 0] + 1) * (eval_boxes[:, 3] - eval_boxes[:, 1] + 1)
    output_areas = (output_boxes[:, 2] - output_boxes[:, 0] + 1) * (output_boxes[:, 3] - output_boxes[:, 1] + 1)
    # The array of IoUs
    ious = np.zeros((len(eval_boxes), len(output_boxes)))
    # calculate the IOU
    for eval_idx in range(len(eval_boxes)):
        xx1 = np.maximum(eval_boxes[eval_idx, 0], output_boxes[:, 0])
        yy1 = np.maximum(eval_boxes[eval_idx, 1], output_boxes[:, 1])
        xx2 = np.minimum(eval_boxes[eval_idx, 2], output_boxes[:, 2])
        yy2 = np.minimum(eval_boxes[eval_idx, 3], output_boxes[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (eval_areas[eval_idx] + output_areas - inter)
        ious[eval_idx, :] = ovr
    return ious


def count_confusions(eval_boxes, output_boxes, iou_thresh=0.5):
    """
    Inputs must be in np.array xyxy format
    """
    ious = find_ious(eval_boxes, output_boxes)
    #     import pdb
    #     pdb.set_trace()
    result = {"true_positive": 0, "false_negative": 0, "false_positive": 0, "true_negative": 0}
    # ious conditions
    eval_trues = []
    output_trues = []
    while True:
        ret = np.where((ious > iou_thresh) & (ious == ious.max()))
        if len(ret[0]) > 0:
            # TODO: Only take the first one is not very suitable, take the one that is max and min with others (strict)
            eval_true_idx = ret[0][0]
            output_true_idx = ret[1][0]
            ious[eval_true_idx, :] = 0  # clean the rows
            ious[:, output_true_idx] = 0  # clean the columns
            eval_trues.append(eval_true_idx)
            output_trues.append(output_true_idx)
        else:
            break
    # True positives
    result["true_positive"] = len(eval_trues)
    # False positives => we predicted in but not in
    result["false_positive"] = sum([1 for i in range(len(output_boxes)) if i not in output_trues])
    # False negatives => in eval but not in true eval
    result["false_negative"] = sum([1 for i in range(len(eval_boxes)) if i not in eval_trues])
    return result


def evaluate_output(eval_dict, outputs, score_thresh_test, iou_thresh=0.5, top_n=5):
    """
    Classes here are still zero based
    """
    output_scores = outputs["instances"].scores.to('cpu').data.numpy()
    output_boxes = np.array([box.cpu().numpy() for box in outputs["instances"].pred_boxes])
    output_classes = outputs['instances'].pred_classes.to('cpu').data.numpy()
    # Filtering the outputs
    if len(output_boxes) > 0:
        # filter by scores #TODO: Different classes may have different thresh
        keep = np.where(output_scores >= score_thresh_test)[0]
        output_boxes = output_boxes[keep]
        output_classes = output_classes[keep]
        output_scores = output_scores[keep]
        # take only top_n
        keep = np.argsort(output_scores)[::-1][:top_n]
        output_boxes = output_boxes[keep]
        output_classes = output_classes[keep]
        ouput_scores = output_scores[keep]
        output_boxes = output_boxes.astype(np.int32)

    annotations = eval_dict['annotations']
    eval_boxes = np.array([anno['bbox'] for anno in annotations])
    eval_classes = np.array([anno['category_id'] for anno in annotations])
    result = [{'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'true_negative': 0} for _ in range(4)]

    for category_id in range(4):
        eval_keep_for_cls = np.where(eval_classes == category_id)[0]
        output_keep_for_cls = np.where(output_classes == category_id)[0]
        if len(eval_keep_for_cls) == 0:  # there are no evals but saying there are => false positive
            result[category_id]['false_positive'] = len(output_keep_for_cls)
        if len(output_keep_for_cls) == 0:  # there are evals but there are no predictions => false negative
            result[category_id]['false_negative'] = len(eval_keep_for_cls)
        if len(eval_keep_for_cls) > 0 and len(output_keep_for_cls) > 0:  # There are both we need to check
            eval_boxes_for_cls = eval_boxes[eval_keep_for_cls]
            output_boxes_for_cls = output_boxes[output_keep_for_cls]
            result[category_id] = count_confusions(eval_boxes_for_cls, output_boxes_for_cls)

    return result


def predict(test_dicts, cfg,
            score_thresh_test=0.3):  # threshold should be low to not filtering out too much, we will filter at the evaluation time
    start_time = time.time()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test
    predictor = DefaultPredictor(cfg)
    ret = []
    for d in test_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        ret.append(outputs)
    duration = time.time() - start_time
    print(f'Done inferences in {duration / 60} minutes at {datetime.datetime.now()}')
    return ret


def predict_batches(eval_dicts, cfg, batch_size=10):
    def get_test_batch(test_dicts, batch_size=batch_size):
        l = len(test_dicts)
        for ndx in range(0, l, batch_size):
            batch_data = test_dicts[ndx:min(ndx + batch_size, l)]
            ret = []
            for dataset_dict in batch_data:
                dataset_dict = copy.deepcopy(dataset_dict)
                image = utils.read_image(dataset_dict.pop("file_name"), format="BGR").copy()
                image = torch.from_numpy(image).permute(2, 0, 1)  # CHW
                dataset_dict["image"] = image
                ret.append(dataset_dict)
            # read data for it
            yield ret

    output_items = []
    num_batches = len(eval_dicts) // batch_size + 1
    batch_counter = 0
    for batch1 in get_test_batch(eval_dicts, batch_size=10):
        batch_counter += 1
        print(f'Batch {batch_counter}/{num_batches}')
        batch_output_items = predict_with_tta(batch1, cfg, score_thresh_test=0.3)
        output_items += batch_output_items
    return output_items


class DatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    def __init__(self, cfg):
        self.min_sizes = cfg.TEST.AUG.MIN_SIZES
        self.max_size = cfg.TEST.AUG.MAX_SIZE
        self.flip = cfg.TEST.AUG.FLIP
        self.image_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a dict in standard model input format. See tutorials for details.

        Returns:
            list[dict]:
                a list of dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
        shape = numpy_image.shape
        orig_shape = (dataset_dict["height"], dataset_dict["width"])
        if shape[:2] != orig_shape:
            # It transforms the "original" image in the dataset to the input image
            pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
        else:
            pre_tfm = NoOpTransform()

        # Create all combinations of augmentations to use
        aug_candidates = []  # each element is a list[Augmentation]
        for min_size in self.min_sizes:
            resize = ResizeShortestEdge(min_size, self.max_size)
            aug_candidates.append([resize])  # resize only
            if self.flip:
                flip = RandomFlip(prob=1.0)
                aug_candidates.append([resize, flip])  # resize + flip

        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = pre_tfm + tfms
            dic["image"] = torch_image
            ret.append(dic)
        return ret


def predict_with_tta(test_dicts, cfg, score_thresh_test=0.5, batch_size=3):
    with torch.no_grad():
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test
        predictor = DefaultPredictor(cfg)
        model = predictor.model
        tta_model = GeneralizedRCNNWithTTA(cfg, model, batch_size=batch_size)
        tta_model.image_format = tta_model.tta_mapper.image_format
        ret = tta_model(test_dicts)
    return ret


def evaluate_thresh_test(eval_dicts, output_items, score_thresh_test, iou_thresh=0.5):
    all_confusions = [evaluate_output(eval_dicts[i], output_items[i], score_thresh_test, iou_thresh=iou_thresh) for i in
                      range(len(eval_dicts))]
    true_positives = sum(
        [sum([confusion[cls_idx]['true_positive'] for cls_idx in range(4)]) for confusion in all_confusions])
    false_positives = sum(
        [sum([confusion[cls_idx]['false_positive'] for cls_idx in range(4)]) for confusion in all_confusions])
    false_negatives = sum(
        [sum([confusion[cls_idx]['false_negative'] for cls_idx in range(4)]) for confusion in all_confusions])
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return 2 * (precision * recall) / (precision + recall)


def evaluate_model(eval_dicts, output_items, score_thresh_tests):
    return [evaluate_thresh_test(eval_dicts, output_items, score_thresh_test=score_thresh_test, iou_thresh=0.5) for
            score_thresh_test in score_thresh_tests]


def evaluate_models(cfg, eval_dicts, the_model_names):
    model_bests = []
    threshold_bests = []
    f1_bests = []
    for the_model_name in the_model_names:
        print(f'Evaluating model {the_model_name}')
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, f"{the_model_name}.pth")
        output_items = predict(eval_dicts, cfg,
                               score_thresh_test=0.5)  # the score thresh test here should be small enough to not drop many boxes
        score_thresh_tests = np.arange(0.3, 1.0, 0.01)
        model_evals = evaluate_model(eval_dicts, output_items, score_thresh_tests)
        model_bests.append(max(model_evals))
        plt.plot(score_thresh_tests, model_evals, label=the_model_name)
        max_idx = np.argmax(model_evals)
        threshold_bests.append(score_thresh_tests[max_idx])
        f1_bests.append(model_evals[max_idx])
        print(f'{the_model_name} max f1 {model_evals[max_idx]} at {score_thresh_tests[max_idx]}')
    plt.legend()
    return model_bests, threshold_bests, f1_bests


def get_evaluation_configuration(OUTPUT_DIR, base_config_file, num_gpus=2, ims_per_batch=16):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = OUTPUT_DIR
    cfg.merge_from_file(model_zoo.get_config_file(base_config_file))

    cfg.DATASETS.TRAIN = ("road_damage_train",)
    cfg.DATASETS.TEST = ()
    # # for validation
    cfg.DATASETS.TEST = ("road_damage_eval",)

    cfg.DATALOADER.NUM_WORKERS = ims_per_batch
    cfg.SOLVER.REFERENCE_WORLD_SIZE = num_gpus

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # 12500 # 4096   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    return cfg
