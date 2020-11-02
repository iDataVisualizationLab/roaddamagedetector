import copy
import datetime
import time

import cv2
import numpy as np
from detectron2.engine import DefaultPredictor


def nms(boxes, scores, classes, min_size=16, nms_thresh=0.7, n_post_nms=5):
    # Remove predicted boxes with either height or width < threshold
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = np.where((hs >= min_size) & (ws >= min_size))[0]
    boxes = boxes[keep, :]
    # Also filter the class scores
    scores = scores[keep]
    classes = classes[keep]

    # Split the locations into x1, y1, x2, y2
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # This is the index (argsort) for the new array (different index already, after filtered)
    order = scores.argsort()[::-1]  # this is descending order start:stop:step

    # Only keep those who have overlap less than a threshold with the top N and we do from top down
    keep = []
    while order.size > 0:
        i = order[0]  # take the 1st elt in order and append to keep
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (area[i] + area[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        # +1 here because the order array in the processing is order[1:], so it was cut off 1 compared to original order
        order = order[inds + 1]
    keep = keep[:n_post_nms]  # while training/testing , use accordingly
    boxes = boxes[keep]  # the final region proposals
    scores = scores[keep]
    classes = classes[keep]
    return boxes, scores, classes


# Post processing
# same type and stay
def group_overlapping_boxes(boxes, classes):
    keep_boxes = []
    keep_classes = []
    if len(boxes) == 0:
        return boxes, classes

    boxes = np.array(boxes)
    classes = np.array(classes)

    for cls in np.unique(classes):
        boxes_for_cls = boxes[classes == cls]
        if len(boxes_for_cls) == 1:
            keep_boxes.append(list(boxes_for_cls[0]))
            keep_classes.append(cls)
        else:
            x1 = boxes_for_cls[:, 0]
            y1 = boxes_for_cls[:, 1]
            x2 = boxes_for_cls[:, 2]
            y2 = boxes_for_cls[:, 3]
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = np.arange(len(boxes_for_cls))
            while order.size > 0:
                i = np.array(order[0])
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (area[i] + area[order[1:]] - inter)

                # to combine
                to_combine = np.concatenate([[i], order[np.where(ovr > 0)[0] + 1]])
                to_check = order[np.where(ovr <= 0)[0] + 1]

                if to_combine.size > 1:  # >1 since include the i itself now
                    # Now combine
                    xc1 = x1[to_combine].min()
                    yc1 = y1[to_combine].min()
                    xc2 = x2[to_combine].max()
                    yc2 = y2[to_combine].max()
                    # update this current one
                    x1[i] = xc1
                    y1[i] = yc1
                    x2[i] = xc2
                    y2[i] = yc2
                    # add it to the list of to check for overlapping with others
                    to_check = np.insert(to_check, 0, i)

                else:  # no more overlapping then keep this top box
                    keep_boxes.append([x1[i], y1[i], x2[i], y2[i]])
                    keep_classes.append(cls)

                order = to_check  # go for another round

    return keep_boxes, keep_classes


def output_str_to_img_id_boxes_classes(output_str):
    img_id, pred_str = output_str.split(',')
    boxes = []
    category_ids = []
    if pred_str != '':
        nums = [int(float(part)) for part in pred_str.split(' ')]
        box_count = len(nums) // 5
        for i in range(box_count):
            category_ids.append(nums[i * 5 + 0])
            boxes.append([nums[i * 5 + j] for j in range(1, 5)])

    return img_id, boxes, category_ids


# # test data
# def group_overlapping_boxes(boxes, classes, scores, min_overlap_thresh=0):
#     keep_boxes = []
#     keep_scores = []
#     keep_classes = []
#     if len(boxes) == 0:
#         return np.array(keep_boxes), np.array(keep_classes), np.array(keep_scores)
#
#     for cls in np.unique(classes):
#         # Only keep those who have overlap less than a threshold with the top N and we do from top down
#         boxes_for_cls = boxes[classes == cls]
#         scores_for_cls = scores[classes == cls]
#         if len(boxes_for_cls) == 1:
#             keep_boxes.append(list(boxes_for_cls[0]))
#             keep_scores.append(scores_for_cls[0])
#             keep_classes.append(cls)
#         else:
#             x1 = boxes_for_cls[:, 0]
#             y1 = boxes_for_cls[:, 1]
#             x2 = boxes_for_cls[:, 2]
#             y2 = boxes_for_cls[:, 3]
#             area = (x2 - x1 + 1) * (y2 - y1 + 1)
#
#             order = scores_for_cls.argsort()[::-1]
#
#             while order.size > 0:
#                 i = np.array(order[0])  # The first one in the order
#
#                 # find  those who has overlap with this
#                 xx1 = np.maximum(x1[i], x1[order[1:]])
#                 yy1 = np.maximum(y1[i], y1[order[1:]])
#                 xx2 = np.minimum(x2[i], x2[order[1:]])
#                 yy2 = np.minimum(y2[i], y2[order[1:]])
#                 w = np.maximum(0.0, xx2 - xx1 + 1)
#                 h = np.maximum(0.0, yy2 - yy1 + 1)
#                 inter = w * h
#                 ovr = inter / (area[i] + area[order[1:]] - inter)
#
#                 # to combine
#                 to_combine = np.concatenate([[i], order[np.where(ovr > 0)[0] + 1]])
#                 to_check = order[np.where(ovr <= min_overlap_thresh)[0] + 1]
#
#                 if to_combine.size > 1:  # >1 since include the i itself now
#                     # Now combine
#                     xc1 = x1[to_combine].min()
#                     yc1 = y1[to_combine].min()
#                     xc2 = x2[to_combine].max()
#                     yc2 = y2[to_combine].max()
#                     # update this current one
#                     x1[i] = xc1
#                     y1[i] = yc1
#                     x2[i] = xc2
#                     y2[i] = yc2
#                     # add it to the list of to check for overlapping with others
#                     to_check = np.insert(to_check, 0, i)
#
#                 else:  # no more overlapping then keep this top box
#                     keep_boxes.append([x1[i], y1[i], x2[i], y2[i]])
#                     keep_scores.append(scores_for_cls[i])
#                     keep_classes.append(cls)
#
#                 order = to_check  # go for another round
#     # reordering them by scores
#     order = np.argsort(keep_scores)[::-1]
#     keep_boxes = np.array(keep_boxes)[order]
#     keep_classes = np.array(keep_classes)[order]
#     keep_scores = np.array(keep_scores)[order]
#
#     return keep_boxes, keep_classes, keep_scores


def get_output_str(d, outputs, top_n=5):
    image_id = d['image_id']
    scores = outputs["instances"].scores.to('cpu').data.numpy()
    boxes = np.array([box.cpu().numpy() for box in outputs["instances"].pred_boxes])
    classes = outputs['instances'].pred_classes.to('cpu').data.numpy()

    #     # group them
    #     pred_bboxes = None
    #     pred_scores = None
    #     pred_category_ids = None
    #     pred_labels = None
    #     if len(list(outputs['instances'].pred_boxes)) > 0:
    #         pred_bboxes = outputs['instances'].pred_boxes.tensor.cpu().data.numpy()
    #         pred_scores = outputs['instances'].scores.cpu().data.numpy()
    #         pred_category_ids = outputs['instances'].pred_classes.cpu().data.numpy()
    #         pred_labels = [damage_label_mappings[lb+1] for lb in pred_category_ids]
    #     if pred_bboxes is None:
    #         pred_bboxes = []

    #     boxes = np.array(pred_bboxes)
    #     classes = np.array(pred_category_ids)
    #     scores = np.array(pred_scores)

    #     boxes, classes, scores = group_overlapping_boxes(boxes, classes, scores)

    pred_str = ''
    if len(boxes) > 0:
        # do the NMS
        # boxes, scores, classes = nms(boxes, scores, classes, min_size=16, nms_thresh=0.7, n_post_nms=5)

        # Take only top n
        keep = np.argsort(scores)[::-1][:top_n]
        boxes = boxes[keep]
        classes = classes[keep]

        boxes = boxes.astype(np.int32)
        classes = classes + 1

        classes = classes[:, np.newaxis]
        combined = np.hstack([classes, boxes])
        pred_str = ' '.join([' '.join(p.astype(str)) for p in combined])
    ret_str = f'{image_id}.jpg,{pred_str}\n'
    return ret_str


def write_output(file_name, output_lines):
    with open('submissions/' + file_name, 'w') as f:
        f.writelines(output_lines)


def process_submission(model_name, test_dicts, cfg, score_thresh_tests, top_n=5, base_lines=[]):
    # NOTE ALSO TO CHOOSE TOP 5 OR 3
    for score_thresh_test in score_thresh_tests:
        base_lines_copy = copy.deepcopy(base_lines)
        start_time = time.time()
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test
        predictor = DefaultPredictor(cfg)

        output_lines = base_lines_copy
        for d in test_dicts:
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            output_lines.append(get_output_str(d, outputs, top_n))

        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #     cfg.SOLVER.MAX_ITER
        file_name = f"{model_name}_{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}_{time_str}.txt"
        write_output(file_name, output_lines)
        duration = time.time() - start_time
        print(f'Written: {file_name} in {duration / 60} minutes at {datetime.datetime.now()}')


def submissions_for_outputs(model_name, test_dicts, output_items, score_thresh_test, top_n=5, base_lines=[]):
    test_dicts = copy.deepcopy(test_dicts)
    output_items = copy.deepcopy(output_items)
    base_lines_copy = copy.deepcopy(base_lines)

    output_lines = base_lines_copy
    for item_idx, outputs in enumerate(output_items):
        image_id = test_dicts[item_idx]['image_id']
        output_scores = outputs["instances"].scores.to('cpu').data.numpy()
        output_boxes = np.array([box.cpu().numpy() for box in outputs["instances"].pred_boxes])
        output_classes = outputs['instances'].pred_classes.to('cpu').data.numpy()
        # Filtering the outputs
        pred_str = ''
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
            # type conversions
            output_boxes = output_boxes.astype(np.int32)
            # add 1 for outputs
            output_classes = output_classes + 1
            output_classes = output_classes[:, np.newaxis]

            combined = np.hstack([output_classes, output_boxes])
            pred_str = ' '.join([' '.join(p.astype(str)) for p in combined])
        ret_str = f'{image_id}.jpg,{pred_str}\n'
        output_lines.append(ret_str)
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{model_name}_{score_thresh_test}_{time_str}.txt"
    write_output(file_name, output_lines)
    print(f'Written: {file_name} at {datetime.datetime.now()}')
    return output_lines

