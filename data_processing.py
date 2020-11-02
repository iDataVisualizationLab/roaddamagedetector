import os
import pickle
from xml.etree import ElementTree

import cv2
import numpy as np
import pandas as pd
from detectron2.structures import BoxMode
from sklearn.model_selection import StratifiedKFold


def get_data_dicts_for(country, train_test):
    base_path = os.path.abspath('./')
    dataset_dicts = []
    # note that this is zero based index, 4 is background (not 0 is background)
    damage_id_mappings = {'D00': 0, 'D10': 1, 'D20': 2, 'D40': 3}
    image_path = base_path + f'/{train_test}/' + country + '/images'
    annotation_path = base_path + f'/{train_test}/' + country + '/annotations/xmls'
    file_list = [filename.split('.')[0] for filename in os.listdir(image_path) if not filename.startswith('.')]

    for file_name in file_list:
        if file_name == '.DS_Store':
            pass
        else:
            # the image
            record = {}
            imagename = image_path + '/' + file_name + '.jpg'
            height, width = cv2.imread(imagename).shape[:2]

            record["file_name"] = imagename
            record["image_id"] = file_name
            record["height"] = height
            record["width"] = width
            record["country"] = country

            if train_test == "train":
                # the labels and bounding boxes
                infile_xml = open(annotation_path + '/' + file_name + '.xml')
                tree = ElementTree.parse(infile_xml)
                root = tree.getroot()

                objs = []
                for obj in root.iter('object'):
                    cls_name = obj.find('name').text
                    if cls_name in damage_id_mappings.keys():
                        # labels
                        label = damage_id_mappings[cls_name]  # not ethat this is zero based indexx
                        # bounding box
                        xmlbox = obj.find('bndbox')
                        xmin = int(xmlbox.find('xmin').text)
                        xmax = int(xmlbox.find('xmax').text)
                        ymin = int(xmlbox.find('ymin').text)
                        ymax = int(xmlbox.find('ymax').text)
                        obj = {
                            "bbox": [xmin, ymin, xmax, ymax],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            'category_id': label
                            # NOTE this is zero based index (and the_num_of_cls is for background, not 0 is background)
                        }
                        objs.append(obj)
                record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


def check_file_exists(file_name):
    return os.path.exists(file_name)


def check_pickle_exists(pickle_file):
    return check_file_exists('processed_pickles/' + pickle_file)


def save_obj(file_name, obj):
    with open('processed_pickles/' + file_name, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(file_name):
    with open('processed_pickles/' + file_name, 'rb') as f:
        return pickle.load(f)


def process_data():
    for country in ['Czech', 'India', 'Japan']:
        for train_test in ["train", "test1", "test2"]:
            dataset_name = f"{country}_{train_test}"
            data_dicts = get_data_dicts_for(country, train_test)
            save_obj(f"{dataset_name}.pkl", data_dicts)
            print(f"Saved {dataset_name}")


def load_data_dicts_for(country, train_test):
    file_name = f"{country}_{train_test}.pkl"
    return load_obj(file_name)


def load_test_data_dicts(test_set):
    return load_data_dicts_for('Czech', f'{test_set}') + load_data_dicts_for('India',
                                                                             f'{test_set}') + load_data_dicts_for(
        'Japan', f'{test_set}')


def load_train_data_dicts():
    return load_data_dicts_for('Czech', 'train') + load_data_dicts_for('India', 'train') + load_data_dicts_for('Japan',
                                                                                                               'train')


def generate_train_df(train_dicts):
    image_id = []
    width = []
    height = []
    x = []
    y = []
    w = []
    h = []
    category_id = []

    for item in train_dicts:
        for annotation in item['annotations']:
            image_id.append(item['image_id'])
            img_width = item['width']
            img_height = item['height']
            bbox = annotation['bbox']
            b_x = bbox[0]
            b_y = bbox[1]
            b_w = bbox[2] - bbox[0]
            b_h = bbox[3] - bbox[1]

            # check size
            b_x = 0 if b_x < 0 else img_width - 1 if b_x >= img_width else b_x
            b_y = 0 if b_y < 0 else img_height - 1 if b_y >= img_height else b_y
            b_w = 1 if b_w <= 0 else img_width - b_x if b_w > img_width - b_x else b_w
            b_h = 1 if b_h <= 0 else img_height - b_y if b_h > img_height - b_y else b_h

            width.append(img_width)
            height.append(img_height)
            x.append(b_x)
            y.append(b_y)
            w.append(b_w)
            h.append(b_h)
            category_id.append(annotation['category_id'])

    columns = ['image_id', 'width', 'height', 'x', 'y', 'w', 'h', 'category_id']
    df = pd.DataFrame(np.array([image_id, width, height, x, y, w, h, category_id]).T, columns=columns)
    int_cols = ['width', 'height', 'x', 'y', 'w', 'h', 'category_id']
    for int_col in int_cols:
        df[int_col] = pd.to_numeric(df[int_col])
    return df


def find_folds_df(train_df):
    # Split by the the image with number of the category_id that appears the least.
    # 1. Find the least frequent category
    def find_least_frequent_category_id(df):
        unique_ret = np.unique(df['category_id'].values, return_counts=True)
        least_idx = np.argsort(np.unique(df['category_id'].values, return_counts=True)[1])[0]
        return unique_ret[0][least_idx]

    least_frequent_category_id = find_least_frequent_category_id(train_df)
    # 2. the stratify
    df_folds = train_df[['image_id', 'category_id']].copy()
    df_folds = df_folds.groupby('image_id')['category_id'].apply(
        lambda x: (x == least_frequent_category_id).sum()).reset_index(name='stratify_group')
    # 3. split
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    df_folds.loc[:, 'fold'] = 0
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_idx].index, 'fold'] = fold_num
    # 4. return
    return df_folds


def process_train_eval_split_for(country):
    train_split_file = f'{country}_train_split.pkl'
    eval_split_file = f'{country}_eval_split.pkl'
    if (not check_pickle_exists(train_split_file)) or (not check_pickle_exists(eval_split_file)):
        train_dicts = get_data_dicts_for(country, 'train')
        train_df = generate_train_df(train_dicts)
        # find the df_folds
        df_folds = find_folds_df(train_df)
        fold_number = 0
        eval_dicts = [data_dict for data_dict in train_dicts if
                      data_dict['image_id'] in df_folds[df_folds['fold'] == fold_number]['image_id'].values]
        train_dicts = [data_dict for data_dict in train_dicts if
                       data_dict['image_id'] in df_folds[df_folds['fold'] != fold_number]['image_id'].values]

        save_obj(train_split_file, train_dicts)
        save_obj(eval_split_file, eval_dicts)
        print(f'Saved {len(train_dicts)} records for {country} train split')
        print(f'Saved {len(eval_dicts)} records for {country} evaluation split')
    else:
        print(f'{country} records for {country} train split exists')
        print(f'{country} records for {country} evaluation split exists')


def split_train_data():
    train_split_file = f'train_split.pkl'
    eval_split_file = f'eval_split.pkl'
    if (not check_pickle_exists(train_split_file)) or (not check_pickle_exists(eval_split_file)):
        countries = ["Czech", "India", "Japan"]
        # Split by the least occurrence damage type per country
        for country in countries:
            process_train_eval_split_for(country)

        # Load the processed data, join them
        train_dicts = []
        val_dicts = []
        for country in countries:
            train_dicts += load_obj(f'{country}_train_split.pkl')
            val_dicts += load_obj(f'{country}_eval_split.pkl')
        # Save the combined ones
        save_obj(train_split_file, train_dicts)
        save_obj(eval_split_file, val_dicts)
    else:
        print(f'Records for train split exists')
        print(f'Records for evaluation split exists')


def process_tests(test_names):
    for test_name in test_names:
        test_file = f'{test_name}.pkl'
        if not check_pickle_exists(test_file):
            test_dicts = []
            for country in ["Czech", "India", "Japan"]:
                test_dicts += get_data_dicts_for(country, test_name)
            save_obj(test_file, test_dicts)
            print(f'Saved {test_name} records')
        else:
            print(f'{test_name} exists')


def load_train_eval_splits():
    train_split_file = f'train_split.pkl'
    eval_split_file = f'eval_split.pkl'
    if (not check_pickle_exists(train_split_file)) or (not check_pickle_exists(eval_split_file)):
        split_train_data()
    # load
    return load_obj(train_split_file), load_obj(eval_split_file)


def load_tests():
    test1_split_file = f'test1.pkl'
    test2_split_file = f'test2.pkl'
    if (not check_pickle_exists(test1_split_file)) or (not check_pickle_exists(test2_split_file)):
        process_tests(["test1", "test2"])
    # load
    return load_obj(test1_split_file), load_obj(test2_split_file)


# This part is for detectron
# Converting data to COCO format
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def convert_dataset_to_coco_json(output_dir, registered_dataset_name):
    output_coco_json = os.path.join(output_dir, f'{registered_dataset_name}_coco_format.json')
    convert_to_coco_json(registered_dataset_name, output_file=output_coco_json, allow_cached=False)
    # Save old metadata
    metadata = DatasetCatalog.get(registered_dataset_name)
    # Remove the current one
    DatasetCatalog.remove(registered_dataset_name)
    # Register again
    # if your dataset is in COCO format, this cell can be replaced by the following three lines:
    register_coco_instances(registered_dataset_name, {}, output_coco_json, os.path.abspath('./'))
    return output_coco_json

def prepare_test_data(output_dir, test1_data, test2_data):
    # Register datasets
    DatasetCatalog.register("road_damage_test1", lambda:test1_data)
    MetadataCatalog.get("road_damage_test1").set(thing_classes=['D00', 'D10', 'D20', 'D40'])

    DatasetCatalog.register("road_damage_test2", lambda: test2_data)
    MetadataCatalog.get("road_damage_test2").set(thing_classes=['D00', 'D10', 'D20', 'D40'])

    # convert data format
    convert_dataset_to_coco_json(output_dir, 'road_damage_test1')
    convert_dataset_to_coco_json(output_dir, 'road_damage_test2')

    print('Converted and registered: road_damage_test1, road_damage_test2')
    return MetadataCatalog.get("road_damage_test1")

def prepare_data(output_dir, train_data, eval_data, test1_data, test2_data):
    # Register datasets
    DatasetCatalog.register("road_damage_train", lambda: train_data)
    MetadataCatalog.get("road_damage_train").set(thing_classes=['D00', 'D10', 'D20', 'D40'])

    DatasetCatalog.register("road_damage_eval", lambda: eval_data)
    MetadataCatalog.get("road_damage_eval").set(thing_classes=['D00', 'D10', 'D20', 'D40'])

    DatasetCatalog.register("road_damage_test1", lambda:test1_data)
    MetadataCatalog.get("road_damage_test1").set(thing_classes=['D00', 'D10', 'D20', 'D40'])

    DatasetCatalog.register("road_damage_test2", lambda: test2_data)
    MetadataCatalog.get("road_damage_test2").set(thing_classes=['D00', 'D10', 'D20', 'D40'])

    # convert data format
    convert_dataset_to_coco_json(output_dir, 'road_damage_train')
    convert_dataset_to_coco_json(output_dir, 'road_damage_eval')
    convert_dataset_to_coco_json(output_dir, 'road_damage_test1')
    convert_dataset_to_coco_json(output_dir, 'road_damage_test2')

    print('Converted and registered: road_damage_train,  road_damage_eval, road_damage_test1, road_damage_test2')
    return MetadataCatalog.get("road_damage_train")
