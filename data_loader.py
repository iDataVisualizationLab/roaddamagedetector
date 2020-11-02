import os
import pickle
from xml.etree import ElementTree
import cv2
from detectron2.structures import BoxMode


def get_data_dicts_for(country, train_test):
    base_path = os.path.abspath('draft/')
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


def save_obj(file_name, obj):
    with open('processed_pickles/'+file_name, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(file_name):
    with open('processed_pickles/'+file_name, 'rb') as f:
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

