# importing libraries
import json
import numpy as np
import pandas as pd
import glob
import re
import cv2
import warnings
import pickle

warnings.filterwarnings("ignore")

# Reading annotated  (TODO: For every folder)
data_9_43 = pd.read_csv("/Users/ashishsaini/Downloads/9_43/new_data.csv")

# considering rows which has vissible == 1
data_9_43_visible = data_9_43[data_9_43["Visible"] != 0]
data_9_43_visible["ObjectName"] = data_9_43_visible["ObjectName"].apply(
    lambda x: x.split("_")[0].lower()
)


# for those images which are in folder 10_00
bop_csv = pd.read_csv("/Users/ashishsaini/Downloads/10_00 (2)/new_data.csv")
bop_csv_visible = bop_csv[bop_csv["Visible"] != 0]


# accessing names of all images
files = sorted(glob.glob("/Users/ashishsaini/Downloads/9_43/images/*.jpg"))


# function for height and width of image
def get_height_width(path):
    img = cv2.imread(path)
    height = img.shape[0]
    width = img.shape[1]
    return height, width


# making dict for images
images = []
for i in range(len(files)):
    images_dict = dict.fromkeys(
        ["id", "dataset_id", "path", "height", "width", "file_name"],
    )
    images_dict["id"] = int(
        "943" + files[i].split("/")[-1].split(".")[0]
    )  # image_id with dataset_id
    images_dict["dataset_id"] = "9_43"
    images_dict["path"] = files[i]
    height, width = get_height_width(files[i])
    images_dict["height"] = height
    images_dict["width"] = width
    images_dict["file_name"] = files[0].split("/")[-1]

    images.append(images_dict)


# images[0]


# dict for category mapping
category = {
    "small_load_carrier": 3,
    "forklift": 5,
    "pallet": 7,
    "stillage": 10,
    "pallet_truck": 11,
}

# area of the box
def area(box):
    height = box[-1]
    width = box[-2]
    return width * height


# segmentation of box (by using x_min, x_max, y_min, y_max)
def segmentation(box):
    x_min = box[1]
    y_min = box[2]
    height = box[-1]
    width = box[-2]
    x_max = width + x_min
    y_max = height + y_min

    return [
        x_min,
        y_min,
        x_min,
        y_min + y_max,
        x_min + x_max,
        y_min + y_max,
        x_min + x_max,
        y_max,
    ]


# making dict for annotations
annotations = []
for i in range(data_9_43_visible.shape[0]):

    file_name = data_9_43_visible.iloc[i]["fileName"]  # for image id
    object_name = data_9_43_visible.iloc[i]["ObjectName"]  # category id
    int_list = data_9_43_visible.iloc[i]["BoundingBox"]  # bounding box
    box = int_list.strip("][").split(", ")  # converting '[]' representation to []
    box = [int(i) for i in box]  # converting values to int

    annotations_dict = dict.fromkeys(
        [
            "id",
            "image_id",
            "category_id",
            "segmentation",
            "area",
            "bbox",
            "iscrowd",
            "isbbox",
            "color",
            "metadata",
        ],
    )
    annotations_dict["id"] = i
    annotations_dict["image_id"] = int(file_name.split("/")[-1].split(".")[0])
    annotations_dict["category_id"] = category[object_name]
    annotations_dict["segmentation"] = [segmentation(box)]
    annotations_dict["area"] = area(box)
    annotations_dict["bbox"] = box
    annotations_dict["iscrowd"] = False
    annotations_dict["isbbox"] = True
    annotations_dict["color"] = None
    annotations_dict["metadata"] = {}

    annotations.append(annotations_dict)


# len(images), len(annotations)
# images[0]

dump = json.dumps(images)  # converting list of dict to jason

# saving both lists images and annotations using pickle
with open("images_9_43.pkl", "wb") as f:
    pickle.dump(images, f)


with open("annotations_9_43.pkl", "wb") as f:
    pickle.dump(annotations, f)


# pickle.load('annotations_9_43.pkl')

# saving updated jason files
with open("images_9_34.jason", "w") as f:
    json.dump(images, f)
