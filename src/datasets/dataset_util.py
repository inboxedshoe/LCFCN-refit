
from os import listdir
from os.path import isfile, join, splitext, exists

import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image, ExifTags, ImageOps

def get_points_matrix(image_path, annotations_list, format = ".jpg", single_image = True):


    annotations_list = [torch.tensor(x) for x in annotations_list]

    mats = []
    if not single_image:
        images = [f for f in listdir(image_path) if (isfile(join(image_path, f)) and format in f.lower())]
        for i, image in enumerate(images):

            im = Image.open(image_path + image)
            width, height = im.size

            matrix = torch.zeros((width, height), dtype=torch.int64)
            try:
                matrix.index_put_(tuple(annotations_list[i][:, 0:2].t()), annotations_list[i][:, 2])
                mats.append(matrix)
            except:
                print("annotation outside image bounds: " + image)
    else:
        im = Image.open(image_path)
        width, height = im.size

        matrix = torch.zeros((width, height), dtype=torch.int64)
        try:
            matrix.index_put_(tuple(annotations_list[0][:, 0:2].t()), annotations_list[0][:, 2])
            mats.append(matrix)
        except:
            print("annotation outside image bounds: " + image_path)

    return mats
