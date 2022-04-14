
from os import listdir
from os.path import isfile, join, splitext, exists

import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image, ExifTags, ImageOps

def get_mat_matrix(image_path, annotations_list, format = ".jpg"):

    images = [f for f in listdir(image_path) if (isfile(join(image_path, f)) and format in f.lower())]

    mats = []
    for i, image in enumerate(images):

        im = Image.open(image_path + image)
        width, height = im.size

        matrix = torch.zeros((width,height), dtype = torch.int64)
        try:
            matrix.index_put_(tuple(annotations_list[i][:,0:2].t()), annotations_list[i][:,2])
            mats.append(matrix)
        except:
            print("annotation outside image bounds: " + image)

    return mats
