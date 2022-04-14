from torch.utils import data
import numpy as np
import torch
import os
from skimage.io import imread
from scipy.io import loadmat
import torchvision.transforms.functional as FT
from haven import haven_utils as hu
import transformers
import dataset_util
import pandas as pd

def get_txt_annotation(annotation):
    df = pd.read_csv(annotation, delimiter="\t")
    return [df[["PosX", "PosY", "VarietyID"]].values.tolist()]

class KOB(data.Dataset):
    def __init__(self, split, datadir, K = 1, exp_dict = None):
        self.split = split
        self.exp_dict = exp_dict

        self.n_classes = K

        if split == "train":
            fname = os.path.join(datadir, 'image_sets', 'training.txt')

        elif split == "val":
            fname = os.path.join(datadir, 'image_sets', 'validation.txt')

        elif split == "test":
            fname = os.path.join(datadir, 'image_sets', 'test.txt')

        #self.img_names = [name.replace(".jpg\n", "") for name in hu.read_text(fname)]
        self.img_names = [name for name in hu.read_text(fname)]
        self.img_names = [name.replace("\n", "") for name in self.img_names]
        #self.img_names = [name.replace(".JPG\n", "") for name in self.img_names]
        self.path = os.path.join(datadir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]

        # LOAD IMG, POINT, and ROI
        #image = imread(os.path.join(self.path, name + ".jpg"))
        #points = imread(os.path.join(self.path, name + "dots.png"))[:, :, :1].clip(0, 1)
        #roi = loadmat(os.path.join(self.path, name + "mask.mat"))["BW"][:, :, np.newaxis]

        annotation = name.replace(".jpg", "_SingleHerbsPos.txt")
        annotation = annotation.replace(".JPG", "_SingleHerbsPos.txt")

        annotations = get_txt_annotation(self.path + annotation)

        image = imread(os.path.join(self.path, name))
        points = dataset_util.get_points_matrix(self.path + name, annotations)[0].cpu().detach().numpy()

        # LOAD IMG AND POINT
        #image = image * roi
        image = hu.shrink2roi(image, points)
        points = hu.shrink2roi(points, points).astype("uint8")

        counts = np.arange(self.n_classes)
        counts = torch.LongTensor(np.array([int(points.sum())]))

        collection = list(map(FT.to_pil_image, [image, points]))
        # image, points = transformers.apply_transform(self.split, image, points,
        #                                              transform_name=self.exp_dict['dataset']['transform'])

        return {"images": image,
                "points": points.squeeze(),
                "counts": counts,
                'meta': {"index": index}}

data = KOB("train", "/home/jukebox/Documents/Data/InsectMon-old/2020 YST KOB/")
data.__getitem__(1)