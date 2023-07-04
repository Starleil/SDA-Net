'''
REMIND: if use mask, the data transforms need to be done on both ori data and mask !
if self.joint_transform is not None:
            out_list = self.joint_transform([img]+mask)
            img = out_list[0]
            mask = out_list[1:]
'''
import os
import glob
import pandas as pd
import numpy as np
import random
import torch
from torchvision import transforms
from PIL import Image
from utils.get_pixel import getpixel_pil

class NoduleData(torch.utils.data.Dataset):
    def __init__(self, root='cla/', mode='train', joint_transform=None, transform=None, use_cam=False):

        self.root = root
        self.mode = mode

        if self.mode == 'test':
            self.csv_path = os.path.join(self.root, self.mode)
        else:
            self.csv_path = os.path.join(self.root, 'train', self.mode)
        print(self.csv_path)
        self.raw_samples = []
        self.raw_labels = []
        self.joint_transform = joint_transform
        self.transform = transform
        self.use_cam = use_cam

        diagnosis_df_paths = os.listdir(self.csv_path)
        diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith('.csv')]
        for diagnosis_df_path in diagnosis_df_paths:
            diagnosis_df = pd.read_csv(os.path.join(self.csv_path, diagnosis_df_path), sep=',', header=None, names=['scan_dir', 'label'])
            diagnoses_list = list(diagnosis_df.scan_dir)
            label_list = list(diagnosis_df.label)
            self.raw_samples.extend(diagnoses_list)
            self.raw_labels.extend(label_list)
        assert len(self.raw_samples) == len(self.raw_labels)

        self.samples = []
        for idx, raw_sample in enumerate(self.raw_samples):
            self.samples.append((raw_sample, self.raw_labels[idx]))

        print("Image count in {} path :{}".format(self.mode, len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_id, label = self.samples[index]
        image_path, mask_path = os.path.join('./dataset', self.root.split('/')[2], 'image', image_id), \
                                os.path.join('./dataset', self.root.split('/')[2], 'mask', image_id)

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.joint_transform is not None:
            out_list = self.joint_transform([image, mask])
            image = out_list[0]
            mask = out_list[1]

        if self.transform is not None:
            image = self.transform(image)

        if torch.isnan(image).any():
            print("Dir %s has NaN values." % image_path)
            image[torch.isnan(image)] = 0

        # transform [0,255] to [0,1]
        mask = mask.point(lambda p: 1 if p > 125 else p)
        assert getpixel_pil(mask) == [0,1] or [1,0]

        # equal to transform.ToTensor
        mask = np.array(mask, dtype=np.float32)
        mask = torch.from_numpy(mask).long()

        if self.use_cam:
            return (image, mask, label)
        else:
            return (image, label)