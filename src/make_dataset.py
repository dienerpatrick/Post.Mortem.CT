import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import matplotlib.pyplot as plt


TRAIN_LABELS = "../data/processed/MDCT_SPLIT/train_labels.csv"
VAL_LABELS = "../data/processed/MDCT_SPLIT/val_labels.csv"
MINIP_TRAIN = "../data/processed/MDCT_SPLIT/MINIP/train"
MINIP_VAL = "../data/processed/MDCT_SPLIT/MINIP/val"


class MDCTImageDataset(Dataset):
    """
    Creates a pytorch Dataset from CT images in a folder. The corresponding RA Index is
    read from a csv file, which is converted to a label, according to the class bins.

    :param annotations_file: path to csv file containing the image ID and RA Index
    :param img_dir: path to folder containing CT images with file name structure PREFIX****.png, **** being the four
                    digit ID of the image.
    :param class_names: [list] containing the class names
    :param class_bins: [list] of [tuples] containing the upper and lower RA Index limit for the corresponding class.
    :param transform: transforms.Compose() object containing the transforms to apply on image data
    :param target_transform: transforms.Compose() object containing the transforms to apply on labels
    """

    def __init__(self, annotations_file, img_dir, class_names, class_bins, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.class_names = class_names
        self.class_bins = class_bins

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        prefix = os.listdir(self.img_dir)[0][:-8]
        img_name = prefix + str(self.img_labels.iloc[idx, 1]) + ".png"
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        ra_index = int(self.img_labels.iloc[idx, 2])
        for i, c_bin in enumerate(self.class_bins):
            if c_bin[0] <= ra_index <= c_bin[1]:
                label = self.class_names[i]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def class_quantities(self):
        quantities = {i: 0 for i in self.class_names}
        quantities['TOTAL'] = 0
        for indx in range(len(os.listdir(self.img_dir))):
            label = self.__getitem__(indx)[1]
            quantities[label] += 1
            quantities['TOTAL'] += 1
        return quantities

    def class_summary(self):
        quantities = self.class_quantities()
        print(f"CLASS SUMMARY\n")
        for key, value in quantities.items():
            print(f"{key}:\t\t{value}\n")
        return

    def balanced_class_weights(self):
        quantities = self.class_quantities()
        weights = torch.FloatTensor([1/(quantities[i]/quantities['TOTAL']) for i in list(quantities.keys())[:-1]])
        return weights

    def print_sample_images(self, samples):

        for j, i in enumerate(samples):
            sample = self[i]

            # print(sample[0].shape, sample[1])
            # print(j)
            ax = plt.subplot(1, len(samples), j + 1)
            plt.tight_layout()
            ax.set_title(f'Sample {i} \nCLASS={sample[1]}')
            ax.axis('off')
            plt.imshow(np.transpose(sample[0], (1, 2, 0)))

        plt.show()

