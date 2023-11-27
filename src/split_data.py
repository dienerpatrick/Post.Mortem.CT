import os
import numpy as np
import pandas as pd
import shutil
import random


def train_val_split(root_folder, root_subfolders, dest_folder, val_split, random_seed=None):
    """
    Takes images from subfolders of root_folder, randomly selects same val_split fraction of images for all subfolders,
    moves train and validation split to dest_folder in new folders train and val.
    :param root_folder: path to root folder
    :param root_subfolders: subfolders in root folder
    :param dest_folder: destination folder
    :param val_split: [float] between 0 and 1; validation split
    :return: None
    """
    # create train and validation folders in destination
    for subfolder in root_subfolders:
        os.mkdir(os.path.join(dest_folder, subfolder))
        os.mkdir(os.path.join(dest_folder, subfolder, "train"))
        os.mkdir(os.path.join(dest_folder, subfolder, "val"))
    # get file indexes and number
    file_indexes = [name[-8:-4] for name in os.listdir(os.path.join(root_folder, root_subfolders[0])) if name[-4:] == ".png"]
    num_files = len(file_indexes)
    # compute number of files in validation split
    num_val = int(num_files*val_split)
    # sample num_val files from filenames
    if random_seed:
        random.seed(random_seed)
    val_indexes = random.sample(file_indexes, num_val)
    # get train indexes
    train_indexes = list(set(file_indexes) - set(val_indexes))
    # copy train/validation files to train/validation folders
    for subfolder in root_subfolders:
        prefix = os.listdir(os.path.join(root_folder, subfolder))[0][:-8]
        for val_index in val_indexes:
            filename = prefix + val_index + ".png"
            shutil.copyfile(os.path.join(root_folder, subfolder, filename),
                            os.path.join(dest_folder, subfolder, "val", filename))
        for train_index in train_indexes:
            filename = prefix + train_index + ".png"
            shutil.copyfile(os.path.join(root_folder, subfolder, filename),
                            os.path.join(dest_folder, subfolder, "train", filename))
    return


def label_split(root_labels, dest_labels, train_folder, val_folder):
    """
        Takes csv file containing image-ID/label pairs, and splits them, based on the folders containing the images
        split in train and val, into two csv files, containing the labels for train and val splits respectively.
        :param root_labels: path to root csv file
        :param dest_labels: destination folder
        :param train_folder: path to folder containing train images
        :param val_folder: path to folder containing val images
        :return: None
        """
    train_idxs = [int(name[-8:-4]) for name in os.listdir(os.path.join(train_folder)) if name[-4:] == ".png"]
    val_idxs = [int(name[-8:-4]) for name in os.listdir(os.path.join(val_folder)) if name[-4:] == ".png"]
    root_csv = pd.read_csv(root_labels)
    train_csv = root_csv.loc[root_csv['ID'].isin(train_idxs)]
    val_csv = root_csv.loc[root_csv['ID'].isin(val_idxs)]
    train_csv.to_csv(os.path.join(dest_labels, "train_labels.csv"))
    val_csv.to_csv(os.path.join(dest_labels, "val_labels.csv"))
    return

