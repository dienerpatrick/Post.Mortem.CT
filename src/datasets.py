import torchvision.transforms as T
import torch
from make_dataset import MDCTImageDataset
import os


TRAIN_LABELS = "../data/processed/MDCT_SPLIT/train_labels.csv"
VAL_LABELS = "../data/processed/MDCT_SPLIT/val_labels.csv"
MINIP_TRAIN = "../data/processed/MDCT_SPLIT/MINIP/train"
MINIP_VAL = "../data/processed/MDCT_SPLIT/MINIP/val"

########################### MINIP AP Greyscale ###########################

tsfrm = T.Compose([T.ToPILImage(),
                   T.Grayscale(1),
                   T.Resize((512, 512)),
                   T.ToTensor()])

minipAPGS = MDCTImageDataset(annotations_file=TRAIN_LABELS,
                             img_dir=MINIP_TRAIN,
                             class_names=[0, 1],
                             class_bins=[(0, 6), (7, 100)],
                             transform=tsfrm)


########################### MINIP AP ResNet ###########################
tsfrm = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

minipAPResNet = MDCTImageDataset(annotations_file=TRAIN_LABELS,
                                 img_dir=MINIP_TRAIN,
                                 class_names=[0, 1, 2],
                                 class_bins=[(0, 1), (2, 25), (26, 100)],
                                 transform=tsfrm)

########################### MINIP AP ResNet AUG ###########################

tsfrm = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.RandomHorizontalFlip(1.0),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

flipped = MDCTImageDataset(annotations_file=TRAIN_LABELS,
                           img_dir=MINIP_TRAIN,
                           class_names=[0, 1],
                           class_bins=[(0, 6), (7, 100)],
                           transform=tsfrm)

minipAPResNetAUG = torch.utils.data.ConcatDataset([flipped, minipAPResNet])


########################### MINIP AP InceptionV3 ###########################

tsfrm = T.Compose([
    T.ToPILImage(),
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

minipAPInceptionV3 = MDCTImageDataset(annotations_file=TRAIN_LABELS,
                             img_dir=MINIP_TRAIN,
                             class_names=[0, 1, 2],
                             class_bins=[(0, 1), (2, 25), (26, 100)],
                             transform=tsfrm)

########################### MINIP AP AlexNet ###########################
tsfrm = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(227),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

minipAPAlexNet = MDCTImageDataset(annotations_file=TRAIN_LABELS,
                                  img_dir=MINIP_TRAIN,
                                  class_names=[0, 1],
                                  class_bins=[(0, 25), (26, 100)],
                                  transform=tsfrm)

########################### MINIP AP AlexNet 25 ###########################

tsfrm = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(227),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

minipAPAlexNet25 = MDCTImageDataset(annotations_file=TRAIN_LABELS,
                                  img_dir=MINIP_TRAIN,
                                  class_names=[0, 1, 2, 3],
                                  class_bins=[(0, 25), (26, 50), (51, 75), (76, 100)],
                                  transform=tsfrm)

########################### MINIP AP AlexNet AUG ###########################

tsfrm = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(227),
    T.RandomHorizontalFlip(1.0),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

flipped = MDCTImageDataset(annotations_file=TRAIN_LABELS,
                           img_dir=MINIP_TRAIN,
                           class_names=[0, 1],
                           class_bins=[(0, 25), (26, 100)],
                           transform=tsfrm)

minipAPAlexNetAUG = torch.utils.data.ConcatDataset([flipped, minipAPAlexNet])
