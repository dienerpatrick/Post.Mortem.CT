# PMCT RAI Prediction Algorithm

This algorithm predicts the radiological alteration index (RAI) in post mortem computed tomography images.

## Description and Status of the Project

The goal of this project is to implement an algorithm which, using machine learning tools, predicts the radiological alteration index (RAI) of a post mortem computed tomography (PMCT) image.

## System Requirements

The implemented code has been tested on the following operation systems:

MacOS Monterey 12.2.1

Required Packages:

matplotlib 3.5.1  
numpy 1.22.1  
scipy 1.7.3  
scikit-learn 1.0.2  
pytorch 1.10.1  

**note:** *maybe additional missing packages are required!*

I highly recommend to install all the requirements in a conda environment! If this is not familiar to you, refer to 	[here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).


## Preprocessing

**Input files**

The input images must be in **.png** format and have a width of 512 pixels, the height may vary. The last four digits of the filename must contain the file ID. A seperate csv file contains the IDs in the first column and the corresponding RA-Index in the second column. The files should be split and stored as depicted in the section **Creating a Dataset**.

## Creating a Dataset

### split_data.py

The file **split_data.py** contains two functions **train_val_split()** and **label_split()**.

### Recipe to split raw data

The initial state the data must be in is the following:
All the image groups (for example different projections or views) must be in seperate folders in the **/raw** folder.
The folder structure might look like this:

```

├── data
│   ├── raw
│   │   ├── labels.csv
│   │   ├── AVG
│   │   │   ├── average_1823.png
│   │   │   ├── ...
│   │   ├── COMBINED
│   │   │   ├── combined_1823.png
│   │   │   ├── ...
│   │   ├── MINIP
│   │   │   ├── minip_1823.png
│   │   │   ├── ...
│   │   ├── MIP
│   │   │   ├── mip_1823.png
│   │   │   ├── ...
│   ├── split
├── src
│   ├── datasets.py
│   ├── make_dataset.py
│   ├── split_data.py
│   ├── transforms.py
│   ├── models.py
│   ├── train_model.py
├── log
└── main.py
```

To split the data folders into train and val split, the functions in **split_data.py** are used.
The function **train_val_split()** splits the images of a given group of folders into **train** and **val** groups and saves them into seperate folders in a destination directory.

```
train_val_split(root_folder, root_subfolders, dest_folder, val_split, random_seed=None):

Takes images from subfolders of root_folder, randomly selects same val_split fraction of images for all subfolders,
moves train and validation split to dest_folder in new folders train and val.

root_folder:        path to root folder
root_subfolders:    subfolders in root folder
dest_folder:        destination folder
val_split:          [float] between 0 and 1, validation split
return:             None

```

The function call to split the above example folders would look like the following:

```
train_val_split(root_folder = "./data/raw",
                root_subfolders = ["AVG", "COMBINED", "MINIP", "MIP"],
                dest_folder = "./data/split"
                val_split = 0.2)
```

To split the **labels.csv** file according to the image split, the function **label_split()** is used. It takes the lines of the .csv file and splits them into two seperate file **train_labels.csv** and **val_labels.csv** according to the images in a given split folder. This can be any of the folders containing the split images, since 

```
label_split(root_labels, dest_labels, train_folder, val_folder):

Takes csv file containing image-ID/label pairs, and splits them, based on the folders containing the images
split in train and val, into two csv files, containing the labels for train and val splits respectively.

root_labels:        path to root csv file
dest_labels:        destination folder
train_folder:       path to folder containing train images
val_folder:         path to folder containing val images
return:             None
```

To split the **labels.csv** file into two seperate files, the following function call is made:

```
label_split(root_labels = "./data/raw/labels.csv", 
            dest_labels = "./data/split", 
            train_folder = "./data/split/AVG/train", 
            val_folder = "./data/split/AVG/val")
```

After the two function calls, the example folder structure would look as follows:

```

├── data
│   ├── raw
│   │   ├── labels.csv
│   │   ├── AVG
│   │   │   ├── average_1823.png
│   │   │   ├── ...
│   │   ├── COMBINED
│   │   │   ├── combined_1823.png
│   │   │   ├── ...
│   │   ├── MINIP
│   │   │   ├── minip_1823.png
│   │   │   ├── ...
│   │   ├── MIP
│   │   │   ├── mip_1823.png
│   │   │   ├── ...
│   ├── split
│   │   ├── train_labels.csv
│   │   ├── val_labels.csv
│   │   ├── AVG
│   │   │   ├── train
│   │   │   │   ├── average_1823.png
│   │   │   │   ├── ...
│   │   │   ├── val
│   │   │   │   ├── average_1931.png
│   │   │   │   ├── ...
│   │   ├── COMBINED
│   │   │   ├── train
│   │   │   │   ├── combined_1823.png
│   │   │   │   ├── ...
│   │   │   ├── val
│   │   │   │   ├── combined_1931.png
│   │   │   │   ├── ...
│   │   ├── MINIP
│   │   │   ├── train
│   │   │   │   ├── minip_1823.png
│   │   │   │   ├── ...
│   │   │   ├── val
│   │   │   │   ├── minip_1931.png
│   │   │   │   ├── ...
│   │   ├── MIP
│   │   │   ├── train
│   │   │   │   ├── mip_1823.png
│   │   │   │   ├── ...
│   │   │   ├── val
│   │   │   │   ├── mip_1931.png
│   │   │   │   ├── ...
├── src
│   ├── datasets.py
│   ├── make_dataset.py
│   ├── split_data.py
│   ├── transforms.py
│   ├── models.py
│   ├── train_model.py
├── log
└── main.py
```


### make_dataset.py

The file **make_dataset.py** contains the definition of the class **MDCTImageDataset**, which creates a pytorch dataset from CT images in a folder. The corresponding RA-Index is read from a csv file, which is converted to a label, according to the class bins.

```
MDCTImageDataset(Dataset)

The MDCTImageDataset Object creates a pytorch Dataset from CT images in a folder. The corresponding RA Index is
read from a csv file, which is converted to a label, according to the class bins.

annotations_file:   path to csv file containing the image ID and RA Index
img_dir:            path to folder containing CT images with file name structure PREFIX****.png, **** being the four digit ID 
                    of the image.
class_names:        [list] containing the class names
class_bins:         [list] of [tuples] containing the upper and lower RA Index limit for the corresponding class.
transform:          transforms.Compose() object containing the transforms to apply on image data
target_transform:   transforms.Compose() object containing the transforms to apply on labels
```

### datasets.py

The file **datasets.py** provides space to define different datasets of the type **MDCTImageDataset**, ready to be imported into the training algorithm. Different networks require different dataset specifications, which can be predefined in this file and later be imported into **train_model.py**. 


### transforms.py

The file **transforms** provides space to implement different transformation classes, which can be imported and utilized in **make_dataset.py**

## Defining a Model

### models.py

The file **models.py** provides space to define different models to then import in **train_model.py** and train on a corresponding dataset defined in **dataset.py**.

## Training a Model

### main.py

The file **main.py** contains the main training function **train_model()** as well as various helper functions.

```
train_model()

Takes a neural network model and its hyperparameters to train it on a dataset.

model:              [class] Neural network model which should be trained
dataset:            [torch.utils.data.Dataset] Pytorch Dataset
k_folds:            [int] Number of folds to perform in k-fold cross validation
learning_rate:      [float] The learning rate with which the optimizer should step
weight_decay:       [float] Weight decay (L2 Penalty) for optimizer
batch_size:         [int] Number of samples in one batch
epochs:             [int] Number of epochs to perform in each fold.
loss_function:      [function] Pytorch loss function.
optimizer:          [function] Pytorch optimizer.
class_weights:      [torch.Tensor, str] If given, the loss function weights the classes according to the
                    class_weights, if set to "auto_balance" the class_weights are automatically set to balance the classes.
pretrained:         [bool] Set to True if a pretrained Network is used, to avoid resetting pretrained parameters.
save_model:         [bool] Set to True if model should be saved to ./log/training_log.
metrics:            [list] Metrics to calculate and save.
manual_seed:        [int] If given, manual torch seed is set.
log_notes:          [string] String containing notes to add to log file.
```

To train a model (defined in **models.py** and imported to **main.py**) on a dataset (defined in **datasets.py** and imported to **main.py**), the function train_model() simply needs to ne called with the desired parameters.

A function call of **train_model.py** could look like the following example:

```
import src.datasets as ds
import src.models as m

train_model(model=m.AlexNetBN1,
            dataset=ds.minipAPAlexNet25,
            k_folds=5,
            learning_rate=1e-4,
            weight_decay=0,
            batch_size=10,
            epochs=30,
            loss_function=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            class_weights="auto_balance",
            pretrained=False,
            save_model=False,
            metrics=['confusion_matrix', 'loss_plot', 'accuracy_plot'],
            log_notes="AlexNetBN1 model with dataset minipAPAlexNet25")
```

### Log Files

For every call of **train_model()**, a folder of the name **[year_month_day__hour_minute_second]_model_log** is created in the **root/log** folder. This folder contains a **model_log.txt** file containing a description of the model, dataset and hyperparameters used, as well as performance scores. Additionally for every metric chosen under the attribute **metrics** in **train_model()**, a folder is created containing the results for every fold seperately.



