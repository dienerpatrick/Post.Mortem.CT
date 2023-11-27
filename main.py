import torch
from torch import nn
import os
from sklearn.model_selection import KFold
import src.datasets as ds
import src.models as m
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PROJECT_PATH = os.path.dirname(__file__)
TRAIN_LABELS = os.path.join(PROJECT_PATH, "data/processed/MDCT_SPLIT/train_labels.csv")
VAL_LABELS = os.path.join(PROJECT_PATH, "data/processed/MDCT_SPLIT/val_labels.csv")
MINIP_TRAIN = os.path.join(PROJECT_PATH, "data/processed/MDCT_SPLIT/MINIP/train")
MINIP_VAL = os.path.join(PROJECT_PATH, "data/processed/MDCT_SPLIT/MINIP/val")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def reset_weights(model):
  '''
    Resets model weights.
  '''

  for layer in model.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


def train_model(model, dataset, k_folds, learning_rate, weight_decay,  batch_size, epochs, loss_function, optimizer,
                class_weights=None, pretrained=False, save_model=False, metrics=None, manual_seed = None,
                log_notes=""):
    """
    Takes a neural network model and its hyperparameters to train it on a dataset.

    :param model: [class] Neural network model which should be trained
    :param dataset: [torch.utils.data.Dataset] Pytorch Dataset
    :param k_folds: [int] Number of folds to perform in k-fold cross validation
    :param learning_rate: [float] The learning rate with which the optimizer should step
    :param weight_decay: [float] Weight decay (L2 Penalty) for optimizer
    :param batch_size: [int] Number of samples in one batch
    :param epochs: [int] Number of epochs to perform in each fold.
    :param loss_function: [function] Pytorch loss function.
    :param optimizer: [function] Pytorch optimizer.
    :param class_weights: [torch.Tensor, str] If given, the loss function weights the classes according to the
    class_weights, if set to "auto_balance" the class_weights are automatically set to balance the classes.
    :param pretrained: [bool] Set to True if a pretrained Network is used, to avoid resetting pretrained parameters.
    :param save_model: [bool] Set to True if model should be saved to ./log/training_log.
    :param metrics: [list] Metrics to calculate and save.
    :param manual_seed: [int] If given, manual torch seed is set.
    :param log_notes: [string] String containing notes to add to log file.
    :return: None
    """

    # For fold results
    results = {}

    # Set fixed random number seed
    if manual_seed:
        torch.manual_seed(manual_seed)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Init the neural network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = model
    model = model.to(device=device)

    # create log folder
    now = datetime.now()
    LOG_DIR = os.path.join(PROJECT_PATH, "log/training_log", now.strftime("%Y_%m_%d__%H_%M_%S") + "_model_log")
    os.mkdir(LOG_DIR)

    # write to log file
    log_file = open(os.path.join(LOG_DIR, "model_log.txt") ,"w")
    log_file.write("k_folds = " + str(k_folds) + "\n" +
                   "learning_rate = " + str(learning_rate) + "\n" +
                   "batch_size = " + str(batch_size) + "\n" +
                   "epochs = " + str(epochs) + "\n" +
                   "loss_function = " + str(loss_function) + "\n" +
                   "optimizer = " + str(optimizer) + "\n" +
                   "class_weights = " + str(class_weights) + "\n" +
                   "pretrained = " + str(pretrained) + "\n" +
                   "manual_seed = " + str(manual_seed) + "\n" +
                   "model = " + str(model) + "\n\n" +
                   log_notes)
    log_file.close()

    # Add weights to loss function
    if class_weights is not None:
        if class_weights == "auto_balance":
            weights = dataset.balanced_class_weights()
            loss_function.weight = weights
            print(f"Class weights set to auto balance: {weights}")
        else:
            loss_function.weight = class_weights

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold+1}/{k_folds}')
        print('--------------------------------')

        # Losses for Plot
        train_losses = []
        test_losses = []

        # Accuracies for Plot
        train_accuracies = []
        test_accuracies = []

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, sampler=test_subsampler)

        # Reset weights
        if not pretrained:
            network.apply(reset_weights)
        else:
            for layer in model.fc.children():
                try:
                    torch.nn.init.xavier_uniform_(layer.weight)
                except AttributeError:
                    pass

        # Initialize optimizer
        optim = optimizer(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Run the training loop for defined number of epochs
        for epoch in range(0, epochs):

            # Print epoch
            print(f'Starting epoch {epoch + 1}/{epochs}')

            # Set current loss value
            current_loss = 0.0

            # epoch loss
            epoch_train_loss = 0.0

            # lists for F1 scores
            epoch_train_pred = []
            epoch_train_targets = []

            # Iterate over the DataLoaders batches for training data
            for i, data in enumerate(trainloader, 0):

                # Get inputs
                inputs, targets = data

                # Zero the gradients
                optim.zero_grad()

                # Perform forward pass
                outputs = network(inputs)

                # save output and targets
                epoch_train_pred.extend(outputs)
                epoch_train_targets.extend(targets)

                # Compute loss
                loss = loss_function(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optim.step()

                # Print statistics
                current_loss += loss.item()
                epoch_train_loss += loss.item()/(len(trainloader)*batch_size)

                if i % 10 == 9:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 100))
                    current_loss = 0.0

            train_losses.append(epoch_train_loss)

            # calculate F1 scores
            epoch_train_targets = [int(i) for i in epoch_train_targets]
            epoch_train_pred = [int(np.argmax(i.detach().numpy())) for i in epoch_train_pred]
            epoch_train_accuracy = f1_score(epoch_train_targets, epoch_train_pred, average='micro')
            epoch_train_macro_f1 = f1_score(epoch_train_targets, epoch_train_pred, average='macro')
            epoch_train_weighted_f1 = f1_score(epoch_train_targets, epoch_train_pred, average='weighted')

            train_accuracies.append(epoch_train_accuracy)
            # print epoch Train metrics
            print("\nEpoch Train Metrics\n")
            print(f"Accuracy: {int(epoch_train_accuracy*100)}%")
            print(f"Macro F1 Score: {int(epoch_train_macro_f1*100)}%")
            print(f"Weighted F1 Score: {int(epoch_train_weighted_f1*100)}%\n")

            # Evaluation for this epoch
            epoch_test_loss = 0.0
            epoch_test_pred = []
            epoch_test_targets = []

            with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):
                    # Get inputs
                    inputs, targets = data

                    # Generate outputs
                    outputs = network(inputs)

                    # save output and targets
                    epoch_test_pred.extend(outputs)
                    epoch_test_targets.extend(targets)

                    # calculate loss
                    epoch_test_loss += loss_function(outputs, targets)/(len(testloader)*batch_size)

            # add loss to test_losses list
            test_losses.append(epoch_test_loss)

            # calculate test F1 score
            epoch_test_targets = [int(i) for i in epoch_test_targets]
            epoch_test_pred = [int(np.argmax(i)) for i in epoch_test_pred]
            epoch_test_accuracy = f1_score(epoch_test_targets, epoch_test_pred, average='micro')
            epoch_test_macro_f1 = f1_score(epoch_test_targets, epoch_test_pred, average='macro')
            epoch_test_weighted_f1 = f1_score(epoch_test_targets, epoch_test_pred, average='weighted')

            # print epoch test metrics
            print("Epoch Test Metrics\n")
            print(f"Accuracy: {int(epoch_test_accuracy*100)}%")
            print(f"Macro F1 Score: {int(epoch_test_macro_f1*100)}%")
            print(f"Weighted F1 Score: {int(epoch_test_weighted_f1*100)}%\n")

            print("====================================\n")




        # Process is complete.
        print('Training process has finished.')
        print('Starting testing')

        # Saving the model
        if save_model:
            torch.save(network.state_dict(), os.path.join(LOG_DIR, f'model-fold-{fold}.pth'))

        # Evaluation for this fold
        y_pred = []
        y_true = []
        correct, total = 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):

                # Get inputs
                inputs, targets = data
                y_true.extend(targets)
                # Generate outputs
                outputs = network(inputs)
                y_pred.extend((torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy())
                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold + 1, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)

        # Plot and save metrics
        if metrics:
            # confusion matrix
            # https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
            if 'confusion_matrix' in metrics:
                # Build confusion matrix
                cf_matrix = confusion_matrix(y_true, y_pred)
                # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10)
                df_cm = pd.DataFrame(cf_matrix)
                plt.figure(figsize=(12, 7))
                sn.heatmap(df_cm, annot=True, fmt='g')
                plt.ylabel("true")
                plt.xlabel("predicted")
                plt.title("Confusion Matrix")
                try:
                    os.mkdir(os.path.join(LOG_DIR, "confmat"))
                except FileExistsError:
                    pass
                plt.savefig(os.path.join(LOG_DIR, f"confmat/confmat_FOLD{fold + 1}.png"))

            if 'loss_plot' in metrics:
                plt.figure(figsize=(6, 4))
                plt.plot(range(epochs), train_losses, 'aquamarine', label='Train Loss')
                plt.plot(range(epochs), test_losses, 'orange', label='Test Loss')
                plt.legend(loc="upper left")
                plt.ylabel("loss")
                plt.xlabel("epochs")
                plt.title("Train and Test Loss")
                try:
                    os.mkdir(os.path.join(LOG_DIR, "loss_plot"))
                except FileExistsError:
                    pass
                plt.savefig(os.path.join(LOG_DIR, f"loss_plot/loss_plot_FOLD{fold + 1}.png"))

            if 'accuracy_plot' in metrics:
                plt.figure(figsize=(6, 4))
                plt.plot(range(epochs), train_accuracies, 'aquamarine', label='Train Accuracy')
                plt.plot(range(epochs), test_accuracies, 'orange', label='Test Accuracy')
                plt.legend(loc="upper left")
                plt.ylabel("accuracy in %")
                plt.xlabel("epochs")
                plt.ylim((0, 100))
                plt.title("Train and Test Accuracy")
                try:
                    os.mkdir(os.path.join(LOG_DIR, "accuracy_plot"))
                except FileExistsError:
                    pass
                plt.savefig(os.path.join(LOG_DIR, f"accuracy_plot/accuracy_plot_FOLD{fold + 1}.png"))


    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    fold_sum = 0.0
    performance = ""
    for key, value in results.items():
        performance += f'Fold {key}: {value} % \n'
        fold_sum += value
    performance += f'Average: {fold_sum / len(results.items())} %'
    print(performance)

    # write to log file
    performance_file = open(os.path.join(LOG_DIR, "model_performance.txt") ,"w")
    performance_file.write("Performance:\n" + performance)
    performance_file.close()


# example run
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

