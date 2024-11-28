"""
EECS 445 - Introduction to Machine Learning
Fall 2023 - Project 2

Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from model.challenge import Challenge
from train_common import *
from utils import config
import utils
import copy


def freeze_layers(model, num_layers=0):
    """Stop tracking gradients on selected layers."""
    # TODO: modify model with the given layers frozen
    #      e.g. if num_layers=2, freeze CONV1 and CONV2
    #      Hint: https://pytorch.org/docs/master/notes/autograd.html

    for i, (name, param) in enumerate(model.named_parameters()):
        if 'conv' in name and (i < num_layers * 2):
            param.requires_grad = False
        else:
            break


def train(tr_loader, va_loader, te_loader, model, model_name, num_layers=0):
    """Train transfer learning model."""
    # TODO: Define loss function and optimizer. Replace "None" with the appropriate definitions.
    criterion = torch.nn.CrossEntropyLoss()
    #TRY DIFFEREENT WEIGHT DECAY AND LEARNING RATES
    optimizer = torch.optim.Adam(model.parameters(),lr=10**-2)

    print("Loading target model with", num_layers, "layers frozen")
    model, start_epoch, stats = restore_checkpoint(model, model_name)

    axes = utils.make_training_plot("Target Training")

    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
    )

    # initial val loss for early stopping
    global_min_loss = stats[0][1]

    # TODO: Define patience for early stopping. Replace "None" with the patience value.
    #TRY DIFFERENT PATIENCE
    patience = 5
    curr_count_to_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_count_to_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, model_name, stats)

        curr_count_to_patience, global_min_loss = early_stopping(
            stats, curr_count_to_patience, global_min_loss
        )
        epoch += 1

    print("Finished Training")

    # Keep plot open
    utils.save_tl_training_plot(num_layers)
    utils.hold_training_plot()



def main():
    # Data loaders
    if check_for_augmented_data("./data"):
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target", batch_size=config("challenge.batch_size"), augment=True
        )
    else:
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target",
            batch_size=config("challenge.batch_size"),
        )
    # Model

    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("target.batch_size"),
    )

    freeze_none = Challenge()

    print("Loading source...")
    freeze_none, _, _ = restore_checkpoint(
        freeze_none, config("source.checkpoint"), force=True, pretrain=True
    )

    #freeze_one = copy.deepcopy(freeze_none)
    freeze_two = copy.deepcopy(freeze_none)
    #freeze_three = copy.deepcopy(freeze_none)

    #freeze_layers(freeze_one, 1)
    freeze_layers(freeze_two, 2)
    #freeze_layers(freeze_three, 3)


    #try challenge on each
    #train(tr_loader, va_loader, te_loader, freeze_none, "./checkpoints/target0/", 0)
    #train(tr_loader, va_loader, te_loader, freeze_one, "./checkpoints/challenge/", 1)
    train(tr_loader, va_loader, te_loader, freeze_two, "./checkpoints/challenge/", 2)
    #train(tr_loader, va_loader, te_loader, freeze_three, "./checkpoints/target3/", 3)

    


if __name__ == "__main__":
    main()
