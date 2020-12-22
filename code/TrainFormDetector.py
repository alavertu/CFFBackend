import argparse
import copy
import os

import numpy as np
from imgaug import augmenters as iaa
from tqdm import tqdm

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from training_utils import (
    ImgAugTransform,
    AddRandomLines,
    TemplateClassifierDataset,
    adjust_learning_rate,
)
from models import TemplateNet


class TrainFormDetector(object):
    def __init__(
        self,
        path_2_positives,
        out_path,
        path_2_train_negatives="../data/train/negative/",
        path_2_val_negatives="../data/val/negative/",
        verbose=False,
    ):

        self.verbose = verbose
        self.out_path = out_path

        # Prep data set for training
        # List of additional augmentations not offered by pytorch
        aug_list = iaa.Sequential(
            [
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.0))),
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=(0, 0.2))),
                iaa.Sometimes(0.25, iaa.Dropout(p=(0, 0.3))),
                iaa.Sometimes(0.25, iaa.Cutout(nb_iterations=(1, 5), size=0.2, cval=1)),
            ]
        )

        # Form pixel value mean
        form_mu = (0.9094463458110517,)
        # Form pixel value std
        form_std = (0.1274794325726292,)

        # Data transformations for augmenting training data to mimic fax noise
        train_transf = transforms.Compose(
            [
                ImgAugTransform(aug_list),
                transforms.ToPILImage(),
                transforms.Resize((550, 425)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            2.0,
                            translate=(0.03, 0.03),
                            scale=(0.5, 1.0),
                            shear=(2, 2),
                            fillcolor=1.0,
                        )
                    ]
                ),
                transforms.ToTensor(),
                transforms.RandomApply([AddRandomLines(5, 25)]),
                transforms.Normalize(form_mu, form_std),
            ]
        )

        # Load training data
        self.train_data = TemplateClassifierDataset(
            path_2_positives,
            path_2_train_negatives,
            transforms=train_transf,
            verbose=self.verbose,
        )

        # Data transformations for standard input
        val_transf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((550, 425)),
                transforms.ToTensor(),
                transforms.Normalize(form_mu, form_std),
            ]
        )

        self.val_data = TemplateClassifierDataset(
            path_2_positives,
            path_2_val_negatives,
            transforms=val_transf,
            verbose=self.verbose,
            bootstrap_template=False,
        )

    def write_history(self, train_histories, val_histories):
        with open(os.path.join(self.out_path, "train_history.csv"), "w+") as out_file:
            for i in range(len(train_histories)):
                for epoch in range(len(train_histories[i])):
                    out_file.write(
                        f"model{i},train,{epoch},{train_histories[i][epoch]}\n"
                    )
                    out_file.write(f"model{i},val,{epoch},{val_histories[i][epoch]}\n")

    def train_k_models(self, k=5, num_epochs=100, batch_size=64, learning_rate=0.001):

        num_classes = len(self.train_data.index_2_class)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_training_history = []
        model_val_history = []

        self.train_loader = DataLoader(
            dataset=self.train_data, batch_size=batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            dataset=self.val_data, batch_size=batch_size, num_workers=4
        )

        #### GO THROUGH CODE BELOW
        for i in range(k):
            train_loss_history = []
            val_loss_history = []

            model = TemplateNet(num_classes)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion.to(device)

            for epoch in range(1, num_epochs):
                _ = model.train()

                rolling_loss = 0
                for images, labels in tqdm(
                    self.train_loader,
                    total=len(self.train_loader),
                    disable=self.verbose,
                    desc=f"Epoch {epoch}:",
                ):
                    images = images.to(device, dtype=torch.float)

                    labels = torch.LongTensor(labels)
                    labels = labels.to(device, dtype=torch.long)
                    # Forward pass
                    outputs = model(images)

                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    rolling_loss += loss.item()

                # Test the model
                _ = model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for images, labels in tqdm(self.val_loader):
                        images = images.to(device, dtype=torch.float)

                        labels = torch.LongTensor(labels)
                        labels = labels.to(device, dtype=torch.long)

                        outputs = model(images)

                        val_loss += criterion(outputs, labels)

                avg_epoch_loss = rolling_loss / len(self.train_loader)
                avg_val_loss = val_loss / len(self.val_loader)

                if avg_epoch_loss < best_train_loss:
                    best_train_loss = avg_epoch_loss
                    best_model = copy.deepcopy(model.state_dict())
                    rounds_since_best = 0
                else:
                    rounds_since_best += 1

                if rounds_since_best >= 10:
                    adjust_learning_rate(optimizer, 0.9)
                    rounds_since_best = 0

                train_loss_history.append(avg_epoch_loss)
                val_loss_history.append(avg_val_loss)
                train_loss_rounded = np.round(avg_epoch_loss, 4)
                val_loss_rounded = np.round(avg_val_loss.cpu().numpy(), 4)

                if self.verbose and epoch % 10 == 0:
                    print(
                        f"Epoch {epoch}, Train Loss: {train_loss_rounded}, Val Loss: {val_loss_rounded}"
                    )

            model_training_history.append(train_loss_history)
            model_val_history.append(train_loss_history)
            torch.save(
                best_model,
                os.path.join(
                    self.out_path, f"formDet_best{i}_{np.round(best_train_loss, 4)}.pt"
                ),
            )

        self.write_history(model_training_history, model_val_history)

        if self.verbose:
            print("Model training finished.")
            print(f"Find your trained models at {self.output_path}")


"""
Parse the command line
"""


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="TrainFormDetector, Trains a set of k models to recognize templates"
    )
    requiredNamed = parser.add_argument_group("required arguments")
    requiredNamed.add_argument(
        "-p",
        "--positives_dir",
        help="Directory containing positive examples of each template",
        required=True,
    )
    requiredNamed.add_argument(
        "-O", "--output_dir", help="Output directory, must already exist", required=True
    )
    parser.add_argument(
        "-n",
        "--negatives_train_dir",
        help="Directory containing negative training examples",
        default="../data/train/negative/",
    )
    parser.add_argument(
        "-t",
        "--negatives_val_dir",
        help="Directory containing negative training examples",
        default="../data/val/negative/",
    )
    parser.add_argument(
        "-k", "--train_k_models", help="Number of models to train", default=5, type=int
    )
    parser.add_argument(
        "-v", "--verbose", help="Verbose mode", action="store_true", default=False
    )

    options = parser.parse_args()
    return options


"""
Main
"""
if __name__ == "__main__":
    options = parse_command_line()
    TFD = TrainFormDetector(
        options.positives_dir,
        options.output_dir,
        path_2_train_negatives=options.negatives_train_dir,
        path_2_val_negatives=options.negatives_val_dir,
        verbose=options.verbose,
    )
    TFD.train_k_models(options.train_k_models)
