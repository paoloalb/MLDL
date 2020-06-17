from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import re


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split

        split_file = open(self.split + ".txt")

        classes = [d.name for d in os.scandir(root) if d.is_dir() if d.name != 'BACKGROUND_Google']
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.dataset = []
        self.indexes = []  # Dataset indexes are provided for later train/val splitting
        self.labels = []

        idx = 0
        # Read the split file line by line
        for line in split_file:
            category = line.split("/")[0]
            if category == "BACKGROUND_Google":
                continue
            path = os.path.join(root, line.strip("\n"))
            img = pil_loader(path)
            self.dataset.append(img)
            self.labels.append(class_to_idx[category])
            self.indexes.append(idx)
            idx += 1

    def __getitem__(self, index):

        image = self.dataset[index]
        label = self.labels[index]
        # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        length = len(self.dataset) # Provide a way to get the length (number of elements) of the dataset
        return length
