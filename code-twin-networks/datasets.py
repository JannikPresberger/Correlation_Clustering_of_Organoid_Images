import os
import random
from collections import defaultdict
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from tifffile import imread
from torch.utils.data import Dataset
import albumentations as A
from torchvision.transforms import Compose
from torchvision.transforms.v2 import ToDtype


class RandomApply(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, p: float = 0.5):
        super(RandomApply, self).__init__()
        self.module = module
        self.p = p

    def forward(self, tensor):
        if torch.rand(1) <= self.p:
            return self.module.forward(tensor)

        return tensor


class RandomExclusiveListApply(torch.nn.Module):
    """
    Applies exactly one of the transformations with the given probablities
    """

    def __init__(self, choice_modules: torch.nn.ModuleList, probabilities: np.ndarray = None):
        super(RandomExclusiveListApply, self).__init__()
        self.choice_modules = choice_modules
        if probabilities:
            self.probabilities = torch.tensor(probabilities / np.sum(probabilities))
        else:
            self.probabilities = torch.tensor(np.ones(len(choice_modules)) / len(choice_modules))

    def forward(self, tensor):
        if len(self.choice_modules) == 0:
            return tensor
        # todo: check if multithreading here is a problem
        module_index = torch.multinomial(self.probabilities, num_samples=1)
        # module_index = np.random.choice(range(len(self.choice_modules)), p=self.probabilities)
        return self.choice_modules[module_index].forward(tensor)


class RandomOrganoidHistPairDataset(Dataset):
    def __init__(self,
                 data_dirs: Union[str, List[str]],
                 split: str,
                 num_batches: int = None,
                 batch_size: int = None
                 ):
        super(RandomOrganoidHistPairDataset, self).__init__()

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.num_batches = num_batches
        self.batch_size = batch_size

        self.grouped_examples = defaultdict(list)

        for data_dir in data_dirs:
            for file_class in os.listdir(os.path.join(data_dir, split)):
                directory = os.path.join(data_dir, split, file_class)
                for file in os.listdir(directory):
                    self.grouped_examples[file_class].append(os.path.join(directory, file))

        self.organoid_classes = list(self.grouped_examples.keys())
        self.num_examples = sum(map(lambda x: len(x), self.grouped_examples.values()))

    def __len__(self):
        if self.num_batches and self.batch_size:
            return self.num_batches * self.batch_size

        return self.num_examples ** 2

    def __getitem__(self, index):
        # pick some random class for the first image
        selected_class = random.choice(self.organoid_classes)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, len(self.grouped_examples[selected_class]) - 1)

        # pick the index to get the first image
        path_1 = self.grouped_examples[selected_class][random_index_1]

        # get the first image
        with open(path_1, "rb") as f:
            hist_1 = np.load(f)

        hist_1 = torch.tensor(hist_1.astype(float), dtype=torch.float32)

        # same class
        if index % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, len(self.grouped_examples[selected_class]) - 1)

            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, len(self.grouped_examples[selected_class]) - 1)

            # pick the index to get the second image
            path_2 = self.grouped_examples[selected_class][random_index_2]

            # get the second image
            with open(path_2, "rb") as f:
                hist_2 = np.load(f)
            hist_2 = torch.tensor(hist_2.astype(float), dtype=torch.float32)

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)

        # different class
        else:
            # pick a random class
            other_selected_class = random.choice(self.organoid_classes)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.choice(self.organoid_classes)

            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, len(self.grouped_examples[other_selected_class]) - 1)

            # pick the index to get the second image
            path_2 = self.grouped_examples[other_selected_class][random_index_2]

            with open(path_2, "rb") as f:
                hist_2 = np.load(f)

            # get the second image
            hist_2 = torch.tensor(hist_2.astype(float), dtype=torch.float32)

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        return hist_1, hist_2, target


class RandomOrganoidPairDataset(Dataset):
    def __init__(self,
                 data_dirs: str,
                 split: str,
                 num_batches: int = None,
                 batch_size: int = None,
                 transforms: torchvision.transforms.Compose = None
                 ):
        super(RandomOrganoidPairDataset, self).__init__()

        self.transforms = transforms
        self.num_batches = num_batches
        self.batch_size = batch_size

        self.grouped_examples = defaultdict(list)

        for data_dir in data_dirs:
            for file_class in os.listdir(os.path.join(data_dir, split)):
                directory = os.path.join(data_dir, split, file_class)
                for file in os.listdir(directory):
                    self.grouped_examples[file_class].append(os.path.join(directory, file))

        self.organoid_classes = list(self.grouped_examples.keys())

    def __len__(self):
        return self.num_batches * self.batch_size

    def __getitem__(self, index):
        # pick some random class for the first image
        selected_class = random.choice(self.organoid_classes)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, len(self.grouped_examples[selected_class]) - 1)

        # pick the index to get the first image
        path_1 = self.grouped_examples[selected_class][random_index_1]

        # get the first image

        image_1 = torch.tensor(imread(path_1).astype(float), dtype=torch.float32).div(2**16 - 1)

        # same class
        if index % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, len(self.grouped_examples[selected_class]) - 1)

            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, len(self.grouped_examples[selected_class]) - 1)

            # pick the index to get the second image
            path_2 = self.grouped_examples[selected_class][random_index_2]

            # get the second image
            image_2 = torch.tensor(imread(path_2).astype(float), dtype=torch.float32).div(2**16 - 1)

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)

        # different class
        else:
            # pick a random class
            other_selected_class = random.choice(self.organoid_classes)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.choice(self.organoid_classes)

            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, len(self.grouped_examples[other_selected_class]) - 1)

            # pick the index to get the second image
            path_2 = self.grouped_examples[other_selected_class][random_index_2]

            # get the second image
            image_2 = torch.tensor(imread(path_2).astype(float), dtype=torch.float32).div(2**16 - 1)

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        image_1 = image_1.permute(2, 0, 1)[:3, :, :]
        image_2 = image_2.permute(2, 0, 1)[:3, :, :]


        if self.transforms is not None:
            return self.transforms(image_1), self.transforms(image_2), target
        return image_1, image_2, target


class DeterministicOrganoidDataset(Dataset):

    def __init__(self,
                 data_dirs: List[str],
                 transforms: nn.Module = None,
                 return_file_ids: bool = False
                 ):
        super(DeterministicOrganoidDataset, self).__init__()

        self.transforms = transforms
        self.paths = []
        self.labels = []

        for data_dir in data_dirs:
            for file_class in os.listdir(os.path.join(data_dir)):
                directory = os.path.join(data_dir, file_class)
                for file in os.listdir(directory):
                    self.paths.append(os.path.join(directory, file))
                    self.labels.append(file_class)

        self.num_examples = len(self.paths)
        self.return_file_ids = return_file_ids

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        index = index

        label = self.labels[index]

        path = self.paths[index]

        image = torch.tensor(imread(path).astype(float), dtype=torch.float32).permute(2, 0, 1)[:3, :, :].div(2**16 - 1)

        if self.transforms is not None:
            image = self.transforms(image)

        if self.return_file_ids:
            return image, label, path
        else:
            return image, label


class DeterministicOrganoidPairDataset(Dataset):
    def __init__(self,
                 data_dirs: str,
                 split: str,
                 transforms: nn.Module = None
                 ):
        super(DeterministicOrganoidPairDataset, self).__init__()

        self.transforms = transforms
        self.paths = []
        self.organoid_classes = []

        for data_dir in data_dirs:
            for file_class in os.listdir(os.path.join(data_dir, split)):
                directory = os.path.join(data_dir, split, file_class)
                for file in os.listdir(directory):
                    self.paths.append(os.path.join(directory, file))
                    self.organoid_classes.append(file_class)

        self.num_examples = len(self.paths)

    def __len__(self):
        return self.num_examples * self.num_examples

    def __getitem__(self, index):
        index_1 = index % self.num_examples
        index_2 = index // self.num_examples

        path_1 = self.paths[index_1]
        path_2 = self.paths[index_2]

        image_1 = torch.tensor(imread(path_1).astype(float), dtype=torch.float32).permute(2, 0, 1)[:3, :, :]
        image_2 = torch.tensor(imread(path_2).astype(float), dtype=torch.float32).permute(2, 0, 1)[:3, :, :]

        if self.organoid_classes[index_1] == self.organoid_classes[index_2]:
            target = torch.tensor(1, dtype=torch.float)
        else:
            target = torch.tensor(0, dtype=torch.float)

        image_1 = image_1.div(2**16 - 1)
        image_2 = image_2.div(2**16 - 1)

        if self.transforms is not None:
            return self.transforms(image_1), self.transforms(image_2), target
        return image_1, image_2, target


class DeterministicOrganoidHistDataset(Dataset):

    def __init__(self,
                 data_dirs: List[str],
                 return_file_ids: bool = False
                 ):
        super(DeterministicOrganoidHistDataset, self).__init__()

        self.paths = []
        self.labels = []

        for data_dir in data_dirs:
            for file_class in os.listdir(os.path.join(data_dir)):
                directory = os.path.join(data_dir, file_class)
                for file in os.listdir(directory):
                    self.paths.append(os.path.join(directory, file))
                    self.labels.append(file_class)

        self.num_examples = len(self.paths)
        self.return_file_ids = return_file_ids

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        label = self.labels[index]

        path = self.paths[index]

        with open(path, "rb") as f:
            hist = np.load(f)

        hist = torch.tensor(hist.astype(float), dtype=torch.float32)

        if self.return_file_ids:
            return hist, label, path
        else:
            return hist, label


class DeterministicOrganoidHistPairDataset(Dataset):
    def __init__(self,
                 data_dirs: Union[str, List[str]],
                 split: str
                 ):
        super(DeterministicOrganoidHistPairDataset, self).__init__()

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.paths = []
        self.organoid_classes = []

        for data_dir in data_dirs:
            for file_class in os.listdir(os.path.join(data_dir, split)):
                directory = os.path.join(data_dir, split, file_class)
                for file in os.listdir(directory):
                    self.paths.append(os.path.join(directory, file))
                    self.organoid_classes.append(file_class)

        self.num_examples = len(self.paths)

    def __len__(self):
        return self.num_examples * self.num_examples

    def __getitem__(self, index):
        index_1 = index % self.num_examples
        index_2 = index // self.num_examples

        path_1 = self.paths[index_1]
        path_2 = self.paths[index_2]

        with open(path_1, "rb") as f:
            hist_1 = np.load(path_1)
        with open(path_2, "rb") as f:
            hist_2 = np.load(path_2)

        hist_1 = torch.tensor(hist_1.astype(float), dtype=torch.float32)
        hist_2 = torch.tensor(hist_2.astype(float), dtype=torch.float32)

        if self.organoid_classes[index_1] == self.organoid_classes[index_2]:
            target = torch.tensor(1, dtype=torch.float)
        else:
            target = torch.tensor(0, dtype=torch.float)

        return hist_1, hist_2, target



class OnlineDeterministicOrganoidHistPairDataset(Dataset):
    def __init__(self,
                 data_dirs: Union[str, List[str]],
                 split: str
                 ):
        super(OnlineDeterministicOrganoidHistPairDataset, self).__init__()

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.paths = []
        self.organoid_classes = []

        for data_dir in data_dirs:
            for file_class in os.listdir(os.path.join(data_dir, split)):
                directory = os.path.join(data_dir, split, file_class)
                for file in os.listdir(directory):
                    self.paths.append(os.path.join(directory, file))
                    self.organoid_classes.append(file_class)

        self.num_examples = len(self.paths)

    def __len__(self):
        return self.num_examples * self.num_examples

    def compute_histogram(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        bgr_planes = cv2.split(image)
        histSize = [2 ** 16]
        histRange = [0, 2 ** 16]

        histograms = np.empty(shape=(3, 2 ** 16))

        for i in range(3):
            histogram = cv2.calcHist(bgr_planes, [i], None, histSize, histRange, accumulate=False)
            cv2.normalize(histogram, histogram, 1, 0, cv2.NORM_L1)
            histograms[i, :] = histogram[:, 0]
            histograms[i, :] = np.cumsum(histograms[i, :])
        return histograms


    def __getitem__(self, index):
        index_1 = index % self.num_examples
        index_2 = index // self.num_examples

        path_1 = self.paths[index_1]
        path_2 = self.paths[index_2]

        image_1 = imread(path_1)
        image_1 = torch.tensor(image_1.astype(float), dtype=torch.float32).permute(2, 0, 1)[:3, :, :].div(2 ** 16 - 1)
        hist_1 = self.compute_histogram(np.uint16(image_1.permute(1, 2, 0).multiply(2 ** 16 - 1)))

        image_2 = imread(path_2)
        image_2 = torch.tensor(image_2.astype(float), dtype=torch.float32).permute(2, 0, 1)[:3, :, :].div(2 ** 16 - 1)
        hist_2 = self.compute_histogram(np.uint16(image_2.permute(1, 2, 0).multiply(2 ** 16 - 1)))


        hist_1 = torch.tensor(hist_1.astype(float), dtype=torch.float32)
        hist_2 = torch.tensor(hist_2.astype(float), dtype=torch.float32)

        if self.organoid_classes[index_1] == self.organoid_classes[index_2]:
            target = torch.tensor(1, dtype=torch.float)
        else:
            target = torch.tensor(0, dtype=torch.float)

        return hist_1, hist_2, target


class OnlineRandomOrganoidHistPairDataset(Dataset):
    def __init__(self,
                 data_dirs: Union[str, List[str]],
                 split: str,
                 transforms,
                 num_batches: int = None,
                 batch_size: int = None
                 ):
        super(OnlineRandomOrganoidHistPairDataset, self).__init__()

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.num_batches = num_batches
        self.batch_size = batch_size

        self.grouped_examples = defaultdict(list)

        for data_dir in data_dirs:
            for file_class in os.listdir(os.path.join(data_dir, split)):
                directory = os.path.join(data_dir, split, file_class)
                for file in os.listdir(directory):
                    self.grouped_examples[file_class].append(os.path.join(directory, file))

        self.organoid_classes = list(self.grouped_examples.keys())
        self.num_examples = sum(map(lambda x: len(x), self.grouped_examples.values()))
        self.transforms = transforms

    def compute_histogram(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        bgr_planes = cv2.split(image)
        histSize = [2 ** 16]
        histRange = [0, 2 ** 16]

        histograms = np.empty(shape=(3, 2 ** 16))

        for i in range(3):
            histogram = cv2.calcHist(bgr_planes, [i], None, histSize, histRange, accumulate=False)
            cv2.normalize(histogram, histogram, 1, 0, cv2.NORM_L1)
            histograms[i, :] = histogram[:, 0]
            histograms[i, :] = np.cumsum(histograms[i, :])
        return histograms

    def __len__(self):
        if self.num_batches and self.batch_size:
            return self.num_batches * self.batch_size

        return self.num_examples ** 2

    def __getitem__(self, index):
        # pick some random class for the first image
        selected_class = random.choice(self.organoid_classes)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, len(self.grouped_examples[selected_class]) - 1)

        # pick the index to get the first image
        path_1 = self.grouped_examples[selected_class][random_index_1]

        # get the first image
        image_1 = imread(path_1)
        image_1 = torch.tensor(image_1.astype(float), dtype=torch.float32).permute(2, 0, 1)[:3, :, :].div(2**16 - 1)
        image_1 = self.transforms(image_1)
        hist_1 = self.compute_histogram(np.uint16(image_1.permute(1, 2, 0).multiply(2**16 - 1)))

        hist_1 = torch.tensor(hist_1.astype(float), dtype=torch.float32)

        # same class
        if index % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, len(self.grouped_examples[selected_class]) - 1)

            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, len(self.grouped_examples[selected_class]) - 1)

            # pick the index to get the second image
            path_2 = self.grouped_examples[selected_class][random_index_2]
            image_2 = imread(path_2)
            image_2 = torch.tensor(image_2.astype(float), dtype=torch.float32).permute(2, 0, 1)[:3, :, :].div(2**16 - 1)
            image_2 = self.transforms(image_2)
            hist_2 = self.compute_histogram(np.uint16(image_2.permute(1, 2, 0).multiply(2 ** 16 - 1)))

            hist_2 = torch.tensor(hist_2.astype(float), dtype=torch.float32)

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)

        # different class
        else:
            # pick a random class
            other_selected_class = random.choice(self.organoid_classes)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.choice(self.organoid_classes)

            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, len(self.grouped_examples[other_selected_class]) - 1)

            # pick the index to get the second image
            path_2 = self.grouped_examples[other_selected_class][random_index_2]
            image_2 = imread(path_2)
            image_2 = torch.tensor(image_2.astype(float), dtype=torch.float32).permute(2, 0, 1)[:3, :, :].div(2**16 - 1)
            image_2 = self.transforms(image_2)
            hist_2 = self.compute_histogram(np.uint16(image_2.permute(1, 2, 0).multiply(2 ** 16 - 1)))

            hist_2 = torch.tensor(hist_2.astype(float), dtype=torch.float32)

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        return hist_1, hist_2, target


class SquarePad(nn.Module):
    def __call__(self, image):
        s = image.size()
        max_wh = np.max([s[-1], s[-2]])
        hp = int((max_wh - s[-1]) / 2)
        vp = int((max_wh - s[-2]) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0.0, 'constant')



def get_dataset_means_and_stds(
        data_dir: str,
        resize_size: int = 512,
        pad: bool = True,
        out_dir: str = "dataset_stats"
):
    dataset = DeterministicOrganoidDataset(
        data_dirs=[data_dir],
        transforms=torch.nn.Sequential(
            torchvision.transforms.ToTensor(),
            SquarePad(),
            torchvision.transforms.Resize(
                (resize_size, resize_size),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True
            )
        )
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

    sums = torch.zeros((3,), dtype=torch.float64)
    squared_sums = torch.zeros((3,), dtype=torch.float64)
    num_samples = len(dataset) * resize_size ** 2

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            sums += torch.sum(images, dim=(0, 2, 3))
            squared_sums += torch.sum(
                images ** 2,
                dim=(0, 2, 3)
            )

    mean = sums / num_samples
    squared_mean = squared_sums / num_samples
    std = torch.sqrt(squared_mean - mean ** 2)

    print(mean)
    print(std)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fn = f"stats_r{resize_size} "


if __name__ == "__main__":
    train_dataset = RandomOrganoidPairDataset(
        ["/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids New/split/"],
        split="train",
        num_batches=100,
        batch_size=64,
        transforms=torchvision.transforms.Compose(
            [
                SquarePad(),
                torchvision.transforms.Resize(
                    (256, 256),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )
            ]
        ))

    # brightness from 0.5 -> 2.0
    # contrast from 0.5 -> 2.0
    # saturation from 0.5 -> 2.0
    # hue from -0.05 to 0.05

    transform = torchvision.transforms.ColorJitter(
        brightness=(0.5, 2.0),
        contrast=(0.5, 2.0),
        saturation=(0.5, 1.5),
        hue=(0.05, 0.05)
    )

    for (image1, image2, label) in train_dataset:
        image1aug = transform(image1)
        print(image2.shape)
        image2aug = transform(image2)

        print(image1.max())
        print(image1aug.max())

        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(image1.permute(1, 2, 0))
        axes[0, 1].imshow(image1aug.permute(1, 2, 0))
        axes[1, 0].imshow(image2.permute(1, 2, 0))
        axes[1, 1].imshow(image2aug.permute(1, 2, 0))
        plt.show()

    # dataset = RandomOrganoidHistPairDataset(
    #     data_dirs=[
    #         "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/hist-dataset"
    #     ],
    #     split="train"
    # )
    #
    # print(len(dataset))
    #
    # for hist1, hist2, target in dataset:
    #     print(hist1.shape)

    # get_dataset_means_and_stds(
    #     data_dir="/run/media/dstein/NVMe/Organoid Daten/dataset/train",
    #     pad=True,
    #     out_dir="",
    #     resize_size=256
    # )

    # train_dataset = RandomOrganoidPairDataset(
    #     data_dirs=["/run/media/dstein/NVMe/Organoid Daten/dataset/"],
    #     split="train",
    #     num_batches=1,
    #     batch_size=64,
    #     transforms=torch.nn.Sequential(
    #         torchvision.transforms.CenterCrop((64, 128)),
    #         SquarePad(),
    #         # torchvision.transforms.Resize(
    #         #     (256, 256),
    #         #     interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    #         #     antialias=True
    #         # )
    #     ))
    # # path_1 = "/run/media/dstein/NVMe/Organoid Daten/dataset/train/6/single_organoid_60.tif"
    # # image_1 = imread(path_1)
    # # plt.figure()
    # # image_1 = torch.tensor(image_1.astype(float), dtype=torch.float32).permute(2, 0, 1).permute(1, 2, 0)
    # # image_1 = image_1 / 65535
    # # print(image_1.max())
    # # print(image_1.min())
    # #
    # # resize = torchvision.transforms.Resize((224, 224))
    # # image_1 = resize(image_1)
    # #
    # # plt.imshow(image_1)
    # # plt.show()
    # for (image1, image2, target) in train_dataset:
    #     plt.figure()
    #     image1 = image1.permute(1, 2, 0) / 65535
    #     print(image1.max())
    #     print(image1.min())
    #
    #     plt.imshow(image1)
    #     plt.show()
