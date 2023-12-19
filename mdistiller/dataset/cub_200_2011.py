import os
import time

import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets  #, transforms
from PIL import Image

from mdistiller.dataset import transforms


IMAGE_SIZE = 224 #448
TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
TEST_MEAN = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
TEST_STD = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]


def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder

class CUB(Dataset):

    def __init__(self, path, train=True, transform=None, target_transform=None):

        self.root = path
        self.is_train = train
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = {}
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id
        
        self.data_id = []
        if self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if int(is_train):
                        self.data_id.append(image_id)
        if not self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if not int(is_train):
                        self.data_id.append(image_id)

    def __len__(self):
        return len(self.data_id)
    
    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.data_id[index]
        class_id = int(self._get_class_by_id(image_id)) - 1
        path = self._get_path_by_id(image_id)
        image = cv2.imread(os.path.join(self.root, 'images', path))
        
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            class_id = self.target_transform(class_id)
        return image, class_id

    def _get_path_by_id(self, image_id):

        return self.images_path[image_id]
    
    def _get_class_by_id(self, image_id):

        return self.class_ids[image_id]
    
# class CIFAR100Instance(datasets.CIFAR100):
#     """CIFAR100Instance Dataset."""

#     def __getitem__(self, index):
#         img, target = super().__getitem__(index)
#         return img, target, index

class CUBInstance(CUB):
    """CIFAR100Instance Dataset."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class CUBInstanceSample(CUB):
    """
    CUB+Sample Dataset
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        k=4096,
        mode="exact",
        is_sample=True,
        percent=1.0,
    ):
        super().__init__(
            root=root,
            train=train,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )
        
        self.root = path
        self.is_train = train
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = {}
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id
        
        self.data_id = []
        if self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if int(is_train):
                        self.data_id.append(image_id)
        if not self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if not int(is_train):
                        self.data_id.append(image_id)
                        
        self.k = k
        self.mode = mode
        self.is_sample = is_sample
        
        self.targets=[int(self._get_class_by_id(image_id)) - 1 for image_id in self.data_id]
        
        num_classes = 200
        num_samples = len(self.data_id)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [
            np.asarray(self.cls_positive[i]) for i in range(num_classes)
        ]
        self.cls_negative = [
            np.asarray(self.cls_negative[i]) for i in range(num_classes)
        ]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [
                np.random.permutation(self.cls_negative[i])[0:n]
                for i in range(num_classes)
            ]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data_id[index], self.targets[index]
        p = self._get_path_by_id(img)
        img = cv2.imread(os.path.join(self.root, 'images', p))
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == "exact":
                pos_idx = index
            elif self.mode == "relax":
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(
                self.cls_negative[target], self.k, replace=replace
            )
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx



def get_cub_200_2011_train_transform():
    train_transform=transforms.Compose([
            transforms.ToCVImage(),
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
        ])

    return train_transform


def get_cub_200_2011_test_transform():
    test_transforms=transforms.Compose([
        transforms.ToCVImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(TEST_MEAN,TEST_STD)
        ])
    return test_transforms


def get_cub_200_2011_dataloader(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_transforms = get_cub_200_2011_train_transform()
    test_transforms = get_cub_200_2011_test_transform()
    path=data_folder + '/CUB_200_2011'
    
    train_dataset = CUBInstance(
            path,
            train=True,
            transform=train_transforms,
            target_transform=None,
        )
        # print(len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    num_data = len(train_dataset)
    test_dataset = CUB(
            path,
            train=False,
            transform=test_transforms,
            target_transform=None
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    return train_loader, test_loader, num_data


# 
def get_cub_200_2011_dataloader_sample(
    batch_size, val_batch_size, num_workers, k, mode="exact"
):
    data_folder = get_data_folder()
    train_transforms = get_cub_200_2011_train_transform()
    test_transforms = get_cub_200_2011_test_transform()
    
    path=data_folder + '/CUB_200_2011'
    train_dataset = CUBInstanceSample(
            path,
            train=True,
            transform=train_transforms,
            target_transform=None,
            k=k,
            mode=mode,
            is_sample=True,
            percent=1.0,
        )
        # print(len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    num_data = len(train_dataset)
    test_dataset = CUB(
            path,
            train=False,
            transform=test_transforms,
            target_transform=None
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    return train_loader, test_loader, num_data
