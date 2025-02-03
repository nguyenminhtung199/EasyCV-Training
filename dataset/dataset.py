import torch
from torchvision import datasets
import torchvision.transforms as transforms
import os

from dataset.augment import random_augment, resize_and_pad
from dataset.multitask import CustomImageDataset

data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(lambda img: random_augment(img, photometric=transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1))),
        transforms.Lambda(lambda img: resize_and_pad(img, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Lambda(lambda img: resize_and_pad(img, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def get_dataset(cfg):
    data_dir = os.path.join(cfg.dataset_path, cfg.model_type)
    if cfg.multitask: 
        image_datasets = {x: CustomImageDataset(os.path.join(data_dir, x + ".txt"), data_transforms[x])
            for x in cfg.phases}
    else: 
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                            for x in cfg.phases}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=cfg.batch_size,
                                        shuffle=True, num_workers=4)
        for x in cfg.phases}
    dataset_sizes = {x: len(image_datasets[x]) for x in cfg.phases}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names
