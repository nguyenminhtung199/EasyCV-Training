from encodings.punycode import T
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from dataset.augment import random_augment, transform_resize_padding

class CustomImageDataset(Dataset):
    def __init__(self, config, phase: str = "train", augment: bool = False):
        assert phase in ["train", "validation"], f"""Input must be in {["train", "validation"]}"""

        super(CustomImageDataset, self).__init__()
        self.config = config
        self.augment = augment
        self.images = []
        self.classes = []

        if isinstance(self.config.dataset_path, str):
            for class_dir in os.listdir(os.path.join(self.config.dataset_path, phase)):
                self.classes.append(class_dir)

            self.images = self.get_images(os.path.join(self.config.dataset_path, phase))

        elif isinstance(self.config.dataset_path, list):
            for dir_path in os.path.join(self.config.dataset_path, phase):
                for class_dir in os.listdir(dir_path):
                    self.classes.append(class_dir)

                self.images = self.get_images(dir_path)
        else:
            assert False, f"""dataset_path must be a string or a list of strings, got {type(self.config.dataset_path)}"""

        label = [x.split("/")[-2] for x in self.images]
        self.count_label = {i:label.count(i) for i in self.classes}
        for class_name in self.classes:
            assert class_name in self.count_label.keys(), f""""{class_name}" class is missing image"""
        
    def get_images(self, folder_name: str):
        for class_dir in os.listdir(folder_name):
            for image_name in os.listdir(os.path.join(folder_name, class_dir)):
                self.images.append(os.path.join(folder_name, class_dir, image_name))
        return self.images

    def preprocess_image_transform(self, image):
        image = image.astype(np.float32) / 255.0

        mean = np.array(self.config.norm_mean, dtype=np.float32)
        std = np.array(self.config.norm_std, dtype=np.float32)
        image = (image - mean) / std
        
        image = np.transpose(image, (2, 0, 1))

        return image
    
    def preprocess_image_augmentation(self, image):
        if self.augment:
            image = random_augment(sample=image, cfg=self.config.augmentation)

        image = transform_resize_padding(sample=image, target_size=self.config.image_size)
        return image
    
    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # augment
        image = self.preprocess_image_augmentation(image)
        # transform
        image = self.preprocess_image_transform(image)
        
        class_name = self.images[index].split('/')[-2]
        label = self.classes.index(class_name)
        return image, label

    def __len__(self):
        return len(self.images)
    

