import torch
import os

def get_dataset(cfg):
    if cfg.model.multitask: 
        from dataset.multitask import CustomImageDataset
        image_datasets = {x: CustomImageDataset(os.path.join(cfg.dataset_path, x + ".txt"))
            for x in cfg.dataset.phases}
    else: 
        from dataset.dataloader import CustomImageDataset
        image_datasets = {x: CustomImageDataset(cfg.dataset, x, augment=(x == 'train'))
                            for x in cfg.dataset.phases}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=cfg.batch_size,
                                        shuffle=True, num_workers=cfg.num_workers)
        for x in cfg.dataset.phases}
    dataset_sizes = {x: len(image_datasets[x]) for x in cfg.dataset.phases}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names
