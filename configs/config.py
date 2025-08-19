from easydict import EasyDict as edict

config = edict()
config.gpu_num = 4
config.num_epochs = 250
config.batch_size = 256
config.num_workers = 4
config.convert_onnx = True
config.fix_dynamic_shapes = False


# Model configuration
config.model = edict()
config.model.name = "model"
config.model.type = "resnet18"
config.model.multitask = False
config.model.task = 'mask'
config.model.save_folder = "weight/model_v2"

config.model.criterion = "CrossEntropyLoss"
config.model.optimizer = "SGD"
config.model.lr = 0.001
config.model.momentum = 0.9
config.model.step_size = 10
config.model.gamma = 0.1


# Dataset configuration
config.dataset = edict()
config.dataset.dataset_path =  "/home1/data/.DATASETS/tungcao/mask"
config.dataset.phases = ["train", "validation"]
# config.dataset.phases = ["train"]
config.dataset.image_size = [224, 224]
config.dataset.norm_mean = [0.485, 0.456, 0.406]
config.dataset.norm_std = [0.229, 0.224, 0.225]

# Data augmentation configuration
config.dataset.augmentation = edict()
config.dataset.augmentation.blur = edict()
config.dataset.augmentation.blur.prob = 0.4

config.dataset.augmentation.color_jitter = edict()
config.dataset.augmentation.color_jitter.prob = 0.3
config.dataset.augmentation.color_jitter.ratio = [0.3, 0.3, 0.3, 0.0] # brightness / contrast / saturation / hue
config.dataset.augmentation.color_jitter.range = [0.2, 0.2, 0.2, 0.05] # brightness / contrast / saturation / hue

config.dataset.augmentation.resize = edict()
config.dataset.augmentation.resize.prob = 0.4
config.dataset.augmentation.resize.range = [24, 168]

config.dataset.augmentation.gaussian_noise = edict()
config.dataset.augmentation.gaussian_noise.prob = 0.3
config.dataset.augmentation.gaussian_noise.mean = 0.0
config.dataset.augmentation.gaussian_noise.var = 0.5

config.dataset.augmentation.flip = edict()
config.dataset.augmentation.flip.prob = 0.5
config.dataset.augmentation.flip.direction = 1 #  0: vertical 1: horizontal, 2: both

config.dataset.augmentation.mask = edict()
config.dataset.augmentation.mask.prob = 0.0
config.dataset.augmentation.mask.max_ratio = 0.1
