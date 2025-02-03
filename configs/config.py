from easydict import EasyDict as edict
config = edict()

config.dataset_path =  "../data/"
config.save_path = "weight/model_v1"
config.name = "model"

config.model = "mbn"
config.multitask = False
config.model_type = 'person_body_walking'

config.batch_size = 64

# config.phases = ["train", "validation"]
config.phases = ["train"]

config.criterion = "CrossEntropyLoss"

config.optimizer = "SGD"
config.lr = 0.001
config.momentum = 0.9

config.step_size = 10
config.gamma = 0.1

config.num_epochs = 500


