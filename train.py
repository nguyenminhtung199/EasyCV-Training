import os
import time
import torch
import copy
import argparse
import json
from torch import nn, optim
from torch.optim import lr_scheduler

from model import ModelFactory
from dataset import get_dataset  
from utils import get_config, seconds_to_text, setup_logging, config2yaml

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.makedirs(cfg.model.save_folder, exist_ok=True)

    def train_model(self, dataloaders, dataset_sizes):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.cfg.num_epochs):
            tik0 = time.time()
            for phase in self.cfg.dataset.phases:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'validation' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                if len(self.cfg.dataset.phases) > 1:
                    if phase == 'validation' and epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        self.model.load_state_dict(best_model_wts)
                        torch.save(self.model, self.save_path)
                else:
                    if phase == 'train' and epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        self.model.load_state_dict(best_model_wts)
                        torch.save(self.model, self.save_path)
            logger.info(
                f'ModelTask: {self.cfg.model.task:<10}  '
                f'Epoch: {epoch+1}/{self.cfg.num_epochs:<3}  '
                f'Loss: {epoch_loss:<8.4f}  '
                f'Acc: {epoch_acc:<8.4f}  '
                f'BestAcc: {best_acc:<8.4f}  '
                f'LearningRate: {self.optimizer.param_groups[0]["lr"]:<10.6f}  '
                f'Time: {(time.time() - tik0):<6.2f}s  '
                f'Required: {seconds_to_text(int((time.time() - tik0) * (self.cfg.num_epochs - 1 - epoch)))}'
            )

        time_elapsed = time.time() - since
        logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        logger.info(f'Best val Acc: {best_acc:4f}')

    def run(self):
        dataloaders, dataset_sizes, class_names = get_dataset(self.cfg)

        self.cfg.model.num_output = len(class_names)
        logger.info("Config:\n%s", config2yaml(self.cfg))
        self.model = ModelFactory.build_model(self.cfg.model)
        self.model.to(self.device)

        self.save_path = os.path.join(self.cfg.model.save_folder, f'{self.cfg.model.name}_{self.cfg.model.task}_{self.cfg.model.type}.pt')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.cfg.model.lr, momentum=self.cfg.model.momentum)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 - epoch / self.cfg.num_epochs)

        logger.info(f"Classes: {class_names}")
        self.train_model(dataloaders, dataset_sizes)
        logger.info(f"Model saved to {self.save_path}")
        if cfg.convert_onnx:
            from onnx_convert import convert_onnx
            output_file_onnx = convert_onnx(self.save_path, cfg.model.multitask, cfg.fix_dynamic_shapes)
            logger.info(f"Model converted to ONNX format: {output_file_onnx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config file")
    args = parser.parse_args()
    cfg = get_config(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_num)
    logger = setup_logging(cfg.model.save_folder)
    trainer = Trainer(cfg)
    trainer.run()
