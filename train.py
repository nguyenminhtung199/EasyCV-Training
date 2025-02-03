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
from utils import get_config, seconds_to_text, setup_logging

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.makedirs(cfg.save_path, exist_ok=True)

    def train_model(self, dataloaders, dataset_sizes):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.cfg.num_epochs):
            tik0 = time.time()
            for phase in self.cfg.phases:
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

                if len(cfg.phases) > 1: 
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
                f'ModelType: {self.cfg.model_type:<10}  '
                f'Epoch: {epoch}/{self.cfg.num_epochs - 1:<3}  '
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

        self.cfg.num_output = len(class_names)
        logger.info("Config:\n%s", json.dumps(self.cfg, indent=4))
        self.model = ModelFactory.build_model(self.cfg)
        self.model.to(self.device)

        self.save_path = os.path.join(self.cfg.save_path, f'{self.cfg.name}{self.cfg.model_type}.pt')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 - epoch / self.cfg.num_epochs)

        logger.info(f"Classes: {class_names}")
        self.train_model(dataloaders, dataset_sizes)
        logger.info(f"Model saved to {self.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config file")
    args = parser.parse_args()
    cfg = get_config(args.config)
    logger = setup_logging(cfg.save_path)
    trainer = Trainer(cfg)
    trainer.run()