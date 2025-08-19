import torch
import torch.nn as nn

class ONNXProcessedModelMultiTask(nn.Module):
    def __init__(self, model):
        super(ONNXProcessedModel, self).__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.model = model
        self.model.eval()
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x / 255.0
        R = (x[:, 0:1, :, :] - self.mean[0]) / self.std[0]
        G = (x[:, 1:2, :, :] - self.mean[1]) / self.std[1]
        B = (x[:, 2:3, :, :] - self.mean[2]) / self.std[2]
        x = torch.cat([R, G, B], dim=1)
        # x = self.model(x)
        # x = self.sm(x)
        out_mask, out_glass, out_cap = self.model(x)
        out_mask = torch.sigmoid(out_mask)
        out_glass = torch.sigmoid(out_glass)
        out_cap = torch.sigmoid(out_cap)

        return out_mask, out_glass, out_cap
        # return x[0, 1]

class ONNXProcessedModel(nn.Module):
    def __init__(self, model):
        super(ONNXProcessedModel, self).__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.model = model
        self.model.eval()
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x / 255.0
        R = (x[:, 0:1, :, :] - self.mean[0]) / self.std[0]
        G = (x[:, 1:2, :, :] - self.mean[1]) / self.std[1]
        B = (x[:, 2:3, :, :] - self.mean[2]) / self.std[2]
        x = torch.cat([R, G, B], dim=1)
        x = self.model(x)
        x = self.sm(x)

        return x[:, 0:1]
        # return x[0]

# class ONNXProcessedModel(nn.Module):
#     def __init__(self, model):
#         super(ONNXProcessedModel, self).__init__()
#         self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
#         self.model = model
#         self.model.eval()
#         self.sm = nn.Softmax(dim=-1)

#     def forward(self, x):
#         x = x / 255.0
#         x = (x - self.mean) / self.std
#         x = self.model(x)
#         x = torch.sigmoid(x)

#         return x[:, 0:1]
        # return x[0]