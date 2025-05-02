import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class FlawHighlighter(nn.Module):
    def __init__(self, params, input_image_size=128):
        super().__init__()

        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features

        self.feature_map_size = input_image_size // (2 ** 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * self.feature_map_size * self.feature_map_size, params['ndf'] * 16)
        self.fc2 = nn.Linear(params['ndf'] * 16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = self.sigmoid(x)
        return out