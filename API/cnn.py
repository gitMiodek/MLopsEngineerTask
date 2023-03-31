import torch.nn as nn
import torch
from torchvision.models.resnet import ResNet18_Weights
from torchvision import models

#inicialize model
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
# Load trained state dict
state_dict = torch.load('plane_car_state_dict.pt')
model.load_state_dict(state_dict)
# Turn model into inference mode
