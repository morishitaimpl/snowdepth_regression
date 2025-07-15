import numpy as np
import sys, pathlib
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 繰り返す回数
epochSize = 500

batchSize = 5

class neuralnetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super(SnowDepthPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)
