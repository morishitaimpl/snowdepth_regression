import numpy as np
import sys, pathlib
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import torch.nn.functional as F

# 繰り返す回数
epochSize = 450

batchSize = 5

# def add_noise(inputs, noise_level=0.01):
#     noise = torch.randn_like(inputs) * noise_level
#     return inputs + noise


class neuralnetwork(nn.Module):
    def __init__(self, input_size=13, output_size=1):
        super(neuralnetwork, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        return self.model(x)
